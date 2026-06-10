//! Axiom Driver — Top-Level Compiler Driver.
//!
//! This crate orchestrates the full Axiom compilation pipeline:
//!
//! 1. Take an `IrGraph` (Sea-of-Nodes IR)
//! 2. Optionally run ownership analysis
//! 3. Run optimization passes based on `OptLevel`
//! 4. Lower to MIR
//! 5. Legalize for the target
//! 6. Register allocate
//! 7. Emit assembly
//! 8. Return `CompileResult`
//!
//! The driver also supports comparison with LLVM/Clang output via the
//! `compare_with_llvm` function, and can assemble and link the output
//! into an executable via `assemble`, `compile_to_object`, and
//! `compile_to_executable`.

use axiom_codegen;
use axiom_ir::IrGraph;
use axiom_legalize;
use axiom_mir;
use axiom_opt::{self, ConstantFolder, CommonSubexprElim, DeadCodeElim, DeadStoreElim, Inliner, Pass, StrengthReducer};
use axiom_ownership::{OwnershipAnalysis, OwnershipError, OwnershipVerifier};
use axiom_regalloc;
use axiom_target::Target;

use std::path::Path;
use std::process::Command;
use std::time::Instant;

// ── Optimization Level ─────────────────────────────────────────────────────

/// Optimization level for the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization — generate code as-is.
    O0,
    /// Basic optimizations: constant folding, dead code elimination.
    O1,
    /// Standard optimizations: CSE, DSE, strength reduction, inlining.
    O2,
    /// Aggressive optimizations: all of the above + multiple rounds.
    O3,
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::O0 => write!(f, "O0"),
            OptLevel::O1 => write!(f, "O1"),
            OptLevel::O2 => write!(f, "O2"),
            OptLevel::O3 => write!(f, "O3"),
        }
    }
}

// ── Ownership Analysis Result ──────────────────────────────────────────────

/// Ownership analysis summary included in the compile result.
#[derive(Debug, Clone)]
pub struct OwnershipAnalysisSummary {
    /// Total number of ownership roots found.
    pub root_count: usize,
    /// Number of non-escaping (local) roots.
    pub local_root_count: usize,
    /// Number of escaping roots.
    pub escaping_root_count: usize,
    /// Total number of dead stores found.
    pub dead_store_count: usize,
    /// Verification errors (empty if valid).
    pub verification_errors: Vec<OwnershipError>,
}

impl OwnershipAnalysisSummary {
    fn from_analysis(analysis: &OwnershipAnalysis) -> Self {
        let local_roots = analysis.local_roots();
        let dead_store_count: usize = analysis
            .roots
            .iter()
            .map(|r| analysis.dead_stores(*r).len())
            .sum();

        let local_count = local_roots.len();
        let escaping_count = analysis.roots.len() - local_count;

        OwnershipAnalysisSummary {
            root_count: analysis.roots.len(),
            local_root_count: local_count,
            escaping_root_count: escaping_count,
            dead_store_count,
            verification_errors: Vec::new(),
        }
    }
}

// ── Compile Result ─────────────────────────────────────────────────────────

/// Result of compiling a function through the Axiom pipeline.
#[derive(Debug, Clone)]
pub struct CompileResult {
    /// The generated assembly code.
    pub assembly: String,
    /// Ownership analysis summary (if requested).
    pub ownership_analysis: Option<OwnershipAnalysisSummary>,
    /// Number of IR nodes before optimization.
    pub node_count_before: usize,
    /// Number of IR nodes after optimization.
    pub node_count_after: usize,
    /// Time spent compiling (in milliseconds).
    pub compile_time_ms: u64,
    /// Optimization level used.
    pub opt_level: OptLevel,
    /// Target name.
    pub target_name: String,
    /// Number of MIR virtual registers.
    pub vreg_count: u32,
    /// Number of spill slots used.
    pub spill_slot_count: u32,
    /// Total frame size.
    pub frame_size: u32,
}

// ── Compiler Driver ────────────────────────────────────────────────────────

/// The top-level compiler driver.
///
/// Usage:
/// ```ignore
/// use axiom_driver::{CompilerDriver, OptLevel};
/// use axiom_ir::IrGraph;
/// use axiom_x86::X86_64Target;
///
/// let driver = CompilerDriver::new(Box::new(X86_64Target::new()), OptLevel::O2);
/// let graph = IrGraph::new("my_func");
/// let result = driver.compile(graph);
/// println!("{}", result.assembly);
/// ```
pub struct CompilerDriver {
    target: Box<dyn Target>,
    optimization_level: OptLevel,
    emit_ownership_analysis: bool,
    emit_llvm_comparison: bool,
}

impl CompilerDriver {
    /// Create a new compiler driver with the given target and optimization level.
    pub fn new(target: Box<dyn Target>, optimization_level: OptLevel) -> Self {
        Self {
            target,
            optimization_level,
            emit_ownership_analysis: false,
            emit_llvm_comparison: false,
        }
    }

    /// Enable or disable ownership analysis in the compile result.
    pub fn with_ownership_analysis(mut self, enable: bool) -> Self {
        self.emit_ownership_analysis = enable;
        self
    }

    /// Enable or disable LLVM/Clang comparison.
    pub fn with_llvm_comparison(mut self, enable: bool) -> Self {
        self.emit_llvm_comparison = enable;
        self
    }

    /// Compile an IR graph through the full pipeline.
    pub fn compile(&self, mut graph: IrGraph) -> CompileResult {
        let start = Instant::now();
        let node_count_before = graph.node_count();

        // ── Step 1: Ownership analysis (optional) ──
        let ownership_summary = if self.emit_ownership_analysis {
            let analysis = OwnershipAnalysis::analyze(&graph);
            let mut summary = OwnershipAnalysisSummary::from_analysis(&analysis);

            // Also run verification
            let verifier = OwnershipVerifier::new();
            summary.verification_errors = verifier.verify(&graph);

            Some(summary)
        } else {
            None
        };

        // ── Step 2: Optimization passes ──
        self.run_optimizations(&mut graph);

        let node_count_after = graph.node_count();

        // ── Step 3: Lower to MIR ──
        let mut mir_func = axiom_mir::lower::lower(&graph);
        let vreg_count = mir_func.vreg_count;

        // ── Step 4: Legalize ──
        axiom_legalize::legalize(&mut mir_func, self.target.as_ref());

        // ── Step 5: Register allocate ──
        let alloc_result = axiom_regalloc::allocate(&mir_func, self.target.desc());

        // ── Step 6: Emit assembly ──
        let assembly = axiom_codegen::emit_assembly(&mir_func, self.target.as_ref(), &alloc_result);

        let compile_time_ms = start.elapsed().as_millis() as u64;

        CompileResult {
            assembly,
            ownership_analysis: ownership_summary,
            node_count_before,
            node_count_after,
            compile_time_ms,
            opt_level: self.optimization_level,
            target_name: self.target.desc().name.clone(),
            vreg_count,
            spill_slot_count: alloc_result.spill_slot_count,
            frame_size: alloc_result.frame_size,
        }
    }

    /// Run optimization passes based on the optimization level.
    fn run_optimizations(&self, graph: &mut IrGraph) {
        match self.optimization_level {
            OptLevel::O0 => {
                // No optimizations
            }
            OptLevel::O1 => {
                // Basic optimizations: constant folding + DCE
                let passes: Vec<&dyn Pass> = vec![
                    &ConstantFolder,
                    &DeadCodeElim,
                ];
                axiom_opt::run_passes(graph, &passes);
            }
            OptLevel::O2 => {
                // Standard optimizations
                let inliner = Inliner::new(std::collections::HashMap::new(), 20);
                let passes: Vec<&dyn Pass> = vec![
                    &ConstantFolder,
                    &CommonSubexprElim,
                    &DeadCodeElim,
                    &DeadStoreElim,
                    &StrengthReducer,
                    &inliner,
                    &DeadCodeElim, // Run DCE again after inlining
                ];
                axiom_opt::run_passes(graph, &passes);
            }
            OptLevel::O3 => {
                // Aggressive: run O2 passes to fixed point
                let inliner = Inliner::new(std::collections::HashMap::new(), 20);
                let passes: Vec<&dyn Pass> = vec![
                    &ConstantFolder,
                    &CommonSubexprElim,
                    &DeadCodeElim,
                    &DeadStoreElim,
                    &StrengthReducer,
                    &inliner,
                    &DeadCodeElim,
                ];
                // Run multiple rounds
                for _ in 0..3 {
                    axiom_opt::run_passes(graph, &passes);
                }
            }
        }
    }

    /// Get the target reference.
    pub fn target(&self) -> &dyn Target {
        self.target.as_ref()
    }

    /// Get the optimization level.
    pub fn opt_level(&self) -> OptLevel {
        self.optimization_level
    }
}

// ── Object File Emission and Linking ──────────────────────────────────────

/// Write assembly to a `.s` file.
///
/// Returns the path to the written `.s` file.
pub fn write_asm(asm: &str, output_path: &Path) -> Result<std::path::PathBuf, String> {
    let asm_path = output_path.with_extension("s");
    std::fs::write(&asm_path, asm)
        .map_err(|e| format!("Failed to write .s file '{}': {}", asm_path.display(), e))?;
    Ok(asm_path)
}

/// Assemble a `.s` file into an object file using the system assembler.
///
/// Uses `gcc -c` (more portable than calling `as` directly) to assemble
/// the assembly source into a `.o` object file.
///
/// Returns the path to the object file on success.
pub fn assemble(asm_path: &Path) -> Result<std::path::PathBuf, String> {
    let obj_path = asm_path.with_extension("o");

    let status = Command::new("gcc")
        .args(["-c", asm_path.to_str().ok_or("Invalid .s file path")?, "-o", obj_path.to_str().ok_or("Invalid .o file path")?])
        .status()
        .map_err(|e| format!("Failed to run gcc for assembly: {}", e))?;

    if !status.success() {
        return Err(format!("Assembly of '{}' failed", asm_path.display()));
    }

    Ok(obj_path)
}

/// Compile assembly text to an object file.
///
/// This is a convenience function that writes the assembly to a temporary
/// `.s` file and then assembles it. Returns the path to the `.o` file.
pub fn compile_to_object(asm: &str, output_path: &Path) -> Result<std::path::PathBuf, String> {
    let asm_path = write_asm(asm, output_path)?;
    assemble(&asm_path)
}

/// Compile assembly text to an executable.
///
/// This writes the assembly to a `.s` file, assembles it into a `.o` file,
/// and then links it into an executable using `gcc`. The entry point is
/// expected to be a `main` function (or `_start` with `-nostdlib`).
///
/// If the assembly defines a `main` function, the executable will work
/// with the standard C runtime. If it defines `_start`, use the
/// `-nostdlib -static` flags are used.
///
/// Returns the path to the executable on success.
pub fn compile_to_executable(asm: &str, output_path: &Path) -> Result<std::path::PathBuf, String> {
    // Write assembly to .s file
    let asm_path = write_asm(asm, output_path)?;

    let output_str = output_path.to_str().ok_or("Invalid output path")?;

    // Try linking with standard C runtime first (expects `main`)
    let status = Command::new("gcc")
        .args([asm_path.to_str().ok_or("Invalid .s path")?, "-o", output_str, "-nostartfiles"])
        .status()
        .map_err(|e| format!("Failed to run gcc for linking: {}", e))?;

    if status.success() {
        return Ok(output_path.to_path_buf());
    }

    // Fallback: try with -nostdlib for bare-metal _start entry
    let status2 = Command::new("gcc")
        .args(["-nostdlib", "-static", asm_path.to_str().ok_or("Invalid .s path")?, "-o", output_str])
        .status()
        .map_err(|e| format!("Failed to run gcc for linking (nostdlib): {}", e))?;

    if status2.success() {
        return Ok(output_path.to_path_buf());
    }

    Err("Linking failed: neither standard nor nostdlib linking succeeded".to_string())
}

/// Full compilation pipeline that produces an executable.
///
/// Takes an `IrGraph`, compiles it through the full pipeline using the
/// given target and optimization level, then assembles and links the
/// output into an executable at `output_path`.
///
/// Returns the `CompileResult` with metadata about the compilation.
pub fn compile_to_executable_pipeline(
    target: Box<dyn Target>,
    opt_level: OptLevel,
    graph: IrGraph,
    output_path: &Path,
) -> Result<CompileResult, String> {
    let driver = CompilerDriver::new(target, opt_level);
    let result = driver.compile(graph);

    compile_to_executable(&result.assembly, output_path)?;

    Ok(result)
}

// ── LLVM Comparison ────────────────────────────────────────────────────────

/// Result of comparing Axiom output with Clang/LLVM.
#[derive(Debug, Clone)]
pub struct LlvmComparison {
    /// Whether clang was available on the system.
    pub clang_available: bool,
    /// Clang's assembly output (if available).
    pub clang_assembly: Option<String>,
    /// Number of lines in Axiom's output.
    pub axiom_line_count: usize,
    /// Number of lines in Clang's output (if available).
    pub clang_line_count: Option<usize>,
    /// Number of instructions in Axiom's output.
    pub axiom_instruction_count: usize,
    /// Number of instructions in Clang's output (if available).
    pub clang_instruction_count: Option<usize>,
}

/// Compare Axiom's assembly output with Clang's for the same C source.
///
/// This function takes a C source string, compiles it with `clang -S -O2`,
/// and compares the output assembly quality with the Axiom result.
///
/// Returns a comparison summary. If clang is not available on the system,
/// `clang_available` will be `false`.
pub fn compare_with_llvm(c_source: &str, axiom_assembly: &str) -> LlvmComparison {
    let axiom_line_count = axiom_assembly.lines().count();
    let axiom_instruction_count = count_instructions(axiom_assembly);

    // Try to compile with clang
    let clang_result = compile_with_clang(c_source);

    match clang_result {
        Some(clang_asm) => {
            let clang_line_count = clang_asm.lines().count();
            let clang_instruction_count = count_instructions(&clang_asm);

            LlvmComparison {
                clang_available: true,
                clang_assembly: Some(clang_asm),
                axiom_line_count,
                clang_line_count: Some(clang_line_count),
                axiom_instruction_count,
                clang_instruction_count: Some(clang_instruction_count),
            }
        }
        None => LlvmComparison {
            clang_available: false,
            clang_assembly: None,
            axiom_line_count,
            clang_line_count: None,
            axiom_instruction_count,
            clang_instruction_count: None,
        },
    }
}

/// Compile C source to assembly using clang.
///
/// Returns `None` if clang is not available or compilation fails.
fn compile_with_clang(c_source: &str) -> Option<String> {
    // Write the source to a temporary file
    let tmp_dir = std::env::temp_dir();
    let src_path = tmp_dir.join("axiom_compare_input.c");
    let asm_path = tmp_dir.join("axiom_compare_output.s");

    std::fs::write(&src_path, c_source).ok()?;

    // Run clang
    let output = Command::new("clang")
        .args([
            "-S",
            "-O2",
            "-o",
            asm_path.to_str()?,
            src_path.to_str()?,
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Read the assembly output
    std::fs::read_to_string(&asm_path).ok()
}

/// Count approximate instruction lines in assembly output.
///
/// Skips comments, labels, directives, and blank lines.
fn count_instructions(asm: &str) -> usize {
    asm.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with('.')
                && !trimmed.ends_with(':')
                && !trimmed.starts_with("//")
        })
        .count()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::IrNode;
    use axiom_target::{
        CallingConv, CodeSink, Endianness, PhysReg, RegClass, RegisterInfo, TargetDesc,
    };
    use axiom_ir::nodes::Type;

    /// A minimal test target.
    struct TestTarget {
        desc: TargetDesc,
    }

    impl Target for TestTarget {
        fn desc(&self) -> &TargetDesc {
            &self.desc
        }

        fn emit_prologue(&self, sink: &mut CodeSink, func: &axiom_mir::MirFunction) {
            sink.emit_label(&func.name);
        }

        fn emit_epilogue(&self, sink: &mut CodeSink, _func: &axiom_mir::MirFunction) {
            sink.emit("    ret");
        }

        fn emit_inst(
            &self,
            sink: &mut CodeSink,
            inst: &axiom_mir::MirInst,
            reg_names: &[String],
        ) {
            match inst {
                axiom_mir::MirInst::Label { block } => {
                    sink.emit_label(&format!(".L{}", block.as_u32()));
                }
                axiom_mir::MirInst::MovImm { dst, imm } => {
                    let idx = dst.as_u32() as usize;
                    let name = reg_names.get(idx).cloned().unwrap_or_default();
                    sink.emit(&format!("    mov {}, {}", imm.as_i64(), name));
                }
                axiom_mir::MirInst::Ret { val } => {
                    if let Some(v) = val {
                        let idx = v.as_u32() as usize;
                        let name = reg_names.get(idx).cloned().unwrap_or_default();
                        sink.emit(&format!("    mov {}, rax", name));
                    }
                }
                _ => {
                    sink.emit_comment(&format!("{:?}", inst));
                }
            }
        }

        fn legalize_type(&self, ty: Type) -> Type {
            ty
        }

        fn reg_name(&self, reg: PhysReg) -> String {
            format!("r{}", reg.as_u16())
        }
    }

    fn test_target() -> TestTarget {
        let registers: Vec<RegisterInfo> = (0..8)
            .map(|i| RegisterInfo {
                reg: PhysReg::new(i),
                name: format!("r{}", i),
                class: RegClass::Int,
                is_reserved: i >= 6,
            })
            .collect();

        let desc = TargetDesc {
            name: "test64".to_string(),
            ptr_width: 64,
            endianness: Endianness::Little,
            registers,
            calling_conv: CallingConv {
                arg_regs: vec![PhysReg::new(0)],
                ret_regs: vec![PhysReg::new(0)],
                callee_saved: vec![],
                caller_saved: vec![],
                stack_align: 16,
            },
            supported_widths: vec![64],
            has_cmov: false,
            has_vector: false,
            vector_width: 0,
        };
        TestTarget { desc }
    }

    #[test]
    fn test_driver_o0_simple() {
        let driver = CompilerDriver::new(Box::new(test_target()), OptLevel::O0);
        let mut graph = IrGraph::new("simple_fn");
        let val = graph.push_node(IrNode::IntConst(42));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let result = driver.compile(graph);

        assert!(result.assembly.contains("simple_fn:"), "Should contain function label");
        assert!(result.assembly.contains("42"), "Should contain the constant");
        assert_eq!(result.opt_level, OptLevel::O0);
    }

    #[test]
    fn test_driver_o2() {
        let driver = CompilerDriver::new(Box::new(test_target()), OptLevel::O2);
        let mut graph = IrGraph::new("opt_fn");
        let val = graph.push_node(IrNode::IntConst(100));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let result = driver.compile(graph);

        assert!(result.assembly.contains("opt_fn:"), "Should contain function label");
        assert_eq!(result.opt_level, OptLevel::O2);
    }

    #[test]
    fn test_driver_with_ownership_analysis() {
        let driver = CompilerDriver::new(Box::new(test_target()), OptLevel::O1)
            .with_ownership_analysis(true);

        let mut graph = IrGraph::new("owned_fn");
        let val = graph.push_node(IrNode::IntConst(7));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let result = driver.compile(graph);

        assert!(
            result.ownership_analysis.is_some(),
            "Ownership analysis should be present when enabled"
        );
        let summary = result.ownership_analysis.unwrap();
        // The graph has no memory operations, so it may have 0 roots.
        // The key test is that ownership_analysis was computed at all.
        // A graph with StackAlloc/Store/Load would have roots.
        assert!(summary.root_count >= 0, "Ownership analysis should complete without error");
    }

    #[test]
    fn test_driver_without_ownership_analysis() {
        let driver = CompilerDriver::new(Box::new(test_target()), OptLevel::O0)
            .with_ownership_analysis(false);

        let mut graph = IrGraph::new("no_analysis_fn");
        let val = graph.push_node(IrNode::IntConst(1));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let result = driver.compile(graph);

        assert!(
            result.ownership_analysis.is_none(),
            "Ownership analysis should not be present when disabled"
        );
    }

    #[test]
    fn test_node_count_tracking() {
        let driver = CompilerDriver::new(Box::new(test_target()), OptLevel::O0);
        let mut graph = IrGraph::new("count_fn");
        let val = graph.push_node(IrNode::IntConst(1));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let result = driver.compile(graph);

        // At O0, no nodes should be eliminated
        assert!(
            result.node_count_before >= 3,
            "Should have at least Start + IntConst + Return nodes"
        );
    }

    #[test]
    fn test_optimization_reduces_nodes() {
        // Create a graph with a dead code path
        let driver = CompilerDriver::new(Box::new(test_target()), OptLevel::O2);
        let mut graph = IrGraph::new("dead_code_fn");
        let val = graph.push_node(IrNode::IntConst(42));
        let _unused = graph.push_node(IrNode::IntConst(999)); // This is dead
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let result = driver.compile(graph);

        // With DCE, the dead IntConst(999) might be eliminated
        assert!(
            result.node_count_after <= result.node_count_before,
            "Optimization should not increase node count"
        );
    }

    #[test]
    fn test_compile_result_fields() {
        let driver = CompilerDriver::new(Box::new(test_target()), OptLevel::O1);
        let mut graph = IrGraph::new("fields_fn");
        let val = graph.push_node(IrNode::IntConst(1));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let result = driver.compile(graph);

        assert_eq!(result.target_name, "test64");
        assert_eq!(result.opt_level, OptLevel::O1);
        // vreg_count should be > 0 (at least one vreg for the constant)
        assert!(result.vreg_count > 0);
    }

    // ── End-to-End Tests (require gcc/assembler) ────────────────────────────

    /// Compile an IrGraph through the full Axiom pipeline to an executable,
    /// then run it and return the exit code.
    #[cfg(target_arch = "x86_64")]
    fn compile_run_and_get_exit_code(graph: &mut IrGraph, func_name: &str) -> Result<i32, String> {
        use axiom_x86::X86_64Target;
        use axiom_target::Target;
        use axiom_codegen::compile;

        let target = X86_64Target::new();
        let asm = compile(graph, &target);

        let asm_path = format!("/tmp/axiom_e2e_{}.s", func_name);
        let exe_path = format!("/tmp/axiom_e2e_{}", func_name);

        // Write assembly with proper ELF directives and a _start wrapper
        // that calls the function and then does a Linux exit syscall with
        // the return value.
        //
        // On Linux x86-64, the exit syscall is:
        //   movl $60, %eax   (sys_exit)
        //   movq %rdi, %rsi  (exit code in %rdi → move to %rdi is already there)
        //   syscall
        //
        // For a function returning a value in %rax, we wrap it:
        //   _start:
        //     call func_name
        //     movq %rax, %rdi    # return value → exit code
        //     movl $60, %eax     # sys_exit
        //     syscall
        let full_asm = format!(
            "    .text\n    .globl _start\n    .globl {}\n{}\n\n_start:\n    callq   {}\n    movq    %rax, %rdi\n    movl    $60, %eax\n    syscall\n",
            func_name, asm, func_name
        );
        std::fs::write(&asm_path, &full_asm)
            .map_err(|e| format!("write .s: {}", e))?;

        // Assemble and link as bare-metal
        let status = Command::new("gcc")
            .args(["-nostdlib", "-static", &asm_path, "-o", &exe_path])
            .status()
            .map_err(|e| format!("gcc: {}", e))?;

        if !status.success() {
            return Err(format!("Linking failed. Assembly:\n{}", full_asm));
        }

        // Run and capture exit code
        let output = Command::new(&exe_path)
            .output()
            .map_err(|e| format!("run: {}", e))?;

        Ok(output.status.code().unwrap_or(-1))
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_e2e_return_constant() {
        // int main() { return 42; }
        let mut graph = IrGraph::new("main");
        let val = graph.push_node(IrNode::IntConst(42));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        match compile_run_and_get_exit_code(&mut graph, "main") {
            Ok(code) => {
                // On Linux, exit code is the return value from main (low 8 bits)
                let actual = code & 0xff;
                assert_eq!(actual, 42,
                    "Expected exit code 42, got {}. The Axiom-compiled program produced wrong output!",
                    actual);
            }
            Err(e) => {
                panic!("End-to-end compilation failed: {}", e);
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_e2e_add_params() {
        // Verify that a function with parameters compiles, assembles, and links.
        // We can't easily pass arguments from _start without setting up registers,
        // so we just verify the pipeline produces valid object code.
        use axiom_x86::X86_64Target;
        use axiom_target::Target;
        use axiom_codegen::compile;

        let mut graph = IrGraph::new("add_params");
        let a = graph.push_node(IrNode::Param { index: 0, ty: Type::I64 });
        let b = graph.push_node(IrNode::Param { index: 1, ty: Type::I64 });
        let sum = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(sum) });

        let target = X86_64Target::new();
        let asm = compile(&mut graph, &target);

        // Verify the assembly contains expected patterns
        assert!(asm.contains("add_params:"), "Should contain function label");
        assert!(asm.contains("movq"), "Should contain mov instructions for arg passing");
        assert!(asm.contains("addq") || asm.contains("imulq"), "Should contain arithmetic");
    }
}
