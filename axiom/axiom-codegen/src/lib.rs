//! Axiom Codegen — Machine Code Generation.
//!
//! This crate generates assembly code from a MIR function using a target's
//! code emission interface. It applies register allocation results to map
//! virtual registers to physical registers (or spill slot references),
//! then emits the prologue, all instructions, and the epilogue.
//!
//! It also provides a `compile` convenience function that runs the full
//! pipeline: IR → MIR → legalize → regalloc → assembly.

use axiom_ir::IrGraph;
use axiom_mir::{MirFunction, VReg};
use axiom_regalloc::{self, RegAllocResult};
use axiom_target::{CodeSink, PhysReg, PrologueInfo, Target};

/// Generate assembly code for a MIR function, applying register allocation.
///
/// This uses the target's emission interface to produce assembly text.
/// Virtual registers are mapped to physical register names or spill-slot
/// addresses using the allocation result.
pub fn emit_assembly(
    func: &MirFunction,
    target: &dyn Target,
    alloc: &RegAllocResult,
) -> String {
    let mut sink = CodeSink::new();

    // Build the reg_names table: for each VReg, produce the name that
    // the target's emit_inst should use (physical register or spill slot).
    let reg_names = build_reg_names(func, target, alloc);

    // Build PrologueInfo from the register allocation result
    let info = build_prologue_info(func, target, alloc);

    // Emit prologue with register allocation info (callee-saved, arg moves)
    target.emit_prologue_with_info(&mut sink, func, &info);

    // Emit all instructions with register allocation applied
    for block in &func.blocks {
        for inst in &block.insts {
            target.emit_inst(&mut sink, inst, &reg_names);
        }
    }

    // Always emit the epilogue — Ret instructions jump to the epilogue label
    target.emit_epilogue_with_info(&mut sink, func, &info);

    // Post-process: replace RET_EPILOGUE_JUMP placeholders with actual jumps
    let epilogue_label = format!(".L{}_epilogue", func.name);
    for line in &mut sink.lines {
        if line.contains("# RET_EPILOGUE_JUMP") {
            *line = format!("    jmp     {}", epilogue_label);
        }
    }

    // Apply peephole optimizations before returning
    let optimized = apply_peephole(&sink.lines);
    optimized.join("\n")
}

/// Build the register name table that maps each VReg to its final name.
///
/// For VRegs assigned to a physical register, we use the target's
/// `reg_name()` method. For spilled VRegs, we generate a stack-slot
/// reference like `[rbp - offset]` (target-specific).
fn build_reg_names(
    func: &MirFunction,
    target: &dyn Target,
    alloc: &RegAllocResult,
) -> Vec<String> {
    let desc = target.desc();
    let _stack_align = desc.calling_conv.stack_align.max(1);
    let slot_size: u32 = 8; // pointer-sized slots

    let mut names = Vec::new();
    for i in 0..func.vreg_count {
        let vreg = VReg::new(i);
        if let Some(&phys_reg) = alloc.allocation.get(&vreg) {
            names.push(target.reg_name(phys_reg));
        } else if let Some(&slot) = alloc.spills.get(&vreg) {
            // Spill slot: generate a stack reference
            let offset = (slot + 1) * slot_size;
            // Use target-specific stack reference format
            let ptr_name = target.reg_name(
                desc.registers
                    .iter()
                    .find(|r| r.name.contains("bp") || r.name.contains("fp"))
                    .map(|r| r.reg)
                    .unwrap_or(axiom_target::PhysReg::new(0)),
            );
            names.push(format!("{}-{}", ptr_name, offset));
        } else {
            // VReg not allocated and not spilled — use default name
            names.push(target.vreg_name(vreg));
        }
    }
    names
}

/// Build the `PrologueInfo` from a `RegAllocResult`.
///
/// This extracts:
/// - The frame size from regalloc
/// - Which callee-saved registers are actually used (have a VReg allocated to them)
/// - The mapping from argument registers to assigned registers for function params
fn build_prologue_info(
    func: &MirFunction,
    target: &dyn Target,
    alloc: &RegAllocResult,
) -> PrologueInfo {
    let desc = target.desc();
    let callee_saved = &desc.calling_conv.callee_saved;
    let arg_regs = &desc.calling_conv.arg_regs;

    // Find which callee-saved registers are actually used
    let mut used_callee_saved: Vec<PhysReg> = Vec::new();
    for &cs_reg in callee_saved {
        let is_used = alloc.allocation.values().any(|&r| r == cs_reg);
        if is_used {
            used_callee_saved.push(cs_reg);
        }
    }

    // Build argument moves: for each parameter, map arg_reg → assigned_reg
    let mut arg_moves: Vec<(PhysReg, PhysReg)> = Vec::new();
    for (i, &param_vreg) in func.params.iter().enumerate() {
        if i < arg_regs.len() {
            let arg_reg = arg_regs[i];
            if let Some(&assigned_reg) = alloc.allocation.get(&param_vreg) {
                arg_moves.push((arg_reg, assigned_reg));
            }
        }
        // Stack arguments (i >= arg_regs.len()) would need RBP+offset loads;
        // these are not handled here yet.
    }

    PrologueInfo {
        frame_size: alloc.frame_size,
        used_callee_saved,
        arg_moves,
    }
}

/// Full compilation pipeline for a single function:
///
/// 1. Run optimization passes on IR (currently a no-op placeholder;
///    the driver orchestrates this)
/// 2. Lower IR to MIR
/// 3. Legalize MIR for the target
/// 4. Register allocate
/// 5. Emit assembly
pub fn compile(graph: &mut IrGraph, target: &dyn Target) -> String {
    // Step 1: Lower IR to MIR
    let mut mir_func = axiom_mir::lower::lower(graph);

    // Step 2: Legalize for the target
    axiom_legalize::legalize(&mut mir_func, target);

    // Step 3: Register allocate
    let alloc_result = axiom_regalloc::allocate(&mir_func, target.desc());

    // Step 4: Insert spill/reload code for spilled VRegs
    let alloc_result = axiom_regalloc::insert_spill_code(&mut mir_func, &alloc_result, target.desc());

    // Step 5: Emit assembly (peephole is applied inside emit_assembly)
    emit_assembly(&mir_func, target, &alloc_result)
}

/// Compile a MIR function (already lowered and legalized) to assembly.
///
/// This is a convenience function that just does regalloc + emission.
pub fn compile_mir(func: &MirFunction, target: &dyn Target) -> String {
    let alloc_result = axiom_regalloc::allocate(func, target.desc());
    // Note: We can't insert spill code here because `func` is immutable.
    // Use `compile` for the full pipeline with spill code insertion.
    emit_assembly(func, target, &alloc_result)
}

// ── Peephole Optimizer ──────────────────────────────────────────────────

/// Apply x86-64 peephole optimizations to assembly lines.
///
/// This post-processes the assembly lines after register allocation
/// and emission, applying the following transformations:
///
/// (a) Zero idiom: `movq $0, %reg` → `xorq %reg, %reg`
///     (2 bytes shorter, breaks dependency chain)
///
/// (b) Self-move elimination: `movq %reg, %reg` → remove (no-op)
///
/// (c) Test idiom: `cmpq $0, %reg` → `testq %reg, %reg`
///     (1 byte shorter, sets flags equivalently)
///
/// (d) Redundant mov elimination: If a mov instruction moves a value
///     to a register that already holds it (from the previous instruction),
///     remove the redundant mov.
pub fn apply_peephole(lines: &[String]) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(lines.len());

    // Track the destination register written by the previous instruction,
    // and the source that was written to it. Used for pattern (d).
    let mut prev_dst: Option<String> = None;
    let mut prev_src_for_dst: Option<String> = None;

    for line in lines {
        let trimmed = line.trim();

        // (a) Zero idiom: movq $0, %reg → xorq %reg, %reg
        // Match: movq    $0, %reg  (AT&T syntax)
        if let Some(rest) = trimmed.strip_prefix("movq") {
            let rest = rest.trim();
            if rest.starts_with("$0,") || rest.starts_with("$0 ,") {
                let parts: Vec<&str> = rest.splitn(2, ',').collect();
                if parts.len() == 2 {
                    let dst = parts[1].trim().to_string();
                    result.push(format!("    xorq    {}, {}", dst, dst));
                    prev_dst = Some(dst.clone());
                    prev_src_for_dst = Some("$0".to_string());
                    continue;
                }
            }
        }

        // (b) Self-move elimination: movq %reg, %reg → remove
        if let Some(rest) = trimmed.strip_prefix("movq") {
            let rest = rest.trim();
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() == 2 {
                let src = parts[0].trim();
                let dst = parts[1].trim();
                if src == dst && src.starts_with('%') {
                    // Self-move — eliminate
                    continue;
                }

                // (d) Redundant mov elimination: if the previous instruction
                // wrote a value from `src` into `dst`, and this instruction
                // is `movq src, dst`, then `dst` already holds the value from
                // `src` and this mov is redundant.
                // Also: if previous instruction was `movq %src, %dst` and
                // this is `movq %src, %dst` (same mov twice), eliminate.
                if src.starts_with('%') && dst.starts_with('%') {
                    if let (Some(ref p_dst), Some(ref p_src)) = (&prev_dst, &prev_src_for_dst) {
                        // If the previous instruction moved the same source to
                        // the same destination, this mov is redundant.
                        if *p_src == src && *p_dst == dst {
                            continue;
                        }
                    }
                }

                // Not eliminated — record for next iteration
                prev_dst = Some(dst.to_string());
                prev_src_for_dst = Some(src.to_string());
                result.push(line.clone());
                continue;
            }
        }

        // (c) Test idiom: cmpq $0, %reg → testq %reg, %reg
        if let Some(rest) = trimmed.strip_prefix("cmpq") {
            let rest = rest.trim();
            if rest.starts_with("$0,") || rest.starts_with("$0 ,") {
                let parts: Vec<&str> = rest.splitn(2, ',').collect();
                if parts.len() == 2 {
                    let reg = parts[1].trim();
                    result.push(format!("    testq   {}, {}", reg, reg));
                    prev_dst = None; // testq doesn't write to a register
                    prev_src_for_dst = None;
                    continue;
                }
            }
        }

        // Also handle movsd self-moves (common after regalloc)
        if let Some(rest) = trimmed.strip_prefix("movsd") {
            let rest = rest.trim();
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() == 2 {
                let src = parts[0].trim();
                let dst = parts[1].trim();
                if src == dst && src.starts_with('%') {
                    continue;
                }
                // Redundant movsd elimination
                if src.starts_with('%') && dst.starts_with('%') {
                    if let (Some(ref p_dst), Some(ref p_src)) = (&prev_dst, &prev_src_for_dst) {
                        if *p_src == src && *p_dst == dst {
                            continue;
                        }
                    }
                }
                prev_dst = Some(dst.to_string());
                prev_src_for_dst = Some(src.to_string());
                result.push(line.clone());
                continue;
            }
        }

        // Also handle movaps self-moves (common after regalloc)
        if let Some(rest) = trimmed.strip_prefix("movaps") {
            let rest = rest.trim();
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() == 2 {
                let src = parts[0].trim();
                let dst = parts[1].trim();
                if src == dst && src.starts_with('%') {
                    continue;
                }
                prev_dst = Some(dst.to_string());
                prev_src_for_dst = Some(src.to_string());
                result.push(line.clone());
                continue;
            }
        }

        // For other instructions, try to extract a destination register
        // for pattern (d) tracking. We look for common patterns like:
        // `op %src, %dst` or `op %src1, %src2, %dst`
        update_prev_dst(trimmed, &mut prev_dst, &mut prev_src_for_dst);

        result.push(line.clone());
    }

    result
}

/// Try to extract the destination register and source from an instruction line,
/// for use in redundant-mov tracking (pattern d).
fn update_prev_dst(trimmed: &str, prev_dst: &mut Option<String>, prev_src_for_dst: &mut Option<String>) {
    // For instructions of the form `op    %src, %dst` (2-operand AT&T),
    // the last %reg is the destination.
    // For instructions of the form `op    %src1, %src2, %dst` (3-operand),
    // the last %reg is the destination.

    // Skip labels, directives, comments
    if trimmed.is_empty() || trimmed.starts_with('.') || trimmed.starts_with('#') || trimmed.ends_with(':') {
        return;
    }

    // Extract all %reg references from the operands part
    let parts: Vec<&str> = if let Some(space_idx) = trimmed.find(' ') {
        let operands = &trimmed[space_idx..];
        operands.split(',').map(|s| s.trim()).collect()
    } else {
        return;
    };

    if parts.is_empty() {
        return;
    }

    // The last operand is typically the destination in AT&T syntax
    let last = parts.last().unwrap();
    if last.starts_with('%') {
        *prev_dst = Some(last.to_string());
        // The source is typically the first operand (for 2-operand instructions)
        if parts.len() >= 2 {
            let first = parts[0].trim();
            if first.starts_with('%') {
                *prev_src_for_dst = Some(first.to_string());
            } else {
                *prev_src_for_dst = None;
            }
        } else {
            *prev_src_for_dst = None;
        }
    } else {
        // Not a register destination — reset tracking
        *prev_dst = None;
        *prev_src_for_dst = None;
    }
}

/// Run x86-64 peephole optimizations on assembly output.
///
/// This is the public API that takes a complete assembly string,
/// splits it into lines, applies peephole, and returns the result.
pub fn peephole_x86_64(asm: String) -> String {
    let lines: Vec<String> = asm.lines().map(|l| l.to_string()).collect();
    apply_peephole(&lines).join("\n")
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;
    use axiom_ir::{IrGraph, IrNode};
    use axiom_mir::Imm64;
    use axiom_target::{
        CallingConv, Endianness, PhysReg, RegClass, RegisterInfo, TargetDesc,
    };

    /// A minimal test target that produces simple assembly output.
    struct TestTarget {
        desc: TargetDesc,
    }

    impl Target for TestTarget {
        fn desc(&self) -> &TargetDesc {
            &self.desc
        }

        fn emit_prologue(&self, sink: &mut CodeSink, func: &axiom_mir::MirFunction) {
            sink.emit_label(&func.name);
            sink.emit("    push rbp");
            sink.emit("    mov rsp, rbp");
        }

        fn emit_epilogue(&self, sink: &mut CodeSink, _func: &axiom_mir::MirFunction) {
            sink.emit("    mov rbp, rsp");
            sink.emit("    pop rbp");
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
                    let dst_name = rn(reg_names, *dst);
                    sink.emit(&format!("    mov {}, {}", imm.as_i64(), dst_name));
                }
                axiom_mir::MirInst::Add { dst, lhs, rhs } => {
                    let dst_name = rn(reg_names, *dst);
                    let lhs_name = rn(reg_names, *lhs);
                    let rhs_name = rn(reg_names, *rhs);
                    sink.emit(&format!("    add {}, {}, {}", lhs_name, rhs_name, dst_name));
                }
                axiom_mir::MirInst::Ret { val } => {
                    if let Some(v) = val {
                        sink.emit(&format!("    mov {}, rax", rn(reg_names, *v)));
                    }
                    sink.emit("    ret");
                }
                _ => {
                    sink.emit_comment(&format!("unhandled: {:?}", inst));
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

    fn rn(reg_names: &[String], vreg: VReg) -> String {
        let idx = vreg.as_u32() as usize;
        if idx < reg_names.len() {
            reg_names[idx].clone()
        } else {
            format!("v{}", vreg.as_u32())
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
                callee_saved: vec![PhysReg::new(4), PhysReg::new(5)],
                caller_saved: vec![PhysReg::new(0), PhysReg::new(1), PhysReg::new(2), PhysReg::new(3)],
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
    fn test_emit_simple_function() {
        let mut func = axiom_mir::MirFunction::new("add42");
        let _block = func.new_block();
        let v0 = func.alloc_vreg();
        let v1 = func.alloc_vreg();

        func.params.push(v0);
        func.blocks[0].insts.push(axiom_mir::MirInst::MovImm {
            dst: v1,
            imm: Imm64::new(42),
        });
        func.blocks[0]
            .insts
            .push(axiom_mir::MirInst::Add {
                dst: v1,
                lhs: v0,
                rhs: v1,
            });
        func.blocks[0]
            .insts
            .push(axiom_mir::MirInst::Ret { val: Some(v1) });

        let target = test_target();
        let result = compile_mir(&func, &target);

        assert!(
            result.contains("add42:"),
            "Assembly should contain function label"
        );
        assert!(
            result.contains("ret"),
            "Assembly should contain return instruction"
        );
    }

    #[test]
    fn test_full_compile_pipeline() {
        let mut graph = IrGraph::new("test_fn");
        let val = graph.push_node(IrNode::IntConst(42));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let target = test_target();
        let result = compile(&mut graph, &target);

        assert!(
            result.contains("test_fn:"),
            "Assembly should contain function label"
        );
        assert!(
            result.contains("42"),
            "Assembly should contain the constant 42"
        );
    }

    // ── Peephole Tests ────────────────────────────────────────────────────

    #[test]
    fn test_peephole_zero_idiom() {
        let lines: Vec<String> = vec![
            "    movq    $0, %rax".to_string(),
            "    retq".to_string(),
        ];
        let result = apply_peephole(&lines);
        assert!(result[0].contains("xorq"), "movq $0 should become xorq");
        assert!(!result.iter().any(|l| l.contains("movq    $0")), "movq $0 should be eliminated");
    }

    #[test]
    fn test_peephole_self_move_elimination() {
        let lines: Vec<String> = vec![
            "    movq    %rax, %rax".to_string(),
            "    retq".to_string(),
        ];
        let result = apply_peephole(&lines);
        assert!(!result.iter().any(|l| l.contains("movq    %rax, %rax")), "Self-move should be eliminated");
        assert!(result.iter().any(|l| l.contains("retq")), "ret should remain");
    }

    #[test]
    fn test_peephole_test_idiom() {
        let lines: Vec<String> = vec![
            "    cmpq    $0, %rax".to_string(),
            "    jne .L1".to_string(),
        ];
        let result = apply_peephole(&lines);
        assert!(result[0].contains("testq"), "cmpq $0 should become testq");
        assert!(!result.iter().any(|l| l.contains("cmpq    $0")), "cmpq $0 should be eliminated");
    }

    #[test]
    fn test_peephole_movsd_self_move() {
        let lines: Vec<String> = vec![
            "    movsd   %xmm0, %xmm0".to_string(),
            "    addsd %xmm1, %xmm0".to_string(),
        ];
        let result = apply_peephole(&lines);
        assert!(!result.iter().any(|l| l.contains("movsd   %xmm0, %xmm0")), "movsd self-move should be eliminated");
        assert!(result.iter().any(|l| l.contains("addsd")), "addsd should remain");
    }

    #[test]
    fn test_peephole_preserves_normal_moves() {
        let lines: Vec<String> = vec![
            "    movq    %rax, %rbx".to_string(),
            "    movq    $5, %rax".to_string(),
        ];
        let result = apply_peephole(&lines);
        assert!(result.iter().any(|l| l.contains("movq    %rax, %rbx")), "Normal move should be preserved");
        assert!(result.iter().any(|l| l.contains("movq    $5, %rax")), "Non-zero movq should be preserved");
    }

    #[test]
    fn test_peephole_redundant_mov_elimination() {
        // Two consecutive movq with same source and destination: second is redundant
        let lines: Vec<String> = vec![
            "    movq    %rax, %rbx".to_string(),
            "    movq    %rax, %rbx".to_string(),
            "    retq".to_string(),
        ];
        let result = apply_peephole(&lines);
        // Only one movq %rax, %rbx should remain
        let mov_count = result.iter().filter(|l| l.contains("movq    %rax, %rbx")).count();
        assert_eq!(mov_count, 1, "Redundant mov should be eliminated, got {:?}",
                   result.iter().filter(|l| l.contains("movq")).collect::<Vec<_>>());
    }

    #[test]
    fn test_peephole_redundant_mov_different_src() {
        // Two movq with different sources: both should be kept
        let lines: Vec<String> = vec![
            "    movq    %rax, %rbx".to_string(),
            "    movq    %rcx, %rbx".to_string(),
            "    retq".to_string(),
        ];
        let result = apply_peephole(&lines);
        let mov_count = result.iter().filter(|l| l.contains("movq")).count();
        assert_eq!(mov_count, 2, "Both different movs should be kept");
    }

    #[test]
    fn test_peephole_backward_compat_string_api() {
        let asm = "    movq    $0, %rax\n    retq".to_string();
        let result = peephole_x86_64(asm);
        assert!(result.contains("xorq"), "movq $0 should become xorq via peephole_x86_64");
    }
}
