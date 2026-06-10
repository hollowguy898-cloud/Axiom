//! Axiom Target — target description system and code emission traits.
//!
//! This crate defines the `Target` trait that every backend must implement,
//! along with supporting types for physical registers, calling conventions,
//! and code emission. Each backend crate (axiom-x86, axiom-aarch64, etc.)
//! provides a concrete implementation of `Target`.

use axiom_ir::nodes::Type;
use axiom_mir::{MirFunction, MirInst, VReg};

// ── Prologue Info ───────────────────────────────────────────────────────

/// Information needed by the prologue/epilogue to correctly save/restore
/// callee-saved registers and move function arguments from calling-convention
/// registers to their allocated VRegs.
///
/// This struct is populated by the codegen crate from `RegAllocResult` and
/// passed to the target's `emit_prologue_with_info` / `emit_epilogue_with_info`
/// methods. It lives in `axiom-target` to avoid a circular dependency between
/// `axiom-target` and `axiom-regalloc`.
#[derive(Debug, Clone, Default)]
pub struct PrologueInfo {
    /// Total frame size (in bytes) from the register allocator.
    /// Includes spill slots, rounded to stack alignment.
    pub frame_size: u32,

    /// Callee-saved registers that are actually used by the function
    /// (i.e., some VReg is allocated to them) and therefore need to be
    /// saved in the prologue and restored in the epilogue.
    pub used_callee_saved: Vec<PhysReg>,

    /// Mapping from argument registers to the physical registers that the
    /// register allocator assigned to the corresponding parameter VRegs.
    /// Each entry is `(arg_reg, assigned_reg)`. If `arg_reg == assigned_reg`
    /// the move can be skipped.
    pub arg_moves: Vec<(PhysReg, PhysReg)>,
}

// ── Physical Register ──────────────────────────────────────────────────

/// Physical register identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysReg(pub u16);

impl PhysReg {
    pub const fn new(id: u16) -> Self {
        PhysReg(id)
    }

    pub const fn as_u16(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for PhysReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "preg{}", self.0)
    }
}

// ── Register Class ─────────────────────────────────────────────────────

/// Register class (integer, float, special).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    Int,
    Float,
    Special,
}

// ── Endianness ─────────────────────────────────────────────────────────

/// Target byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

// ── Register Info ──────────────────────────────────────────────────────

/// Information about a single physical register.
#[derive(Debug, Clone)]
pub struct RegisterInfo {
    pub reg: PhysReg,
    pub name: String,
    pub class: RegClass,
    pub is_reserved: bool,
}

// ── Calling Convention ─────────────────────────────────────────────────

/// Calling convention description.
#[derive(Debug, Clone)]
pub struct CallingConv {
    /// Argument registers in order.
    pub arg_regs: Vec<PhysReg>,
    /// Return value registers.
    pub ret_regs: Vec<PhysReg>,
    /// Callee-saved registers.
    pub callee_saved: Vec<PhysReg>,
    /// Caller-saved (volatile) registers.
    pub caller_saved: Vec<PhysReg>,
    /// Stack alignment in bytes.
    pub stack_align: u32,
}

// ── Target Description ─────────────────────────────────────────────────

/// Describes the target machine.
#[derive(Debug, Clone)]
pub struct TargetDesc {
    /// Target name (e.g., "x86_64", "aarch64").
    pub name: String,
    /// Pointer width in bits.
    pub ptr_width: u32,
    /// Endianness.
    pub endianness: Endianness,
    /// Available physical registers.
    pub registers: Vec<RegisterInfo>,
    /// Calling convention.
    pub calling_conv: CallingConv,
    /// Supported type widths in bits.
    pub supported_widths: Vec<u32>,
    /// Whether the target has conditional moves.
    pub has_cmov: bool,
    /// Whether the target has SSE/AVX-style vector support.
    pub has_vector: bool,
    /// Vector register width in bits (0 if no vector support).
    pub vector_width: u32,
}

// ── Code Sink ──────────────────────────────────────────────────────────

/// Code emission sink — collects assembly text line by line.
#[derive(Debug, Clone)]
pub struct CodeSink {
    pub lines: Vec<String>,
    pub current_section: String,
}

impl CodeSink {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            current_section: ".text".to_string(),
        }
    }

    /// Emit a raw line of assembly.
    pub fn emit(&mut self, line: &str) {
        self.lines.push(line.to_string());
    }

    /// Emit a label (with trailing colon).
    pub fn emit_label(&mut self, label: &str) {
        self.lines.push(format!("{}:", label));
    }

    /// Emit a comment.
    pub fn emit_comment(&mut self, comment: &str) {
        self.lines.push(format!("# {}", comment));
    }

    /// Switch to a different section.
    pub fn section(&mut self, section: &str) {
        self.current_section = section.to_string();
    }
}

impl Default for CodeSink {
    fn default() -> Self {
        Self::new()
    }
}

// ── Target Trait ───────────────────────────────────────────────────────

/// The main Target trait — every backend implements this.
pub trait Target {
    /// Return the static target description.
    fn desc(&self) -> &TargetDesc;

    /// Emit function prologue.
    fn emit_prologue(&self, sink: &mut CodeSink, func: &MirFunction);

    /// Emit function epilogue.
    fn emit_epilogue(&self, sink: &mut CodeSink, func: &MirFunction);

    /// Emit a single MIR instruction.
    ///
    /// `reg_names` maps VReg indices to the physical register or spill-slot
    /// names assigned by the register allocator.
    fn emit_inst(&self, sink: &mut CodeSink, inst: &MirInst, reg_names: &[String]);

    /// Legalize (promote / split) a type for this target.
    fn legalize_type(&self, ty: Type) -> Type;

    /// Return the assembly name for a physical register.
    fn reg_name(&self, reg: PhysReg) -> String;

    /// Return the display name for a virtual register.
    /// The default shows `vN`; backends may override after regalloc.
    fn vreg_name(&self, vreg: VReg) -> String {
        format!("v{}", vreg.as_u32())
    }

    /// Emit function prologue with register allocation info.
    ///
    /// Backends should override this to properly save callee-saved registers
    /// and move function arguments from calling-convention registers to their
    /// allocated physical registers. The default falls back to `emit_prologue`.
    fn emit_prologue_with_info(
        &self,
        sink: &mut CodeSink,
        func: &MirFunction,
        info: &PrologueInfo,
    ) {
        let _ = info;
        self.emit_prologue(sink, func);
    }

    /// Emit function epilogue with register allocation info.
    ///
    /// Backends should override this to properly restore callee-saved registers.
    /// The default falls back to `emit_epilogue`.
    fn emit_epilogue_with_info(
        &self,
        sink: &mut CodeSink,
        func: &MirFunction,
        info: &PrologueInfo,
    ) {
        let _ = info;
        self.emit_epilogue(sink, func);
    }

    /// Emit an entire function (convenience wrapper).
    fn emit_function(&self, sink: &mut CodeSink, func: &MirFunction) {
        self.emit_prologue(sink, func);
        let reg_names: Vec<String> = (0..func.vreg_count)
            .map(|i| self.vreg_name(VReg::new(i)))
            .collect();
        for block in &func.blocks {
            for inst in &block.insts {
                self.emit_inst(sink, inst, &reg_names);
            }
        }
        self.emit_epilogue(sink, func);
    }
}
