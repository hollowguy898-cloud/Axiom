//! Axiom AArch64 — backend target for ARMv8-A (AArch64).
//!
//! Implements the `Target` trait from `axiom-target` with AArch64 assembly
//! syntax, 31 GPRs (x0–x30) + 32 SIMD/FP registers (v0–v31), and the
//! AAPCS calling convention.

use axiom_ir::nodes::Type;
use axiom_mir::{CmpCond, FCmpCond, MirFunction, MirInst, VReg};
use axiom_target::{
    CallingConv, CodeSink, Endianness, PhysReg, RegClass, RegisterInfo, Target, TargetDesc,
};

// ── Physical register indices ──────────────────────────────────────────
//
// GPRs: 0–30  (x0–x30, where x29=fp, x30=lr, x31(sp) is implicit)
// SIMD: 32–63 (v0–v31)

const X0: PhysReg = PhysReg::new(0);
const X1: PhysReg = PhysReg::new(1);
const X2: PhysReg = PhysReg::new(2);
const X3: PhysReg = PhysReg::new(3);
const X4: PhysReg = PhysReg::new(4);
const X5: PhysReg = PhysReg::new(5);
const X6: PhysReg = PhysReg::new(6);
const X7: PhysReg = PhysReg::new(7);
const X8: PhysReg = PhysReg::new(8);
const X9: PhysReg = PhysReg::new(9);
#[allow(dead_code)]
const X19: PhysReg = PhysReg::new(19);
const X28: PhysReg = PhysReg::new(28);
#[allow(dead_code)]
const X29: PhysReg = PhysReg::new(29); // frame pointer (fp)
#[allow(dead_code)]
const X30: PhysReg = PhysReg::new(30); // link register (lr)

const GPR_NAMES: &[&str] = &[
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
    "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26",
    "x27", "x28", "x29", "x30",
];

const SIMD_NAMES: &[&str] = &[
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
    "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
    "v27", "v28", "v29", "v30", "v31",
];

// ── AArch64 Target ─────────────────────────────────────────────────────

/// AArch64 target backend (AAPCS).
pub struct AArch64Target {
    desc: TargetDesc,
}

impl AArch64Target {
    pub fn new() -> Self {
        let desc = build_target_desc();
        Self { desc }
    }

    /// Map a VReg index to its default AArch64 register name.
    /// First 19 VRegs map to x0–x18; the rest are spill slots.
    fn vreg_to_asm(&self, vreg: VReg) -> String {
        let idx = vreg.as_u32() as usize;
        if idx < GPR_NAMES.len() {
            GPR_NAMES[idx].to_string()
        } else {
            let off = (idx - GPR_NAMES.len()) as i32 * 8 + 16;
            format!("[x29, #{}]", -off)
        }
    }
}

impl Default for AArch64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl Target for AArch64Target {
    fn desc(&self) -> &TargetDesc {
        &self.desc
    }

    fn emit_prologue(&self, sink: &mut CodeSink, func: &MirFunction) {
        sink.emit_comment(&format!("function: {}", func.name));
        sink.emit_label(&func.name);
        // AAPCS: save fp (x29) and lr (x30), create frame
        sink.emit("    stp     x29, x30, [sp, #-16]!");
        sink.emit("    mov     x29, sp");
        let frame_size = compute_frame_size(func);
        if frame_size > 0 {
            sink.emit(&format!("    sub     sp, sp, #{}", frame_size));
        }
    }

    fn emit_epilogue(&self, sink: &mut CodeSink, _func: &MirFunction) {
        sink.emit("    ldp     x29, x30, [sp], #16");
        sink.emit("    ret");
    }

    fn emit_inst(&self, sink: &mut CodeSink, inst: &MirInst, reg_names: &[String]) {
        match inst {
            MirInst::Label { block } => {
                sink.emit_label(&format!(".L{}", block.as_u32()));
            }

            MirInst::Mov { dst, src } => {
                sink.emit(&format!(
                    "    mov     {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::MovImm { dst, imm } => {
                let val = imm.as_i64();
                // Use movz/movk sequence for 64-bit immediates
                if val >= 0 && val <= 0xFFFF {
                    sink.emit(&format!(
                        "    movz    {}, #{}",
                        rn(reg_names, *dst),
                        val
                    ));
                } else if val >= 0 && val <= 0xFFFF_FFFF {
                    let lo = val & 0xFFFF;
                    let hi = (val >> 16) & 0xFFFF;
                    sink.emit(&format!(
                        "    movz    {}, #{}",
                        rn(reg_names, *dst),
                        lo
                    ));
                    sink.emit(&format!(
                        "    movk    {}, #{}, lsl #16",
                        rn(reg_names, *dst),
                        hi
                    ));
                } else {
                    let lo = val & 0xFFFF;
                    let hi1 = (val >> 16) & 0xFFFF;
                    let hi2 = (val >> 32) & 0xFFFF;
                    let hi3 = (val >> 48) & 0xFFFF;
                    sink.emit(&format!(
                        "    movz    {}, #{}",
                        rn(reg_names, *dst),
                        lo
                    ));
                    if hi1 != 0 {
                        sink.emit(&format!(
                            "    movk    {}, #{}, lsl #16",
                            rn(reg_names, *dst),
                            hi1
                        ));
                    }
                    if hi2 != 0 {
                        sink.emit(&format!(
                            "    movk    {}, #{}, lsl #32",
                            rn(reg_names, *dst),
                            hi2
                        ));
                    }
                    if hi3 != 0 {
                        sink.emit(&format!(
                            "    movk    {}, #{}, lsl #48",
                            rn(reg_names, *dst),
                            hi3
                        ));
                    }
                }
            }

            MirInst::Add { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    add     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Sub { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    sub     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Mul { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    mul     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Div { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    sdiv    {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Rem { dst, lhs, rhs } => {
                // AArch64 has no direct remainder; compute rem = lhs - (lhs/rhs)*rhs
                sink.emit(&format!(
                    "    sdiv    {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
                sink.emit(&format!(
                    "    msub    {}, {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *dst),
                    rn(reg_names, *rhs),
                    rn(reg_names, *lhs)
                ));
            }

            MirInst::Neg { dst, src } => {
                sink.emit(&format!(
                    "    neg     {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::And { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    and     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Or { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    orr     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Xor { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    eor     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Shl { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    lsl     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::ShlImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    lsl     {}, {}, #{}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    amount
                ));
            }

            MirInst::Shr { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    lsr     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::ShrImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    lsr     {}, {}, #{}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    amount
                ));
            }

            MirInst::Sar { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    asr     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::SarImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    asr     {}, {}, #{}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    amount
                ));
            }

            MirInst::Not { dst, src } => {
                sink.emit(&format!(
                    "    mvn     {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::Cmp { dst, lhs, rhs, cond } => {
                sink.emit(&format!(
                    "    cmp     {}, {}",
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
                let cc = match cond {
                    CmpCond::Eq => "eq",
                    CmpCond::Ne => "ne",
                    CmpCond::Lt => "lt",
                    CmpCond::Le => "le",
                    CmpCond::Gt => "gt",
                    CmpCond::Ge => "ge",
                };
                sink.emit(&format!(
                    "    cset    {}, {}",
                    rn(reg_names, *dst),
                    cc
                ));
            }

            MirInst::Load { dst, addr } => {
                sink.emit(&format!(
                    "    ldr     {}, [{}]",
                    rn(reg_names, *dst),
                    rn(reg_names, *addr)
                ));
            }

            MirInst::Store { addr, val } => {
                sink.emit(&format!(
                    "    str     {}, [{}]",
                    rn(reg_names, *val),
                    rn(reg_names, *addr)
                ));
            }

            MirInst::StackAlloc { dst, size, align } => {
                sink.emit_comment(&format!(
                    "stack_alloc size={} align={}",
                    size, align
                ));
                let aligned_size = align_to(*size, *align);
                sink.emit(&format!("    sub     sp, sp, #{}", aligned_size));
                sink.emit(&format!(
                    "    mov     {}, sp",
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Call {
                dst,
                func,
                args,
            } => {
                // AAPCS: args in x0–x7
                let arg_regs_count = 8usize;
                for (i, arg) in args.iter().enumerate() {
                    if i < arg_regs_count {
                        sink.emit(&format!(
                            "    mov     x{}, {}",
                            i,
                            rn(reg_names, *arg)
                        ));
                    } else {
                        // Stack-pass remaining args
                        let off = (i - arg_regs_count) as u32 * 8;
                        sink.emit(&format!(
                            "    str     {}, [sp, #{}]",
                            rn(reg_names, *arg),
                            off
                        ));
                    }
                }
                sink.emit(&format!("    bl      {}", func));
                if let Some(d) = dst {
                    sink.emit(&format!(
                        "    mov     {}, x0",
                        rn(reg_names, *d)
                    ));
                }
            }

            MirInst::Ret { val } => {
                if let Some(v) = val {
                    sink.emit(&format!(
                        "    mov     x0, {}",
                        rn(reg_names, *v)
                    ));
                }
                sink.emit("    ldp     x29, x30, [sp], #16");
                sink.emit("    ret");
            }

            MirInst::Jump { target } => {
                sink.emit(&format!("    b       .L{}", target.as_u32()));
            }

            MirInst::Branch {
                cond,
                true_block,
                false_block,
            } => {
                sink.emit(&format!(
                    "    cbnz    {}, .L{}",
                    rn(reg_names, *cond),
                    true_block.as_u32()
                ));
                sink.emit(&format!("    b       .L{}", false_block.as_u32()));
            }

            MirInst::PhiCopy { dst, src } => {
                sink.emit(&format!(
                    "    mov     {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            // ── Extension / Truncation ──────────────────────────

            MirInst::ZExt { dst, src } => {
                // Zero-extend: uxtb for byte, uxth for halfword, uxtw for word
                sink.emit(&format!(
                    "    uxtw    {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::SExt { dst, src } => {
                // Sign-extend: sxtb for byte, sxth for halfword, sxtw for word
                sink.emit(&format!(
                    "    sxtw    {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::Trunc { dst, src } => {
                // Truncation on AArch64: just copy (narrower register access handles it)
                // For i64→i32, use mov w, w (zero-extends to 64)
                sink.emit(&format!(
                    "    mov     {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            // ── Spill / Reload ──────────────────────────────────

            MirInst::SpillStore { vreg, slot } => {
                let offset = (*slot as i32 + 1) * 8;
                sink.emit(&format!(
                    "    str     {}, [x29, #{}]",
                    rn(reg_names, *vreg),
                    -offset
                ));
            }

            MirInst::SpillLoad { vreg, slot } => {
                let offset = (*slot as i32 + 1) * 8;
                sink.emit(&format!(
                    "    ldr     {}, [x29, #{}]",
                    rn(reg_names, *vreg),
                    -offset
                ));
            }

            // ── Floating-Point (AArch64 f64) ──────────────────────
            MirInst::FAdd { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fadd    {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FSub { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fsub    {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FMul { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fmul    {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FDiv { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fdiv    {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FRem { dst, lhs, rhs } => {
                sink.emit_comment("frem — call fmod");
                sink.emit(&format!("    fmov    d0, {}", rn(reg_names, *lhs)));
                sink.emit(&format!("    fmov    d1, {}", rn(reg_names, *rhs)));
                sink.emit("    bl      fmod");
                sink.emit(&format!("    fmov    {}, d0", rn(reg_names, *dst)));
            }
            MirInst::FNeg { dst, src } => {
                sink.emit(&format!(
                    "    fneg    {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FAbs { dst, src } => {
                sink.emit(&format!(
                    "    fabs    {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FSqrt { dst, src } => {
                sink.emit(&format!(
                    "    fsqrt   {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FCmp { dst, lhs, rhs, cond } => {
                sink.emit(&format!(
                    "    fcmp    {}, {}",
                    rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
                let cc = match cond {
                    FCmpCond::Eq => "eq",
                    FCmpCond::Ne => "ne",
                    FCmpCond::Lt => "lo",   // below (unsigned) for FP
                    FCmpCond::Le => "ls",   // below or equal
                    FCmpCond::Gt => "hi",   // above (unsigned) for FP
                    FCmpCond::Ge => "hs",   // above or equal
                };
                sink.emit(&format!(
                    "    cset    {}, {}",
                    rn(reg_names, *dst),
                    cc
                ));
            }
            MirInst::FpToSInt { dst, src } => {
                sink.emit(&format!(
                    "    fcvtzs  {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::SIntToFp { dst, src } => {
                sink.emit(&format!(
                    "    scvtf   {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FpToUInt { dst, src } => {
                sink.emit(&format!(
                    "    fcvtzu  {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::UIntToFp { dst, src } => {
                sink.emit(&format!(
                    "    ucvtf   {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::Copysign { dst, lhs, rhs: _ } => {
                sink.emit_comment("copysign — AArch64 no direct insn; use bit manipulation");
                // Simplified: abs(lhs) OR signbit(rhs)
                sink.emit(&format!(
                    "    fabs    {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs)
                ));
                sink.emit_comment("TODO: extract sign from rhs and OR into dst");
            }
            MirInst::Fmin { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fmin    {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::Fmax { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fmax    {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }

            // ── Vector Operations (NEON) ──────────────────────────
            MirInst::VecBroadcast { dst, src, lane_count: _ } => {
                sink.emit(&format!(
                    "    dup     v{}.4s, {}[0]",
                    dst.as_u32(), rn(reg_names, *src)
                ));
            }
            MirInst::VecLoad { dst, addr, lane_count: _ } => {
                sink.emit(&format!(
                    "    ldr     q{}, [{}]",
                    dst.as_u32(), rn(reg_names, *addr)
                ));
            }
            MirInst::VecStore { addr, val, lane_count: _ } => {
                sink.emit(&format!(
                    "    str     q{}, [{}]",
                    val.as_u32(), rn(reg_names, *addr)
                ));
            }
            MirInst::VecAdd { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    add     v{}.4s, v{}.4s, v{}.4s",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecSub { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    sub     v{}.4s, v{}.4s, v{}.4s",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecMul { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    mul     v{}.4s, v{}.4s, v{}.4s",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecDiv { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fdiv    v{}.4s, v{}.4s, v{}.4s",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecAnd { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    and     v{}.16b, v{}.16b, v{}.16b",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecOr { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    orr     v{}.16b, v{}.16b, v{}.16b",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecXor { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    eor     v{}.16b, v{}.16b, v{}.16b",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecMin { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    smin    v{}.4s, v{}.4s, v{}.4s",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecMax { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    smax    v{}.4s, v{}.4s, v{}.4s",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecNeg { dst, src } => {
                sink.emit(&format!(
                    "    neg     v{}.4s, v{}.4s",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecAbs { dst, src } => {
                sink.emit(&format!(
                    "    abs     v{}.4s, v{}.4s",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecSqrt { dst, src } => {
                sink.emit(&format!(
                    "    fsqrt   v{}.4s, v{}.4s",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecShuffle { dst, src, mask } => {
                sink.emit_comment(&format!("vec_shuffle mask={:?}", mask));
                sink.emit(&format!(
                    "    tbl     v{}.16b, {{v{}.16b}}, v{}.16b",
                    dst.as_u32(), src.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecReduceSum { dst, src, lane_count } => {
                sink.emit_comment(&format!("vec_reduce_sum lane_count={}", lane_count));
                // Use pairwise add to reduce: addv for integer, faddp for float
                sink.emit(&format!(
                    "    addv    s{}, v{}.4s",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::ExtractLane { dst, src, index } => {
                sink.emit(&format!(
                    "    mov     {}, v{}.s[{}]",
                    rn(reg_names, *dst), src.as_u32(), index
                ));
            }
            MirInst::InsertLane { dst, src, index, elem } => {
                if src != dst {
                    sink.emit(&format!(
                        "    mov     v{}.16b, v{}.16b",
                        dst.as_u32(), src.as_u32()
                    ));
                }
                sink.emit(&format!(
                    "    ins     v{}.s[{}], {}",
                    dst.as_u32(), index, rn(reg_names, *elem)
                ));
            }
        }
    }

    fn legalize_type(&self, ty: Type) -> Type {
        match ty {
            // Promote small integers to 64-bit (AArch64 native width)
            Type::I8 | Type::I16 | Type::I32 | Type::U8 | Type::U16 | Type::U32 | Type::Bool => {
                Type::I64
            }
            // Floats remain as-is
            Type::F32 => Type::F32,
            Type::F64 => Type::F64,
            other => other,
        }
    }

    fn reg_name(&self, reg: PhysReg) -> String {
        let idx = reg.as_u16() as usize;
        if idx < GPR_NAMES.len() {
            GPR_NAMES[idx].to_string()
        } else if idx < GPR_NAMES.len() + SIMD_NAMES.len() {
            SIMD_NAMES[idx - GPR_NAMES.len()].to_string()
        } else {
            format!("preg{}", idx)
        }
    }

    fn vreg_name(&self, vreg: VReg) -> String {
        self.vreg_to_asm(vreg)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn rn(reg_names: &[String], vreg: VReg) -> &str {
    let idx = vreg.as_u32() as usize;
    if idx < reg_names.len() {
        &reg_names[idx]
    } else {
        static PLACEHOLDER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        PLACEHOLDER.get_or_init(|| "<undef>".to_string())
    }
}

fn align_to(size: u32, align: u32) -> u32 {
    (size + align - 1) / align * align
}

fn compute_frame_size(func: &MirFunction) -> u32 {
    let spill_count = func.vreg_count.saturating_sub(GPR_NAMES.len() as u32);
    let size = spill_count * 8;
    // Round up to 16-byte alignment
    (size + 15) & !15
}

// ── Build the TargetDesc ───────────────────────────────────────────────

fn build_target_desc() -> TargetDesc {
    let gpr_reserved: &[bool] = &[
        false, false, false, false, // x0–x3
        false, false, false, false, // x4–x7
        false, // x8  (indirect result)
        false, // x9
        false, false, // x10–x11
        false, false, // x12–x13
        false, false, // x14–x15 (intrinsics)
        false, false, // x16–x17 (IP0/IP1)
        false, // x18 (platform)
        false, false, false, // x19–x21 (callee-saved)
        false, false, false, // x22–x24 (callee-saved)
        false, false, false, // x25–x27 (callee-saved)
        false, // x28 (callee-saved)
        true,  // x29 (fp — reserved)
        true,  // x30 (lr — reserved)
    ];

    let mut registers: Vec<RegisterInfo> = Vec::with_capacity(63);

    for (i, &name) in GPR_NAMES.iter().enumerate() {
        registers.push(RegisterInfo {
            reg: PhysReg::new(i as u16),
            name: name.to_string(),
            class: RegClass::Int,
            is_reserved: gpr_reserved[i],
        });
    }

    for (i, &name) in SIMD_NAMES.iter().enumerate() {
        registers.push(RegisterInfo {
            reg: PhysReg::new((31 + i) as u16),
            name: name.to_string(),
            class: RegClass::Float,
            is_reserved: false,
        });
    }

    let calling_conv = CallingConv {
        arg_regs: vec![X0, X1, X2, X3, X4, X5, X6, X7],
        ret_regs: vec![X0, X1],
        callee_saved: vec![
            PhysReg::new(19),
            PhysReg::new(20),
            PhysReg::new(21),
            PhysReg::new(22),
            PhysReg::new(23),
            PhysReg::new(24),
            PhysReg::new(25),
            PhysReg::new(26),
            PhysReg::new(27),
            X28,
        ],
        caller_saved: vec![
            X0, X1, X2, X3, X4, X5, X6, X7, X8, X9,
            PhysReg::new(10), PhysReg::new(11), PhysReg::new(12), PhysReg::new(13),
            PhysReg::new(14), PhysReg::new(15), PhysReg::new(16), PhysReg::new(17),
            PhysReg::new(18),
        ],
        stack_align: 16,
    };

    TargetDesc {
        name: "aarch64".to_string(),
        ptr_width: 64,
        endianness: Endianness::Little,
        registers,
        calling_conv,
        supported_widths: vec![8, 16, 32, 64],
        has_cmov: true, // AArch64 has csel
        has_vector: true,
        vector_width: 128,
    }
}
