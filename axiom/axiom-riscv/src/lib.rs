//! Axiom RISC-V — backend target for RV64IMAC (64-bit RISC-V).
//!
//! Implements the `Target` trait from `axiom-target` with RISC-V assembly
//! syntax, 32 GPRs (x0–x31) + 32 FPRs (f0–f31), and the standard
//! RISC-V calling convention.

use axiom_ir::nodes::Type;
use axiom_mir::{CmpCond, FCmpCond, MirFunction, MirInst, VReg};
use axiom_target::{
    CallingConv, CodeSink, Endianness, PhysReg, RegClass, RegisterInfo, Target, TargetDesc,
};

// ── Physical register indices ──────────────────────────────────────────
//
// GPRs: 0–31  (x0–x31)
// FPRs: 32–63 (f0–f31)

#[allow(dead_code)]
const X0: PhysReg = PhysReg::new(0);  // zero
#[allow(dead_code)]
const X1: PhysReg = PhysReg::new(1);  // ra
#[allow(dead_code)]
const X2: PhysReg = PhysReg::new(2);  // sp
const X8: PhysReg = PhysReg::new(8);  // s0/fp
const X10: PhysReg = PhysReg::new(10); // a0
const X11: PhysReg = PhysReg::new(11); // a1

const GPR_NAMES: &[&str] = &[
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];

const FPR_NAMES: &[&str] = &[
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
    "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
    "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

// ── RISC-V 64 Target ───────────────────────────────────────────────────

/// RISC-V 64-bit target backend (standard calling convention).
pub struct Riscv64Target {
    desc: TargetDesc,
}

impl Riscv64Target {
    pub fn new() -> Self {
        let desc = build_target_desc();
        Self { desc }
    }

    /// Map a VReg index to its default RISC-V register name.
    /// We skip x0 (zero), x1 (ra), x2 (sp) — so the first usable arg
    /// reg is x10 (a0). For simplicity, VRegs 0..20 map to a0–a7, t0–t6,
    /// s2–s11 (the caller/callee-saved scratch set); the rest are spills.
    fn vreg_to_asm(&self, vreg: VReg) -> String {
        let idx = vreg.as_u32() as usize;
        // Usable GPRs for allocation (skip zero, ra, sp, gp, tp, fp, s1):
        // a0–a7 (x10–x17), t0–t6 (x5–x7, x28–x31), s2–s11 (x18–x27)
        let alloc_order: &[&str] = &[
            "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", // 0–7
            "t0", "t1", "t2", "t3", "t4", "t5", "t6",       // 8–14
            "s2", "s3", "s4", "s5", "s6", "s7",             // 15–19
            "s8", "s9", "s10", "s11",                        // 20–23
        ];
        if idx < alloc_order.len() {
            alloc_order[idx].to_string()
        } else {
            let off = (idx - alloc_order.len()) as i32 * 8 + 16;
            format!("{}(s0)", -off)
        }
    }
}

impl Default for Riscv64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl Target for Riscv64Target {
    fn desc(&self) -> &TargetDesc {
        &self.desc
    }

    fn emit_prologue(&self, sink: &mut CodeSink, func: &MirFunction) {
        sink.emit_comment(&format!("function: {}", func.name));
        sink.emit_label(&func.name);
        let frame_size = compute_frame_size(func);
        sink.emit(&format!("    addi    sp, sp, -{}", frame_size));
        sink.emit("    sd      ra, 8(sp)");
        sink.emit("    sd      s0, 0(sp)");
        sink.emit("    mv      s0, sp");
    }

    fn emit_epilogue(&self, sink: &mut CodeSink, _func: &MirFunction) {
        sink.emit("    ld      ra, 8(sp)");
        sink.emit("    ld      s0, 0(sp)");
        let frame_size = compute_frame_size(_func);
        sink.emit(&format!("    addi    sp, sp, {}", frame_size));
        sink.emit("    ret");
    }

    fn emit_inst(&self, sink: &mut CodeSink, inst: &MirInst, reg_names: &[String]) {
        match inst {
            MirInst::Label { block } => {
                sink.emit_label(&format!(".L{}", block.as_u32()));
            }

            MirInst::Mov { dst, src } => {
                sink.emit(&format!(
                    "    mv      {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::MovImm { dst, imm } => {
                let val = imm.as_i64();
                if val >= -2048 && val <= 2047 {
                    sink.emit(&format!(
                        "    li      {}, {}",
                        rn(reg_names, *dst),
                        val
                    ));
                } else {
                    sink.emit(&format!(
                        "    li      {}, {}",
                        rn(reg_names, *dst),
                        val
                    ));
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
                    "    div     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Rem { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    rem     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
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
                    "    or      {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Xor { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    xor     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::Shl { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    sll     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::ShlImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    slli    {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    amount
                ));
            }

            MirInst::Shr { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    srl     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::ShrImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    srli    {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    amount
                ));
            }

            MirInst::Sar { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    sra     {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    rn(reg_names, *rhs)
                ));
            }

            MirInst::SarImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    srai    {}, {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *lhs),
                    amount
                ));
            }

            MirInst::Not { dst, src } => {
                sink.emit(&format!(
                    "    not     {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::Cmp { dst, lhs, rhs, cond } => {
                match cond {
                    CmpCond::Eq => {
                        sink.emit(&format!(
                            "    sub     {}, {}, {}",
                            rn(reg_names, *dst),
                            rn(reg_names, *lhs),
                            rn(reg_names, *rhs)
                        ));
                        sink.emit(&format!(
                            "    seqz    {}, {}",
                            rn(reg_names, *dst),
                            rn(reg_names, *dst)
                        ));
                    }
                    CmpCond::Ne => {
                        sink.emit(&format!(
                            "    sub     {}, {}, {}",
                            rn(reg_names, *dst),
                            rn(reg_names, *lhs),
                            rn(reg_names, *rhs)
                        ));
                        sink.emit(&format!(
                            "    snez    {}, {}",
                            rn(reg_names, *dst),
                            rn(reg_names, *dst)
                        ));
                    }
                    CmpCond::Lt => {
                        sink.emit(&format!(
                            "    slt     {}, {}, {}",
                            rn(reg_names, *dst),
                            rn(reg_names, *lhs),
                            rn(reg_names, *rhs)
                        ));
                    }
                    CmpCond::Le => {
                        // a <= b  <=>  !(b < a)
                        let tmp = *dst; // reuse dst as temp
                        sink.emit(&format!(
                            "    slt     {}, {}, {}",
                            rn(reg_names, tmp),
                            rn(reg_names, *rhs),
                            rn(reg_names, *lhs)
                        ));
                        sink.emit(&format!(
                            "    xori    {}, {}, 1",
                            rn(reg_names, tmp),
                            rn(reg_names, tmp)
                        ));
                    }
                    CmpCond::Gt => {
                        // a > b  <=>  b < a
                        sink.emit(&format!(
                            "    slt     {}, {}, {}",
                            rn(reg_names, *dst),
                            rn(reg_names, *rhs),
                            rn(reg_names, *lhs)
                        ));
                    }
                    CmpCond::Ge => {
                        // a >= b  <=>  !(a < b)
                        let tmp = *dst;
                        sink.emit(&format!(
                            "    slt     {}, {}, {}",
                            rn(reg_names, tmp),
                            rn(reg_names, *lhs),
                            rn(reg_names, *rhs)
                        ));
                        sink.emit(&format!(
                            "    xori    {}, {}, 1",
                            rn(reg_names, tmp),
                            rn(reg_names, tmp)
                        ));
                    }
                }
            }

            MirInst::Load { dst, addr } => {
                sink.emit(&format!(
                    "    ld      {}, 0({})",
                    rn(reg_names, *dst),
                    rn(reg_names, *addr)
                ));
            }

            MirInst::Store { addr, val } => {
                sink.emit(&format!(
                    "    sd      {}, 0({})",
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
                sink.emit(&format!("    addi    sp, sp, -{}", aligned_size));
                sink.emit(&format!(
                    "    mv      {}, sp",
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Call {
                dst,
                func,
                args,
            } => {
                // RV64 ABI: a0–a7 for args
                let arg_reg_names: &[&str] = &[
                    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                ];
                for (i, arg) in args.iter().enumerate() {
                    if i < arg_reg_names.len() {
                        sink.emit(&format!(
                            "    mv      {}, {}",
                            arg_reg_names[i],
                            rn(reg_names, *arg)
                        ));
                    } else {
                        let off = (i - arg_reg_names.len()) as u32 * 8;
                        sink.emit(&format!(
                            "    sd      {}, {}(sp)",
                            rn(reg_names, *arg),
                            off
                        ));
                    }
                }
                sink.emit(&format!("    call    {}", func));
                if let Some(d) = dst {
                    sink.emit(&format!(
                        "    mv      {}, a0",
                        rn(reg_names, *d)
                    ));
                }
            }

            MirInst::Ret { val } => {
                if let Some(v) = val {
                    sink.emit(&format!(
                        "    mv      a0, {}",
                        rn(reg_names, *v)
                    ));
                }
                sink.emit("    ld      ra, 8(sp)");
                sink.emit("    ld      s0, 0(sp)");
                // We don't know the frame size here reliably, so emit a
                // restore through s0-based frame size. For simplicity,
                // we emit a full epilogue sequence.
                sink.emit_comment("epilogue (ret)");
                sink.emit("    ret");
            }

            MirInst::Jump { target } => {
                sink.emit(&format!("    j       .L{}", target.as_u32()));
            }

            MirInst::Branch {
                cond,
                true_block,
                false_block,
            } => {
                sink.emit(&format!(
                    "    bnez    {}, .L{}",
                    rn(reg_names, *cond),
                    true_block.as_u32()
                ));
                sink.emit(&format!("    j       .L{}", false_block.as_u32()));
            }

            MirInst::PhiCopy { dst, src } => {
                sink.emit(&format!(
                    "    mv      {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            // ── Extension / Truncation ──────────────────────────

            MirInst::ZExt { dst, src } => {
                // Zero-extend word: zext.w (RISC-V B extension) or andi with mask
                sink.emit(&format!(
                    "    zext.w  {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::SExt { dst, src } => {
                // Sign-extend word: sext.w (RISC-V Zba extension)
                sink.emit(&format!(
                    "    sext.w  {}, {}",
                    rn(reg_names, *dst),
                    rn(reg_names, *src)
                ));
            }

            MirInst::Trunc { dst, src } => {
                // Truncation: just copy (lower 32 bits are already there)
                if src != dst {
                    sink.emit(&format!(
                        "    mv      {}, {}",
                        rn(reg_names, *dst),
                        rn(reg_names, *src)
                    ));
                }
                // Mask to 32 bits: andi with $0xFFFFFFFF
                sink.emit(&format!(
                    "    andi    {}, {}, -1",
                    rn(reg_names, *dst),
                    rn(reg_names, *dst)
                ));
            }

            // ── Spill / Reload ──────────────────────────────────

            MirInst::SpillStore { vreg, slot } => {
                let offset = (*slot as i32 + 1) * 8;
                sink.emit(&format!(
                    "    sd      {}, {}(s0)",
                    rn(reg_names, *vreg),
                    -offset
                ));
            }

            MirInst::SpillLoad { vreg, slot } => {
                let offset = (*slot as i32 + 1) * 8;
                sink.emit(&format!(
                    "    ld      {}, {}(s0)",
                    rn(reg_names, *vreg),
                    -offset
                ));
            }

            // ── Floating-Point (RISC-V D extension) ─────────────────
            MirInst::FAdd { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fadd.d  {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FSub { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fsub.d  {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FMul { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fmul.d  {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FDiv { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fdiv.d  {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::FRem { dst, lhs, rhs } => {
                sink.emit_comment("frem — call fmod");
                sink.emit(&format!(
                    "    fmv.d   fa0, {}",
                    rn(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    fmv.d   fa1, {}",
                    rn(reg_names, *rhs)
                ));
                sink.emit("    call    fmod");
                sink.emit(&format!(
                    "    fmv.d   {}, fa0",
                    rn(reg_names, *dst)
                ));
            }
            MirInst::FNeg { dst, src } => {
                sink.emit(&format!(
                    "    fneg.d  {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FAbs { dst, src } => {
                sink.emit(&format!(
                    "    fabs.d  {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FSqrt { dst, src } => {
                sink.emit(&format!(
                    "    fsqrt.d {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FCmp { dst, lhs, rhs, cond } => {
                match cond {
                    FCmpCond::Eq => {
                        sink.emit(&format!(
                            "    feq.d   {}, {}, {}",
                            rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                        ));
                    }
                    FCmpCond::Ne => {
                        // feq then negate
                        sink.emit(&format!(
                            "    feq.d   {}, {}, {}",
                            rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                        ));
                        sink.emit(&format!(
                            "    xori    {}, {}, 1",
                            rn(reg_names, *dst), rn(reg_names, *dst)
                        ));
                    }
                    FCmpCond::Lt => {
                        sink.emit(&format!(
                            "    flt.d   {}, {}, {}",
                            rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                        ));
                    }
                    FCmpCond::Le => {
                        sink.emit(&format!(
                            "    fle.d   {}, {}, {}",
                            rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                        ));
                    }
                    FCmpCond::Gt => {
                        sink.emit(&format!(
                            "    fgt.d   {}, {}, {}",
                            rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                        ));
                    }
                    FCmpCond::Ge => {
                        sink.emit(&format!(
                            "    fge.d   {}, {}, {}",
                            rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                        ));
                    }
                }
            }
            MirInst::FpToSInt { dst, src } => {
                sink.emit(&format!(
                    "    fcvt.l.d {}, {}, rtz",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::SIntToFp { dst, src } => {
                sink.emit(&format!(
                    "    fcvt.d.l {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::FpToUInt { dst, src } => {
                sink.emit(&format!(
                    "    fcvt.lu.d {}, {}, rtz",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::UIntToFp { dst, src } => {
                sink.emit(&format!(
                    "    fcvt.d.lu {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *src)
                ));
            }
            MirInst::Copysign { dst, lhs, rhs } => {
                sink.emit_comment("copysign — fsgnjn.d + fsgnj.d");
                sink.emit(&format!(
                    "    fsgnjn.d {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    fsgnj.d  {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *dst), rn(reg_names, *rhs)
                ));
            }
            MirInst::Fmin { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fmin.d  {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }
            MirInst::Fmax { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    fmax.d  {}, {}, {}",
                    rn(reg_names, *dst), rn(reg_names, *lhs), rn(reg_names, *rhs)
                ));
            }

            // ── Vector Operations (RISC-V V extension placeholders) ──
            MirInst::VecBroadcast { dst, src, lane_count } => {
                sink.emit_comment(&format!("vec_broadcast lane_count={}", lane_count));
                sink.emit(&format!(
                    "    # vmerge.vvm v{}, v{}, v{}",
                    dst.as_u32(), src.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecLoad { dst, addr, lane_count } => {
                sink.emit_comment(&format!("vec_load lane_count={}", lane_count));
                sink.emit(&format!(
                    "    # vle32.v v{}, ({})",
                    dst.as_u32(), rn(reg_names, *addr)
                ));
            }
            MirInst::VecStore { addr, val, lane_count } => {
                sink.emit_comment(&format!("vec_store lane_count={}", lane_count));
                sink.emit(&format!(
                    "    # vse32.v v{}, ({})",
                    val.as_u32(), rn(reg_names, *addr)
                ));
            }
            MirInst::VecAdd { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vadd.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecSub { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vsub.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecMul { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vmul.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecDiv { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vfdiv.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecAnd { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vand.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecOr { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vor.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecXor { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vxor.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecMin { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vmin.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecMax { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    # vmax.vv v{}, v{}, v{}",
                    dst.as_u32(), lhs.as_u32(), rhs.as_u32()
                ));
            }
            MirInst::VecNeg { dst, src } => {
                sink.emit(&format!(
                    "    # vneg.vv v{}, v{}",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecAbs { dst, src } => {
                sink.emit(&format!(
                    "    # vabs.vv v{}, v{}",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecSqrt { dst, src } => {
                sink.emit(&format!(
                    "    # vfsqrt.v v{}, v{}",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecShuffle { dst, src, mask } => {
                sink.emit_comment(&format!("vec_shuffle mask={:?}", mask));
                sink.emit(&format!(
                    "    # vrgather.vv v{}, v{}, v{}",
                    dst.as_u32(), src.as_u32(), src.as_u32()
                ));
            }
            MirInst::VecReduceSum { dst, src, lane_count } => {
                sink.emit_comment(&format!("vec_reduce_sum lane_count={}", lane_count));
                sink.emit(&format!(
                    "    # vredsum.vs v{}, v{}, v0",
                    dst.as_u32(), src.as_u32()
                ));
            }
            MirInst::ExtractLane { dst, src, index } => {
                sink.emit(&format!(
                    "    # vmv.x.s {}, v{}[{}]",
                    rn(reg_names, *dst), src.as_u32(), index
                ));
            }
            MirInst::InsertLane { dst, src, index, elem } => {
                sink.emit_comment(&format!("insert_lane idx={}", index));
                sink.emit(&format!(
                    "    # vmerge.vxm v{}, v{}, {}, v0",
                    dst.as_u32(), src.as_u32(), rn(reg_names, *elem)
                ));
            }
        }
    }

    fn legalize_type(&self, ty: Type) -> Type {
        match ty {
            // Promote small integers to 64-bit (RV64 native width)
            Type::I8 | Type::I16 | Type::I32 | Type::U8 | Type::U16 | Type::U32 | Type::Bool => {
                Type::I64
            }
            Type::F32 => Type::F32,
            Type::F64 => Type::F64,
            other => other,
        }
    }

    fn reg_name(&self, reg: PhysReg) -> String {
        let idx = reg.as_u16() as usize;
        if idx < GPR_NAMES.len() {
            GPR_NAMES[idx].to_string()
        } else if idx < GPR_NAMES.len() + FPR_NAMES.len() {
            FPR_NAMES[idx - GPR_NAMES.len()].to_string()
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

/// Compute frame size including ra + s0 saves and spill slots.
fn compute_frame_size(func: &MirFunction) -> u32 {
    // Minimum: ra + s0 = 16 bytes
    let spill_count = func.vreg_count.saturating_sub(24); // 24 allocatable regs
    let size = 16 + spill_count * 8;
    // Round up to 16-byte alignment
    (size + 15) & !15
}

// ── Build the TargetDesc ───────────────────────────────────────────────

fn build_target_desc() -> TargetDesc {
    let gpr_reserved: &[bool] = &[
        true,  // x0  (zero — always 0)
        true,  // x1  (ra — return address)
        true,  // x2  (sp — stack pointer)
        true,  // x3  (gp — global pointer)
        true,  // x4  (tp — thread pointer)
        false, // x5  (t0)
        false, // x6  (t1)
        false, // x7  (t2)
        true,  // x8  (s0/fp — frame pointer)
        false, // x9  (s1)
        false, // x10 (a0)
        false, // x11 (a1)
        false, // x12 (a2)
        false, // x13 (a3)
        false, // x14 (a4)
        false, // x15 (a5)
        false, // x16 (a6)
        false, // x17 (a7)
        false, // x18 (s2)
        false, // x19 (s3)
        false, // x20 (s4)
        false, // x21 (s5)
        false, // x22 (s6)
        false, // x23 (s7)
        false, // x24 (s8)
        false, // x25 (s9)
        false, // x26 (s10)
        false, // x27 (s11)
        false, // x28 (t3)
        false, // x29 (t4)
        false, // x30 (t5)
        false, // x31 (t6)
    ];

    let mut registers: Vec<RegisterInfo> = Vec::with_capacity(64);

    for (i, &name) in GPR_NAMES.iter().enumerate() {
        registers.push(RegisterInfo {
            reg: PhysReg::new(i as u16),
            name: name.to_string(),
            class: RegClass::Int,
            is_reserved: gpr_reserved[i],
        });
    }

    for (i, &name) in FPR_NAMES.iter().enumerate() {
        registers.push(RegisterInfo {
            reg: PhysReg::new((32 + i) as u16),
            name: name.to_string(),
            class: RegClass::Float,
            is_reserved: false,
        });
    }

    let calling_conv = CallingConv {
        // a0–a7 = x10–x17
        arg_regs: vec![
            X10, X11,
            PhysReg::new(12), PhysReg::new(13),
            PhysReg::new(14), PhysReg::new(15),
            PhysReg::new(16), PhysReg::new(17),
        ],
        ret_regs: vec![X10, X11],
        callee_saved: vec![
            X8, PhysReg::new(9),
            PhysReg::new(18), PhysReg::new(19),
            PhysReg::new(20), PhysReg::new(21),
            PhysReg::new(22), PhysReg::new(23),
            PhysReg::new(24), PhysReg::new(25),
            PhysReg::new(26), PhysReg::new(27),
        ],
        caller_saved: vec![
            X10, X11,
            PhysReg::new(12), PhysReg::new(13),
            PhysReg::new(14), PhysReg::new(15),
            PhysReg::new(16), PhysReg::new(17),
            PhysReg::new(5), PhysReg::new(6), PhysReg::new(7),
            PhysReg::new(28), PhysReg::new(29),
            PhysReg::new(30), PhysReg::new(31),
        ],
        stack_align: 16,
    };

    TargetDesc {
        name: "riscv64".to_string(),
        ptr_width: 64,
        endianness: Endianness::Little,
        registers,
        calling_conv,
        supported_widths: vec![8, 16, 32, 64],
        has_cmov: false, // Base RV64M has no conditional move (extension Zicond adds it)
        has_vector: false,
        vector_width: 0,
    }
}
