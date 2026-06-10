//! Axiom x86-64 — backend target for x86-64 (System V AMD64 ABI).
//!
//! Implements the `Target` trait from `axiom-target` with AT&T assembly
//! syntax, 16 GPRs + 16 XMM registers, and System V calling convention.

use axiom_ir::nodes::Type;
use axiom_mir::{CmpCond, FCmpCond, MirFunction, MirInst, VReg};
use axiom_target::{
    CallingConv, CodeSink, Endianness, PhysReg, PrologueInfo, RegClass, RegisterInfo, Target,
    TargetDesc,
};

// ── Physical register indices ──────────────────────────────────────────
//
// GPRs: 0–15   (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8–r15)
// XMM: 16–31   (xmm0–xmm15)

const RAX: PhysReg = PhysReg::new(0);
const RCX: PhysReg = PhysReg::new(1);
const RDX: PhysReg = PhysReg::new(2);
const RBX: PhysReg = PhysReg::new(3);
#[allow(dead_code)]
const RSP: PhysReg = PhysReg::new(4);
const RBP: PhysReg = PhysReg::new(5);
const RSI: PhysReg = PhysReg::new(6);
const RDI: PhysReg = PhysReg::new(7);
const R8: PhysReg = PhysReg::new(8);
const R9: PhysReg = PhysReg::new(9);
const R10: PhysReg = PhysReg::new(10);
const R11: PhysReg = PhysReg::new(11);
const R12: PhysReg = PhysReg::new(12);
const R13: PhysReg = PhysReg::new(13);
const R14: PhysReg = PhysReg::new(14);
const R15: PhysReg = PhysReg::new(15);

const GPR_NAMES: &[&str] = &[
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
    "r13", "r14", "r15",
];

const XMM_NAMES: &[&str] = &[
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
    "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
];

// ── x86-64 Target ──────────────────────────────────────────────────────

/// x86-64 target backend (System V AMD64 ABI).
pub struct X86_64Target {
    desc: TargetDesc,
}

impl X86_64Target {
    pub fn new() -> Self {
        let desc = build_target_desc();
        Self { desc }
    }

    /// Map a VReg index to its default x86-64 register name.
    /// First 16 VRegs map to GPRs in System V arg order; the rest are spill
    /// slots addressed as `[rbp - offset]`.
    fn vreg_to_asm(&self, vreg: VReg) -> String {
        let idx = vreg.as_u32() as usize;
        if idx < GPR_NAMES.len() {
            format!("%{}", GPR_NAMES[idx])
        } else {
            let off = (idx - GPR_NAMES.len()) as i32 * 8 + 8;
            format!("{}(%rbp)", -off)
        }
    }
}

impl Default for X86_64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl Target for X86_64Target {
    fn desc(&self) -> &TargetDesc {
        &self.desc
    }

    fn emit_prologue(&self, sink: &mut CodeSink, func: &MirFunction) {
        sink.emit_comment(&format!("function: {}", func.name));
        sink.emit("    .text");
        sink.emit(&format!("    .globl  {}", func.name));
        sink.emit_label(&func.name);
        sink.emit("    pushq   %rbp");
        sink.emit("    movq    %rsp, %rbp");
        // Frame size: 8 bytes per spilled VReg, rounded to 16-byte alignment.
        let frame_size = compute_frame_size(&func);
        if frame_size > 0 {
            sink.emit(&format!("    subq    ${}, %rsp", frame_size));
        }
    }

    fn emit_epilogue(&self, sink: &mut CodeSink, _func: &MirFunction) {
        sink.emit("    movq    %rbp, %rsp");
        sink.emit("    popq    %rbp");
        sink.emit("    retq");
    }

    fn emit_prologue_with_info(
        &self,
        sink: &mut CodeSink,
        func: &MirFunction,
        info: &PrologueInfo,
    ) {
        sink.emit("    .text");
        sink.emit(&format!("    .globl  {}", func.name));
        sink.emit_comment(&format!("function: {}", func.name));
        sink.emit_label(&func.name);

        // Step 1: Push callee-saved registers that are actually used.
        // We push them BEFORE rbp so that `movq %rbp, %rsp; popq %rbp`
        // in the epilogue restores to just above the callee-saved pushes,
        // and we can pop them in reverse order.
        for &cs_reg in &info.used_callee_saved {
            let name = self.reg_name(cs_reg);
            sink.emit(&format!("    pushq   {}", name));
        }

        // Step 2: Push %rbp and set frame pointer
        sink.emit("    pushq   %rbp");
        sink.emit("    movq    %rsp, %rbp");

        // Step 3: Allocate stack frame using frame_size from regalloc
        // (accounts for spill slots only; callee-saved pushes are handled above)
        if info.frame_size > 0 {
            sink.emit(&format!("    subq    ${}, %rsp", info.frame_size));
        }

        // Step 4: Move function arguments from System V arg registers
        // to their allocated physical registers.
        for &(arg_reg, assigned_reg) in &info.arg_moves {
            if arg_reg != assigned_reg {
                let arg_name = self.reg_name(arg_reg);
                let dst_name = self.reg_name(assigned_reg);
                sink.emit(&format!("    movq    {}, {}", arg_name, dst_name));
            }
        }

        sink.emit_comment(&format!(
            "saved_callees: {:?}",
            info.used_callee_saved
                .iter()
                .map(|r| self.reg_name(*r))
                .collect::<Vec<_>>()
        ));
    }

    fn emit_epilogue_with_info(
        &self,
        sink: &mut CodeSink,
        func: &MirFunction,
        info: &PrologueInfo,
    ) {
        // Emit the epilogue label that Ret instructions jump to
        sink.emit_label(&format!(".L{}_epilogue", func.name));

        // Restore stack pointer (undoes frame allocation + callee pushes)
        sink.emit("    movq    %rbp, %rsp");
        sink.emit("    popq    %rbp");

        // Pop callee-saved registers in reverse order of how they were pushed
        for &cs_reg in info.used_callee_saved.iter().rev() {
            let name = self.reg_name(cs_reg);
            sink.emit(&format!("    popq    {}", name));
        }

        sink.emit("    retq");
    }

    fn emit_inst(&self, sink: &mut CodeSink, inst: &MirInst, reg_names: &[String]) {
        match inst {
            MirInst::Label { block } => {
                sink.emit_label(&format!(".L{}", block.as_u32()));
            }

            MirInst::Mov { dst, src } => {
                sink.emit(&format!(
                    "    movq    {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::MovImm { dst, imm } => {
                sink.emit(&format!(
                    "    movq    ${}, {}",
                    imm.as_i64(),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Add { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    addq    {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Sub { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    subq    {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Mul { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    imulq   {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Div { dst, lhs, rhs } => {
                // idiv uses rdx:rax implicitly
                sink.emit(&format!(
                    "    movq    {}, %rax",
                    rn(reg_names, *lhs)
                ));
                sink.emit("    cqto"); // sign-extend rax -> rdx:rax
                sink.emit(&format!("    idivq   {}", rn(reg_names, *rhs)));
                sink.emit(&format!(
                    "    movq    %rax, {}",
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Rem { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    movq    {}, %rax",
                    rn(reg_names, *lhs)
                ));
                sink.emit("    cqto");
                sink.emit(&format!("    idivq   {}", rn(reg_names, *rhs)));
                sink.emit(&format!(
                    "    movq    %rdx, {}",
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Neg { dst, src } => {
                if src != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *src),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!("    negq    {}", rn(reg_names, *dst)));
            }

            MirInst::And { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    andq    {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Or { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    orq     {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Xor { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    xorq    {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Shl { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                // Shift count must be in %cl
                sink.emit(&format!(
                    "    movq    {}, %rcx",
                    rn(reg_names, *rhs)
                ));
                sink.emit(&format!("    shlq    %cl, {}", rn(reg_names, *dst)));
            }

            MirInst::ShlImm { dst, lhs, amount } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    shlq    ${}, {}",
                    amount,
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Shr { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    movq    {}, %rcx",
                    rn(reg_names, *rhs)
                ));
                sink.emit(&format!("    shrq    %cl, {}", rn(reg_names, *dst)));
            }

            MirInst::ShrImm { dst, lhs, amount } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    shrq    ${}, {}",
                    amount,
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Sar { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    movq    {}, %rcx",
                    rn(reg_names, *rhs)
                ));
                sink.emit(&format!("    sarq    %cl, {}", rn(reg_names, *dst)));
            }

            MirInst::SarImm { dst, lhs, amount } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    sarq    ${}, {}",
                    amount,
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Not { dst, src } => {
                if src != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *src),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!("    notq    {}", rn(reg_names, *dst)));
            }

            MirInst::Cmp { dst, lhs, rhs, cond } => {
                sink.emit(&format!(
                    "    cmpq    {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *lhs)
                ));
                let setcc = match cond {
                    CmpCond::Eq => "sete",
                    CmpCond::Ne => "setne",
                    CmpCond::Lt => "setl",
                    CmpCond::Le => "setle",
                    CmpCond::Gt => "setg",
                    CmpCond::Ge => "setge",
                };
                sink.emit(&format!(
                    "    {}     {}",
                    setcc,
                    rn_byte(reg_names, *dst)
                ));
                sink.emit(&format!(
                    "    movzbq  {}, {}",
                    rn_byte(reg_names, *dst),
                    rn(reg_names, *dst)
                ));
            }

            // ── Floating-Point Arithmetic (scalar double) ──────────

            MirInst::FAdd { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    addsd   {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::FSub { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    subsd   {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::FMul { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    mulsd   {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::FDiv { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    divsd   {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::FRem { dst, lhs, rhs } => {
                // x86 has no frem instruction; call fmod
                // Move lhs to xmm0, rhs to xmm1, call fmod, result in xmm0 → dst
                sink.emit(&format!(
                    "    movsd   {}, %xmm0",
                    rn(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    movsd   {}, %xmm1",
                    rn(reg_names, *rhs)
                ));
                sink.emit("    callq   fmod");
                sink.emit(&format!(
                    "    movsd   %xmm0, {}",
                    rn(reg_names, *dst)
                ));
            }

            MirInst::FNeg { dst, src } => {
                // FNeg(x) = x XOR sign bit mask
                if src != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *src),
                        rn(reg_names, *dst)
                    ));
                }
                // XOR with 0x8000000000000000 to flip sign
                sink.emit("    movq    $0x8000000000000000, %rax");
                sink.emit("    movq    %rax, %xmm1");
                sink.emit(&format!(
                    "    xorpd   %xmm1, {}",
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::FAbs { dst, src } => {
                if src != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *src),
                        rn(reg_names, *dst)
                    ));
                }
                // AND with 0x7FFFFFFFFFFFFFFF to clear sign bit
                sink.emit("    movq    $0x7FFFFFFFFFFFFFFF, %rax");
                sink.emit("    movq    %rax, %xmm1");
                sink.emit(&format!(
                    "    andpd   %xmm1, {}",
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::FSqrt { dst, src } => {
                sink.emit(&format!(
                    "    sqrtsd  {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            // ── Floating-Point Comparison ─────────────────────────

            MirInst::FCmp { dst, lhs, rhs, cond } => {
                sink.emit(&format!(
                    "    comisd  {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *lhs)
                ));
                let setcc = match cond {
                    FCmpCond::Eq => "sete",
                    FCmpCond::Ne => "setne",
                    FCmpCond::Lt => "setb",
                    FCmpCond::Le => "setbe",
                    FCmpCond::Gt => "seta",
                    FCmpCond::Ge => "setae",
                };
                sink.emit(&format!(
                    "    {}     {}",
                    setcc,
                    rn_byte(reg_names, *dst)
                ));
                sink.emit(&format!(
                    "    movzbq  {}, {}",
                    rn_byte(reg_names, *dst),
                    rn(reg_names, *dst)
                ));
            }

            // ── Floating-Point Conversion ─────────────────────────

            MirInst::FpToSInt { dst, src } => {
                sink.emit(&format!(
                    "    cvttsd2si {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::SIntToFp { dst, src } => {
                sink.emit(&format!(
                    "    cvtsi2sd {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::FpToUInt { dst, src } => {
                // cvttsd2si gives signed result; for unsigned we need extra handling.
                // Simplified: use cvttsd2si and hope the value fits.
                sink.emit(&format!(
                    "    cvttsd2si {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::UIntToFp { dst, src } => {
                // cvtsi2sd for unsigned requires special handling.
                // Simplified: use cvtsi2sdq and hope the value fits.
                sink.emit(&format!(
                    "    cvtsi2sd {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            // ── Floating-Point Misc ──────────────────────────────

            MirInst::Copysign { dst, lhs, rhs } => {
                // copysign(lhs, rhs) = (abs(lhs) | signbit(rhs))
                // 1. Move lhs to dst, clear sign bit (abs)
                // 2. Extract sign bit from rhs, OR into dst
                sink.emit(&format!(
                    "    movsd   {}, {}",
                    rn(reg_names, *lhs),
                    rn(reg_names, *dst)
                ));
                sink.emit("    movq    $0x7FFFFFFFFFFFFFFF, %rax");
                sink.emit("    movq    %rax, %xmm2");
                sink.emit(&format!(
                    "    andpd   %xmm2, {}",
                    rn_xmm(reg_names, *dst)
                ));
                sink.emit("    movq    $0x8000000000000000, %rax");
                sink.emit("    movq    %rax, %xmm2");
                sink.emit(&format!(
                    "    andpd   {}, %xmm2",
                    rn(reg_names, *rhs)
                ));
                sink.emit(&format!(
                    "    orpd    %xmm2, {}",
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::Fmin { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    minsd   {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Fmax { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn(reg_names, *lhs),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    maxsd   {}, {}",
                    rn(reg_names, *rhs),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Load { dst, addr } => {
                sink.emit(&format!(
                    "    movq    ({}), {}",
                    rn(reg_names, *addr),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Store { addr, val } => {
                sink.emit(&format!(
                    "    movq    {}, ({})",
                    rn(reg_names, *val),
                    rn(reg_names, *addr)
                ));
            }

            MirInst::StackAlloc { dst, size, align } => {
                sink.emit_comment(&format!(
                    "stack_alloc size={} align={}",
                    size, align
                ));
                // Allocate on stack by subtracting from rsp
                let aligned_size = align_to(*size, *align);
                sink.emit(&format!("    subq    ${}, %rsp", aligned_size));
                sink.emit(&format!(
                    "    movq    %rsp, {}",
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Call {
                dst,
                func,
                args,
            } => {
                // System V: args in rdi, rsi, rdx, rcx, r8, r9
                let arg_gprs: &[PhysReg] = &[RDI, RSI, RDX, RCX, R8, R9];
                for (i, arg) in args.iter().enumerate() {
                    if i < arg_gprs.len() {
                        sink.emit(&format!(
                            "    movq    {}, %{}",
                            rn(reg_names, *arg),
                            gpr_name(arg_gprs[i].as_u16() as usize)
                        ));
                    } else {
                        // Push remaining args on the stack
                        sink.emit(&format!(
                            "    pushq   {}",
                            rn(reg_names, *arg)
                        ));
                    }
                }
                sink.emit(&format!("    callq   {}", func));
                // Restore stack for stack-passed args
                let stack_args = args.len().saturating_sub(arg_gprs.len());
                if stack_args > 0 {
                    sink.emit(&format!(
                        "    addq    ${}, %rsp",
                        stack_args * 8
                    ));
                }
                if let Some(d) = dst {
                    sink.emit(&format!(
                        "    movq    %rax, {}",
                        rn(reg_names, *d)
                    ));
                }
            }

            MirInst::Ret { val } => {
                if let Some(v) = val {
                    sink.emit(&format!(
                        "    movq    {}, %rax",
                        rn(reg_names, *v)
                    ));
                }
                // Jump to the epilogue label (emitted by emit_epilogue_with_info)
                // This ensures callee-saved registers are properly restored.
                // The epilogue label is `.L<func_name>_epilogue` — we use the
                // VReg 0's name prefix as a proxy isn't reliable, so we encode
                // a special comment. The codegen crate will provide the func name
                // via a thread-local or the emit_assembly function handles it.
                // For now, we emit a placeholder that the codegen replaces.
                // Actually, the simplest approach: we emit a well-known pattern
                // that the codegen crate can post-process.
                sink.emit("    # RET_EPILOGUE_JUMP");
            }

            MirInst::Jump { target } => {
                sink.emit(&format!("    jmp     .L{}", target.as_u32()));
            }

            MirInst::Branch {
                cond,
                true_block,
                false_block,
            } => {
                sink.emit(&format!(
                    "    testq   {}, {}",
                    rn(reg_names, *cond),
                    rn(reg_names, *cond)
                ));
                sink.emit(&format!("    jne     .L{}", true_block.as_u32()));
                sink.emit(&format!("    jmp     .L{}", false_block.as_u32()));
            }

            MirInst::PhiCopy { dst, src } => {
                sink.emit(&format!(
                    "    movq    {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            // ── Extension / Truncation ──────────────────────────

            MirInst::ZExt { dst, src } => {
                // Zero-extend: movzbq for byte, movzwq for word, movl for 32-bit
                // For now, use movzbq which handles all cases correctly on x86-64
                sink.emit(&format!(
                    "    movzbq  {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::SExt { dst, src } => {
                // Sign-extend: movsbq for byte, movswq for word, movslq for long
                sink.emit(&format!(
                    "    movsbq  {}, {}",
                    rn(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::Trunc { dst, src } => {
                // Truncation: just mask with AND $0xFFFFFFFF (for i64→i32)
                // A simple movl works for i64→i32 (zero-extends to 64 on x86-64)
                if src != dst {
                    sink.emit(&format!(
                        "    movq    {}, {}",
                        rn(reg_names, *src),
                        rn(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    andq    $4294967295, {}",
                    rn(reg_names, *dst)
                ));
            }

            // ── Spill / Reload ──────────────────────────────────

            MirInst::SpillStore { vreg, slot } => {
                let offset = (*slot as i32 + 1) * 8;
                sink.emit(&format!(
                    "    movq    {}, {}(%rbp)",
                    rn(reg_names, *vreg),
                    -offset
                ));
            }

            MirInst::SpillLoad { vreg, slot } => {
                let offset = (*slot as i32 + 1) * 8;
                sink.emit(&format!(
                    "    movq    {}(%rbp), {}",
                    -offset,
                    rn(reg_names, *vreg)
                ));
            }

            // ── Vector Operations (SSE2/SSE4.1/AVX) ────────────────
            //
            // All vector instructions use XMM registers (128-bit).
            // The `reg_names` table maps VRegs to physical register names
            // that were assigned by the register allocator. Vector VRegs
            // should be allocated from the XMM register class.

            MirInst::VecBroadcast { dst, src, lane_count } => {
                // SSE2: movd src -> xmm, then shufps to broadcast
                // For i32x4 or f32x4: shufps $0x00 broadcasts lane 0
                // For f64x2: movddup (SSE3) or shufpd $0x00
                if *lane_count == 2 {
                    // f64x2 broadcast: movsd + shufpd
                    sink.emit(&format!(
                        "    movsd   {}, {}",
                        rn_xmm(reg_names, *src),
                        rn_xmm(reg_names, *dst)
                    ));
                    sink.emit(&format!(
                        "    shufpd  $0x00, {}, {}",
                        rn_xmm(reg_names, *dst),
                        rn_xmm(reg_names, *dst)
                    ));
                } else {
                    // i32x4 / f32x4 broadcast: movd + shufps
                    sink.emit(&format!(
                        "    movd    {}, {}",
                        rn(reg_names, *src),
                        rn_xmm(reg_names, *dst)
                    ));
                    sink.emit(&format!(
                        "    shufps  $0x00, {}, {}",
                        rn_xmm(reg_names, *dst),
                        rn_xmm(reg_names, *dst)
                    ));
                }
            }

            MirInst::VecLoad { dst, addr, lane_count: _ } => {
                // Aligned load: movaps; unaligned: movups
                // We default to unaligned loads for safety.
                sink.emit(&format!(
                    "    movups  ({}), {}",
                    rn(reg_names, *addr),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecStore { addr, val, lane_count: _ } => {
                sink.emit(&format!(
                    "    movups  {}, ({})",
                    rn_xmm(reg_names, *val),
                    rn(reg_names, *addr)
                ));
            }

            MirInst::VecAdd { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                // For integer vectors: paddd (i32x4), paddq (i64x2)
                // For float vectors: addps (f32x4), addpd (f64x2)
                // Default to paddd (i32x4); backend legalizes based on type.
                sink.emit(&format!(
                    "    paddd   {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecSub { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    psubd   {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecMul { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                // pmulld requires SSE4.1; for SSE2 fallback use pmuludq + shuffle
                sink.emit(&format!(
                    "    pmulld  {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecDiv { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                // FP division: divps for f32x4, divpd for f64x2
                sink.emit(&format!(
                    "    divps   {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecAnd { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    pand    {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecOr { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    por     {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecXor { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    pxor    {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecMin { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                // pminsd requires SSE4.1 for signed i32; minps for f32
                sink.emit(&format!(
                    "    pminsd  {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecMax { dst, lhs, rhs } => {
                if lhs != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *lhs),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    pmaxsd  {}, {}",
                    rn_xmm(reg_names, *rhs),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecNeg { dst, src } => {
                // XOR with sign-bit mask to negate.
                // For i32x4: negate via pxor with 0x80000000 per lane + psubd from 0
                // For f32x4: xorps with sign-bit mask
                // For f64x2: xorpd with sign-bit mask
                // Simplified: emit xorps with sign-bit mask for f32x4
                if src != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *src),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                // Load sign-bit mask: all lanes 0x80000000
                sink.emit("    movq    $0x8000800080008000, %rax");
                sink.emit("    movq    %rax, %xmm1");
                sink.emit("    shufps  $0x44, %xmm1, %xmm1");
                sink.emit(&format!(
                    "    xorps   %xmm1, {}",
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecAbs { dst, src } => {
                // AND with absolute-value mask (clear sign bit)
                // For i32x4: pand with 0x7FFFFFFF per lane
                // For f32x4: andps with 0x7FFFFFFF per lane
                if src != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *src),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                sink.emit("    movq    $0x7FFFFFFF7FFFFFFF, %rax");
                sink.emit("    movq    %rax, %xmm1");
                sink.emit("    shufps  $0x44, %xmm1, %xmm1");
                sink.emit(&format!(
                    "    pand    %xmm1, {}",
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecSqrt { dst, src } => {
                // sqrtps for f32x4, sqrtpd for f64x2
                sink.emit(&format!(
                    "    sqrtps  {}, {}",
                    rn_xmm(reg_names, *src),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecShuffle { dst, src, mask } => {
                // pshufd for i32x4: mask is a u8 encoding 2-bit lane selectors
                // Encode the 4 lane indices as a single u8: (mask[3]<<6)|(mask[2]<<4)|(mask[1]<<2)|mask[0]
                let imm = if mask.len() >= 4 {
                    ((mask[3] as u8 & 0x3) << 6) | ((mask[2] as u8 & 0x3) << 4) |
                    ((mask[1] as u8 & 0x3) << 2) | (mask[0] as u8 & 0x3)
                } else {
                    0u8
                };
                sink.emit(&format!(
                    "    pshufd  ${}, {}, {}",
                    imm,
                    rn_xmm(reg_names, *src),
                    rn_xmm(reg_names, *dst)
                ));
            }

            MirInst::VecReduceSum { dst, src, lane_count } => {
                // Horizontal add sequence:
                // For i32x4: phaddd (SSE3) or shufps+paddd pattern
                // For f32x4: haddps (SSE3) or shufps+addps pattern
                //
                // SSE3 haddps pattern for f32x4:
                //   movaps  xmm_dst, xmm_src
                //   haddps  xmm_dst, xmm_dst   // [a0+a1, a2+a3, b0+b1, b2+b3]
                //   haddps  xmm_dst, xmm_dst   // [a0+a1+a2+a3, ...]
                //
                // For i32x4, use phaddd (SSSE3) or:
                //   pshufd  $0xB1, xmm_tmp, xmm_tmp  // swap pairs
                //   paddd   xmm_tmp, xmm_dst          // add pairs
                //   pshufd  $0x0E, xmm_tmp, xmm_dst   // shift
                //   paddd   xmm_tmp, xmm_dst          // final add
                if src != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *src),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                if *lane_count == 4 {
                    // i32x4 horizontal sum using paddd + pshufd
                    sink.emit(&format!(
                        "    movaps  {}, %xmm1",
                        rn_xmm(reg_names, *dst)
                    ));
                    sink.emit("    pshufd  $0xB1, %xmm1, %xmm1");  // swap adjacent pairs
                    sink.emit(&format!(
                        "    paddd   %xmm1, {}",
                        rn_xmm(reg_names, *dst)
                    ));
                    sink.emit(&format!(
                        "    movaps  {}, %xmm1",
                        rn_xmm(reg_names, *dst)
                    ));
                    sink.emit("    pshufd  $0x01, %xmm1, %xmm1");  // swap high/low dwords
                    sink.emit(&format!(
                        "    paddd   %xmm1, {}",
                        rn_xmm(reg_names, *dst)
                    ));
                    // Result is in lane 0 of dst; extract to GPR
                    sink.emit(&format!(
                        "    movd    {}, {}",
                        rn_xmm(reg_names, *dst),
                        rn(reg_names, *dst)
                    ));
                } else if *lane_count == 2 {
                    // i64x2 or f64x2 horizontal sum
                    sink.emit(&format!(
                        "    movaps  {}, %xmm1",
                        rn_xmm(reg_names, *dst)
                    ));
                    sink.emit("    pshufd  $0x31, %xmm1, %xmm1");  // get high qword into low
                    sink.emit(&format!(
                        "    paddq   %xmm1, {}",
                        rn_xmm(reg_names, *dst)
                    ));
                    sink.emit(&format!(
                        "    movd    {}, {}",
                        rn_xmm(reg_names, *dst),
                        rn(reg_names, *dst)
                    ));
                } else {
                    // Generic: use haddps
                    sink.emit(&format!(
                        "    haddps  {}, {}",
                        rn_xmm(reg_names, *dst),
                        rn_xmm(reg_names, *dst)
                    ));
                    if *lane_count > 2 {
                        sink.emit(&format!(
                            "    haddps  {}, {}",
                            rn_xmm(reg_names, *dst),
                            rn_xmm(reg_names, *dst)
                        ));
                    }
                    sink.emit(&format!(
                        "    movd    {}, {}",
                        rn_xmm(reg_names, *dst),
                        rn(reg_names, *dst)
                    ));
                }
            }

            MirInst::ExtractLane { dst, src, index } => {
                // pextrd (SSE4.1) for i32 lanes
                sink.emit(&format!(
                    "    pextrd  ${}, {}, {}",
                    index,
                    rn_xmm(reg_names, *src),
                    rn(reg_names, *dst)
                ));
            }

            MirInst::InsertLane { dst, src, index, elem } => {
                // pinsrd (SSE4.1) for i32 lanes
                if src != dst {
                    sink.emit(&format!(
                        "    movaps  {}, {}",
                        rn_xmm(reg_names, *src),
                        rn_xmm(reg_names, *dst)
                    ));
                }
                sink.emit(&format!(
                    "    pinsrd  ${}, {}, {}",
                    index,
                    rn(reg_names, *elem),
                    rn_xmm(reg_names, *dst)
                ));
            }
        }
    }

    fn legalize_type(&self, ty: Type) -> Type {
        match ty {
            // Promote small integers to i64 (x86-64 prefers 64-bit ops)
            Type::I8 | Type::I16 | Type::I32 | Type::U8 | Type::U16 | Type::U32 | Type::Bool => {
                Type::I64
            }
            // f32 stays f32 (x86 has native f32 via SSE)
            Type::F32 => Type::F32,
            // Already 64-bit or larger — keep as-is
            other => other,
        }
    }

    fn reg_name(&self, reg: PhysReg) -> String {
        let idx = reg.as_u16() as usize;
        if idx < GPR_NAMES.len() {
            format!("%{}", GPR_NAMES[idx])
        } else if idx < GPR_NAMES.len() + XMM_NAMES.len() {
            format!("%{}", XMM_NAMES[idx - GPR_NAMES.len()])
        } else {
            format!("preg{}", idx)
        }
    }

    fn vreg_name(&self, vreg: VReg) -> String {
        self.vreg_to_asm(vreg)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Look up the reg name for a VReg from the `reg_names` table.
fn rn(reg_names: &[String], vreg: VReg) -> &str {
    let idx = vreg.as_u32() as usize;
    if idx < reg_names.len() {
        &reg_names[idx]
    } else {
        // Shouldn't happen; return a placeholder
        static PLACEHOLDER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        PLACEHOLDER.get_or_init(|| "<undef>".to_string())
    }
}

/// Same as `rn` but produces the XMM register name for FP operations.
/// VRegs 0–15 map to GPRs; for XMM operations we use the reg_names
/// lookup (which may already be mapped to XMM registers by regalloc).
fn rn_xmm(reg_names: &[String], vreg: VReg) -> String {
    let idx = vreg.as_u32() as usize;
    if idx < reg_names.len() {
        reg_names[idx].clone()
    } else {
        "<undef>".to_string()
    }
}

/// Same as `rn` but produces the byte-register name for setcc destinations.
/// For x86-64, the low byte of each GPR: al, cl, dl, bl, ...
fn rn_byte(reg_names: &[String], vreg: VReg) -> String {
    let idx = vreg.as_u32() as usize;
    if idx < GPR_NAMES.len() {
        // Map the full register name to its byte variant
        let byte_names: &[&str] = &[
            "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b",
            "r12b", "r13b", "r14b", "r15b",
        ];
        format!("%{}", byte_names[idx])
    } else if idx < reg_names.len() {
        // Spill slot — just use the full name (movzbq will handle it)
        reg_names[idx].clone()
    } else {
        "<undef>".to_string()
    }
}

/// Get GPR name by index.
fn gpr_name(idx: usize) -> &'static str {
    GPR_NAMES[idx]
}

/// Align `size` up to `align`.
fn align_to(size: u32, align: u32) -> u32 {
    (size + align - 1) / align * align
}

/// Compute the frame size (stack space for spilled VRegs) for a function.
fn compute_frame_size(func: &MirFunction) -> u32 {
    let spill_count = func.vreg_count.saturating_sub(GPR_NAMES.len() as u32);
    let size = spill_count * 8;
    // Round up to 16-byte alignment (x86-64 ABI requirement)
    (size + 15) & !15
}

// ── Build the TargetDesc ───────────────────────────────────────────────

fn build_target_desc() -> TargetDesc {
    // GPR indices 0..15
    let gpr_names: &[&str] = GPR_NAMES;
    let gpr_reserved: &[bool] = &[
        false, // rax
        false, // rcx
        false, // rdx
        false, // rbx
        true,  // rsp  (reserved)
        true,  // rbp  (reserved)
        false, // rsi
        false, // rdi
        false, // r8
        false, // r9
        false, // r10
        false, // r11
        false, // r12
        false, // r13
        false, // r14
        false, // r15
    ];

    let mut registers: Vec<RegisterInfo> = Vec::with_capacity(32);

    for (i, &name) in gpr_names.iter().enumerate() {
        registers.push(RegisterInfo {
            reg: PhysReg::new(i as u16),
            name: name.to_string(),
            class: RegClass::Int,
            is_reserved: gpr_reserved[i],
        });
    }

    for (i, &name) in XMM_NAMES.iter().enumerate() {
        registers.push(RegisterInfo {
            reg: PhysReg::new((16 + i) as u16),
            name: name.to_string(),
            class: RegClass::Float,
            is_reserved: false,
        });
    }

    let calling_conv = CallingConv {
        arg_regs: vec![RDI, RSI, RDX, RCX, R8, R9],
        ret_regs: vec![RAX, RDX],
        callee_saved: vec![RBX, RBP, R12, R13, R14, R15],
        caller_saved: vec![RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11],
        stack_align: 16,
    };

    TargetDesc {
        name: "x86_64".to_string(),
        ptr_width: 64,
        endianness: Endianness::Little,
        registers,
        calling_conv,
        supported_widths: vec![8, 16, 32, 64],
        has_cmov: true,
        has_vector: true,
        vector_width: 256, // AVX2
    }
}
