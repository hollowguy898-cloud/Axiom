//! Axiom WASM — backend target for WebAssembly (wasm32).
//!
//! Implements the `Target` trait from `axiom-target` with WAT
//! (WebAssembly Text Format) output instead of native assembly.
//! Uses stack-based semantics, 32-bit pointers, and SIMD128 support.

use axiom_ir::nodes::Type;
use axiom_mir::{CmpCond, FCmpCond, MirFunction, MirInst, VReg};
use axiom_target::{
    CallingConv, CodeSink, Endianness, PhysReg, RegClass, RegisterInfo, Target, TargetDesc,
};

// ── WASM32 Target ──────────────────────────────────────────────────────

/// WebAssembly 32-bit target backend (WAT text format output).
pub struct Wasm32Target {
    desc: TargetDesc,
}

impl Wasm32Target {
    pub fn new() -> Self {
        let desc = build_target_desc();
        Self { desc }
    }

    /// Map a VReg to a WAT local name: v0 → $l0, v1 → $l1, etc.
    fn vreg_to_wat(&self, vreg: VReg) -> String {
        format!("$l{}", vreg.as_u32())
    }
}

impl Default for Wasm32Target {
    fn default() -> Self {
        Self::new()
    }
}

impl Target for Wasm32Target {
    fn desc(&self) -> &TargetDesc {
        &self.desc
    }

    fn emit_prologue(&self, sink: &mut CodeSink, func: &MirFunction) {
        sink.emit_comment(&format!("function: {}", func.name));

        // (func $name (param $l0 i64) (param $l1 i64) ... (result i64)
        let mut header = format!("  (func ${}", func.name);

        // Params: first N vregs are function parameters
        for param in &func.params {
            header.push_str(&format!(" (param {} i64)", self.vreg_to_wat(*param)));
        }

        // Assume one i64 return value for now
        header.push_str(" (result i64)");

        // Locals: vregs beyond params
        let param_count = func.params.len() as u32;
        if func.vreg_count > param_count {
            let local_count = func.vreg_count - param_count;
            header.push_str(&format!("\n    (local {} i64)", local_count));
        }

        sink.emit(&header);
    }

    fn emit_epilogue(&self, sink: &mut CodeSink, _func: &MirFunction) {
        sink.emit("  )"); // close the (func ...)
    }

    fn emit_inst(&self, sink: &mut CodeSink, inst: &MirInst, reg_names: &[String]) {
        match inst {
            MirInst::Label { block } => {
                // WAT doesn't have explicit labels; use a comment
                sink.emit_comment(&format!("block bb{}", block.as_u32()));
            }

            MirInst::Mov { dst, src } => {
                // local.get $src  →  local.set $dst
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *src)
                ));
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::MovImm { dst, imm } => {
                sink.emit(&format!(
                    "    i64.const {}",
                    imm.as_i64()
                ));
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Add { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.add");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Sub { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.sub");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Mul { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.mul");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Div { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.div_s");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Rem { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.rem_s");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Neg { dst, src } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *src)
                ));
                sink.emit("    i64.const 0");
                sink.emit("    i64.sub");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::And { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.and");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Or { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.or");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Xor { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.xor");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Shl { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.shl");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::ShlImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!("    i64.const {}", amount));
                sink.emit("    i64.shl");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Shr { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.shr_u");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::ShrImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!("    i64.const {}", amount));
                sink.emit("    i64.shr_u");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Sar { dst, lhs, rhs } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                sink.emit("    i64.shr_s");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::SarImm { dst, lhs, amount } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!("    i64.const {}", amount));
                sink.emit("    i64.shr_s");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Not { dst, src } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *src)
                ));
                sink.emit("    i64.const -1");
                sink.emit("    i64.xor");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Cmp { dst, lhs, rhs, cond } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *lhs)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *rhs)
                ));
                let wasm_cmp = match cond {
                    CmpCond::Eq => "i64.eq",
                    CmpCond::Ne => "i64.ne",
                    CmpCond::Lt => "i64.lt_s",
                    CmpCond::Le => "i64.le_s",
                    CmpCond::Gt => "i64.gt_s",
                    CmpCond::Ge => "i64.ge_s",
                };
                sink.emit(&format!("    {}", wasm_cmp));
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Load { dst, addr } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *addr)
                ));
                sink.emit("    i64.load");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Store { addr, val } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *addr)
                ));
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *val)
                ));
                sink.emit("    i64.store");
            }

            MirInst::StackAlloc { dst, size, align } => {
                sink.emit_comment(&format!(
                    "stack_alloc size={} align={} — wasm uses linear memory",
                    size, align
                ));
                // In WASM, we'd need a stack pointer global; for now emit a
                // placeholder that notes the allocation.
                // A real implementation would track a shadow stack pointer.
                sink.emit(&format!(
                    "    ;; stack_alloc: {} bytes, align {}",
                    size, align
                ));
                // Use the stack pointer global to allocate
                sink.emit("    global.get $__stack_pointer");
                sink.emit(&format!("    i64.const {}", size));
                sink.emit("    i64.sub");
                sink.emit("    global.set $__stack_pointer");
                sink.emit("    global.get $__stack_pointer");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Call {
                dst,
                func,
                args,
            } => {
                for arg in args {
                    sink.emit(&format!(
                        "    local.get {}",
                        wat(reg_names, *arg)
                    ));
                }
                sink.emit(&format!("    call ${}", func));
                if dst.is_some() {
                    sink.emit(&format!(
                        "    local.set {}",
                        wat(reg_names, dst.unwrap())
                    ));
                }
            }

            MirInst::Ret { val } => {
                if let Some(v) = val {
                    sink.emit(&format!(
                        "    local.get {}",
                        wat(reg_names, *v)
                    ));
                } else {
                    // WASM requires a value on the stack matching the result type
                    sink.emit("    i64.const 0");
                }
                sink.emit("    return");
            }

            MirInst::Jump { target } => {
                sink.emit(&format!("    br ${}", target.as_u32()));
            }

            MirInst::Branch {
                cond,
                true_block,
                false_block,
            } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *cond)
                ));
                sink.emit(&format!(
                    "    br_if ${}",
                    true_block.as_u32()
                ));
                sink.emit(&format!("    br ${}", false_block.as_u32()));
            }

            MirInst::PhiCopy { dst, src } => {
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *src)
                ));
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            // ── Extension / Truncation ──────────────────────────
            // WASM types are explicit, so extension/truncation is essentially a no-op
            // (the type system handles it at the WASM level).

            MirInst::ZExt { dst, src } => {
                // Zero-extend is a no-op in WASM (types are explicit)
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *src)
                ));
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::SExt { dst, src } => {
                // Sign-extend is a no-op in WASM (types are explicit)
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *src)
                ));
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            MirInst::Trunc { dst, src } => {
                // Truncation in WASM: use i32.wrap_i64 for i64→i32
                sink.emit(&format!(
                    "    local.get {}",
                    wat(reg_names, *src)
                ));
                sink.emit("    i32.wrap_i64");
                sink.emit("    i64.extend_i32_u");
                sink.emit(&format!(
                    "    local.set {}",
                    wat(reg_names, *dst)
                ));
            }

            // ── Spill / Reload ──────────────────────────────────
            // WASM uses locals (no explicit spill/reload needed),
            // but we emit them as identity moves for consistency.

            MirInst::SpillStore { vreg, slot: _ } => {
                // Spill store: in WASM this is a no-op (locals are already on the stack)
                sink.emit_comment(&format!("spill_store v{}", vreg.as_u32()));
            }

            MirInst::SpillLoad { vreg, slot: _ } => {
                // Spill load: in WASM this is a no-op (locals are already on the stack)
                sink.emit_comment(&format!("spill_load v{}", vreg.as_u32()));
            }

            // ── Floating-Point Arithmetic (WASM f64) ──────────────

            MirInst::FAdd { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.add");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FSub { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.sub");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FMul { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.mul");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FDiv { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.div");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FRem { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.rem");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FNeg { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    f64.neg");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FAbs { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    f64.abs");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FSqrt { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    f64.sqrt");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FCmp { dst, lhs, rhs, cond } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                let wasm_fcmp = match cond {
                    FCmpCond::Eq => "f64.eq",
                    FCmpCond::Ne => "f64.ne",
                    FCmpCond::Lt => "f64.lt",
                    FCmpCond::Le => "f64.le",
                    FCmpCond::Gt => "f64.gt",
                    FCmpCond::Ge => "f64.ge",
                };
                sink.emit(&format!("    {}", wasm_fcmp));
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FpToSInt { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    i64.trunc_f64_s");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::SIntToFp { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    f64.convert_i64_s");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::FpToUInt { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    i64.trunc_f64_u");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::UIntToFp { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    f64.convert_i64_u");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::Copysign { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.copysign");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::Fmin { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.min");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            MirInst::Fmax { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f64.max");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }

            // ── Vector Operations (WASM SIMD128) ────────────────────
            MirInst::VecBroadcast { dst, src, lane_count: _ } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    i32x4.splat");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecLoad { dst, addr, lane_count: _ } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *addr)));
                sink.emit("    v128.load");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecStore { addr, val, lane_count: _ } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *addr)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *val)));
                sink.emit("    v128.store");
            }
            MirInst::VecAdd { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    i32x4.add");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecSub { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    i32x4.sub");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecMul { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    i32x4.mul");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecDiv { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    f32x4.div");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecAnd { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    v128.and");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecOr { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    v128.or");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecXor { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    v128.xor");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecMin { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    i32x4.min_s");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecMax { dst, lhs, rhs } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *lhs)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *rhs)));
                sink.emit("    i32x4.max_s");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecNeg { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    i32x4.neg");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecAbs { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    i32x4.abs");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecSqrt { dst, src } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    f32x4.sqrt");
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecShuffle { dst, src, mask } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                // WASM SIMD i8x16.shuffle takes 16 immediate byte indices
                let mut mask16 = [0u8; 16];
                for (i, &m) in mask.iter().enumerate() {
                    if i < 16 { mask16[i] = m; }
                }
                sink.emit(&format!(
                    "    i8x16.shuffle {}",
                    mask16.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(" ")
                ));
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::VecReduceSum { dst, src, lane_count: _ } => {
                // WASM SIMD doesn't have a direct horizontal sum;
                // emit a sequence of i32x4.extract_lane + i32.add
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit("    i32x4.extract_lane 0");
                for lane in 1..4u32 {
                    sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                    sink.emit(&format!("    i32x4.extract_lane {}", lane));
                    sink.emit("    i32.add");
                }
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::ExtractLane { dst, src, index } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit(&format!("    i32x4.extract_lane {}", index));
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
            MirInst::InsertLane { dst, src, index, elem } => {
                sink.emit(&format!("    local.get {}", wat(reg_names, *src)));
                sink.emit(&format!("    local.get {}", wat(reg_names, *elem)));
                sink.emit(&format!("    i32x4.replace_lane {}", index));
                sink.emit(&format!("    local.set {}", wat(reg_names, *dst)));
            }
        }
    }

    fn legalize_type(&self, ty: Type) -> Type {
        match ty {
            // WASM uses i32/i64 natively; promote small types
            Type::I8 | Type::I16 | Type::U8 | Type::U16 | Type::Bool => Type::I32,
            Type::I32 | Type::U32 => Type::I32,
            Type::I64 | Type::U64 | Type::Ptr => Type::I64,
            Type::F32 => Type::F32,
            Type::F64 => Type::F64,
            // i128/u128 not natively supported — lower to i64 pairs (placeholder)
            Type::I128 | Type::U128 => Type::I64,
            Type::Void | Type::Unknown => ty,
        }
    }

    fn reg_name(&self, _reg: PhysReg) -> String {
        // WASM has no physical registers; everything is a local
        "<wasm-local>".to_string()
    }

    fn vreg_name(&self, vreg: VReg) -> String {
        format!("$l{}", vreg.as_u32())
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn wat(reg_names: &[String], vreg: VReg) -> &str {
    let idx = vreg.as_u32() as usize;
    if idx < reg_names.len() {
        &reg_names[idx]
    } else {
        static PLACEHOLDER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        PLACEHOLDER.get_or_init(|| "$lundef".to_string())
    }
}

// ── Build the TargetDesc ───────────────────────────────────────────────

fn build_target_desc() -> TargetDesc {
    // WASM has no physical registers in the traditional sense.
    // We define a minimal set of "virtual" register entries for the
    // interface, but all code emission uses locals.

    let registers: Vec<RegisterInfo> = vec![
        // A single placeholder register to satisfy the interface
        RegisterInfo {
            reg: PhysReg::new(0),
            name: "stack".to_string(),
            class: RegClass::Special,
            is_reserved: true,
        },
        // SIMD128 virtual registers (v0–v31)
        RegisterInfo {
            reg: PhysReg::new(1),
            name: "simd_stack".to_string(),
            class: RegClass::Float,
            is_reserved: false,
        },
    ];

    let calling_conv = CallingConv {
        // WASM is stack-based: no register args
        arg_regs: vec![],
        ret_regs: vec![],
        callee_saved: vec![],
        caller_saved: vec![],
        stack_align: 4, // WASM alignment is naturally 4-byte
    };

    TargetDesc {
        name: "wasm32".to_string(),
        ptr_width: 32,
        endianness: Endianness::Little,
        registers,
        calling_conv,
        supported_widths: vec![32, 64],
        has_cmov: false,
        has_vector: true, // SIMD128 proposal
        vector_width: 128,
    }
}
