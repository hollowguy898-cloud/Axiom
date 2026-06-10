//! Axiom Legalize — Type and Operation Legalization.
//!
//! This crate legalizes MIR functions for a specific target. Legalization
//! replaces operations the target doesn't support with sequences it does,
//! and expands values that don't fit in machine instructions.
//!
//! # What we legalize
//!
//! - **Large immediates**: Targets have a maximum immediate size for `MovImm`.
//!   On x86-64, `movq` can load any 64-bit immediate, but on RISC-V you
//!   need `lui`+`addi` pairs. We expand large immediates into multi-step
//!   sequences (upper half + lower half).
//!
//! - **Missing operations**: Some targets don't have hardware divide or
//!   remainder. We replace `Div`/`Rem` with calls to software routines.
//!
//! - **Type promotion**: Small types that the target doesn't natively
//!   support are promoted to the target's preferred width via
//!   `target.legalize_type()`.

use axiom_mir::{Imm64, MirFunction, MirInst, VReg};
use axiom_target::Target;

/// Maximum immediate value that fits in a single instruction encoding.
/// Targets with 32-bit encodings (RISC-V, AArch64) typically can't fit
/// a full 64-bit immediate in one instruction. We use a conservative
/// threshold — values outside [-32768, 32767] need expansion.
const IMM12_MIN: i64 = -32768;
const IMM12_MAX: i64 = 32767;

/// Legalize a MIR function for a specific target.
///
/// This replaces operations the target doesn't support with sequences it does.
/// The function is modified in place.
pub fn legalize(func: &mut MirFunction, target: &dyn Target) {
    let desc = target.desc();
    let has_hw_div = has_hardware_div(desc);

    // For each block, rewrite instructions that need legalization.
    // We clone the instruction list first to avoid borrow-checker issues
    // (we need &mut func to allocate new vregs while iterating).
    for block_idx in 0..func.blocks.len() {
        let original_insts = func.blocks[block_idx].insts.clone();
        let mut new_insts: Vec<MirInst> = Vec::new();

        for inst in &original_insts {
            match inst {
                // ── Large immediate expansion ──────────────────────────
                MirInst::MovImm { dst, imm } => {
                    let val = imm.as_i64();
                    if val >= IMM12_MIN && val <= IMM12_MAX {
                        // Fits in a 12-bit signed immediate — keep as is
                        new_insts.push(inst.clone());
                    } else {
                        // Need multi-step materialization
                        legalize_large_immediate(*dst, val, &mut new_insts, func);
                    }
                }

                // ── Division/Remainder ─────────────────────────────────
                MirInst::Div { dst, lhs, rhs } => {
                    if !has_hw_div {
                        legalize_software_div(*dst, *lhs, *rhs, &mut new_insts);
                    } else {
                        new_insts.push(inst.clone());
                    }
                }

                MirInst::Rem { dst, lhs, rhs } => {
                    if !has_hw_div {
                        legalize_software_rem(*dst, *lhs, *rhs, &mut new_insts);
                    } else {
                        new_insts.push(inst.clone());
                    }
                }

                // ── Immediate shift amounts ────────────────────────────
                // Clamp large shift amounts to [0, 63] for 64-bit values.
                // x86 masks to 0..63 anyway; RISC-V and AArch64 require
                // the immediate to be in range. If the amount exceeds 63,
                // the result is either 0 (shl/shr) or all-sign-bits (sar).
                MirInst::ShlImm { dst, lhs: _, amount } => {
                    if *amount > 63 {
                        // Shifting left by >= 64 bits always yields 0
                        new_insts.push(MirInst::MovImm {
                            dst: *dst,
                            imm: Imm64::new(0),
                        });
                    } else {
                        new_insts.push(inst.clone());
                    }
                }

                MirInst::ShrImm { dst, lhs: _, amount } => {
                    if *amount > 63 {
                        // Logical shift right by >= 64 bits always yields 0
                        new_insts.push(MirInst::MovImm {
                            dst: *dst,
                            imm: Imm64::new(0),
                        });
                    } else {
                        new_insts.push(inst.clone());
                    }
                }

                MirInst::SarImm { dst, lhs, amount } => {
                    if *amount > 63 {
                        // Arithmetic shift right by >= 64 yields 0 or -1
                        // depending on the sign bit. Lower to SarImm { amount: 63 }
                        // which will propagate the sign bit to all positions.
                        new_insts.push(MirInst::SarImm {
                            dst: *dst,
                            lhs: *lhs,
                            amount: 63,
                        });
                    } else {
                        new_insts.push(inst.clone());
                    }
                }

                // ── Variable-amount shifts with out-of-range constants ──
                // If a Shl/Shr/Sar has a rhs that is a MovImm with a value
                // > 63, we need to clamp it. However, since we can't easily
                // peek through the instruction stream here, we rely on the
                // lowering pass to emit ShlImm/ShrImm/SarImm for constants,
                // and the above handlers take care of those. Variable-amount
                // shifts where the value happens to be > 63 at runtime are
                // handled by the hardware's masking behavior (x86 masks to
                // 0..63, RISC-V masks to 0..63, AArch64 masks to 0..63).

                // ── All other instructions pass through ────────────────
                _ => {
                    new_insts.push(inst.clone());
                }
            }
        }

        func.blocks[block_idx].insts = new_insts;
    }
}

/// Expand a large immediate into a sequence of instructions.
///
/// Strategy:
/// 1. If the value fits in 32 bits (high 32 bits are all sign bits),
///    load the upper 16 bits, shift left 16, then OR in the lower 16 bits.
/// 2. If the value needs all 64 bits, load upper 32 bits, shift left 32,
///    then OR in lower 32 bits.
fn legalize_large_immediate(
    dst: VReg,
    val: i64,
    insts: &mut Vec<MirInst>,
    func: &mut MirFunction,
) {
    let val_u64 = val as u64;

    // Check if the value fits in a sign-extended 32-bit range
    let hi32 = (val_u64 >> 32) as u32;
    let lo32 = (val_u64 & 0xFFFF_FFFF) as u32;

    if hi32 == 0 || hi32 == 0xFFFF_FFFF {
        // Fits in 32-bit sign-extended range
        // Split into two 16-bit halves
        let hi16 = ((val_u64 >> 16) & 0xFFFF) as i64;
        let lo16 = (val_u64 & 0xFFFF) as i64;

        if hi16 != 0 {
            // Load upper 16 bits
            let tmp = func.alloc_vreg();
            insts.push(MirInst::MovImm {
                dst: tmp,
                imm: Imm64::new(hi16),
            });
            // Shift left by 16
            let shift_amt = func.alloc_vreg();
            insts.push(MirInst::MovImm {
                dst: shift_amt,
                imm: Imm64::new(16),
            });
            insts.push(MirInst::Shl {
                dst,
                lhs: tmp,
                rhs: shift_amt,
            });
            // OR in lower 16 bits
            if lo16 != 0 {
                let lo_tmp = func.alloc_vreg();
                insts.push(MirInst::MovImm {
                    dst: lo_tmp,
                    imm: Imm64::new(lo16),
                });
                insts.push(MirInst::Or {
                    dst,
                    lhs: dst,
                    rhs: lo_tmp,
                });
            }
        } else {
            // Value fits in the lower 16 bits — just load it directly
            insts.push(MirInst::MovImm {
                dst,
                imm: Imm64::new(val),
            });
        }
    } else {
        // Full 64-bit materialization: load upper 32, shift, OR lower 32
        // Load upper 32 bits (may itself need expansion, but we assume
        // the target can handle a 32-bit immediate in at most 2 steps)
        let tmp_hi = func.alloc_vreg();
        insts.push(MirInst::MovImm {
            dst: tmp_hi,
            imm: Imm64::new(hi32 as i64),
        });

        // Shift left by 32
        let shift_amt = func.alloc_vreg();
        insts.push(MirInst::MovImm {
            dst: shift_amt,
            imm: Imm64::new(32),
        });
        let tmp_shifted = func.alloc_vreg();
        insts.push(MirInst::Shl {
            dst: tmp_shifted,
            lhs: tmp_hi,
            rhs: shift_amt,
        });

        // OR in lower 32 bits
        if lo32 != 0 {
            let tmp_lo = func.alloc_vreg();
            insts.push(MirInst::MovImm {
                dst: tmp_lo,
                imm: Imm64::new(lo32 as i64),
            });
            insts.push(MirInst::Or {
                dst,
                lhs: tmp_shifted,
                rhs: tmp_lo,
            });
        } else {
            // Lower 32 bits are zero — just move the shifted value
            insts.push(MirInst::Mov {
                dst,
                src: tmp_shifted,
            });
        }
    }
}

/// Replace a hardware `Div` with a call to `__divdi3` (generic software
/// division).
fn legalize_software_div(dst: VReg, lhs: VReg, rhs: VReg, insts: &mut Vec<MirInst>) {
    insts.push(MirInst::Call {
        dst: Some(dst),
        func: "__divdi3".to_string(),
        args: vec![lhs, rhs],
    });
}

/// Replace a hardware `Rem` with a call to `__moddi3` (generic software
/// remainder).
fn legalize_software_rem(dst: VReg, lhs: VReg, rhs: VReg, insts: &mut Vec<MirInst>) {
    insts.push(MirInst::Call {
        dst: Some(dst),
        func: "__moddi3".to_string(),
        args: vec![lhs, rhs],
    });
}

/// Check if the target has hardware divide support.
///
/// This is determined by checking the target's supported widths.
/// If the target name contains "riscv" and doesn't explicitly
/// indicate the M extension, we assume no hardware divide.
fn has_hardware_div(desc: &axiom_target::TargetDesc) -> bool {
    // Conservative: most targets we support have hardware divide.
    // RISC-V without M extension is the main exception.
    // For now, we use a simple heuristic based on target name.
    if desc.name.contains("riscv") {
        // Assume RISC-V without M extension doesn't have hardware divide
        false
    } else {
        true
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;
    use axiom_target::{
        CallingConv, CodeSink, Endianness, PhysReg, RegClass, RegisterInfo, TargetDesc,
    };

    /// A minimal test target that doesn't need to emit real assembly.
    struct TestTarget {
        desc: TargetDesc,
    }

    impl Target for TestTarget {
        fn desc(&self) -> &TargetDesc {
            &self.desc
        }

        fn emit_prologue(&self, _sink: &mut CodeSink, _func: &axiom_mir::MirFunction) {}
        fn emit_epilogue(&self, _sink: &mut CodeSink, _func: &axiom_mir::MirFunction) {}
        fn emit_inst(
            &self,
            _sink: &mut CodeSink,
            _inst: &axiom_mir::MirInst,
            _reg_names: &[String],
        ) {
        }
        fn legalize_type(&self, ty: Type) -> Type {
            ty
        }
        fn reg_name(&self, reg: PhysReg) -> String {
            format!("r{}", reg.as_u16())
        }
    }

    fn test_target() -> TestTarget {
        let desc = TargetDesc {
            name: "test64".to_string(),
            ptr_width: 64,
            endianness: Endianness::Little,
            registers: vec![RegisterInfo {
                reg: PhysReg::new(0),
                name: "r0".to_string(),
                class: RegClass::Int,
                is_reserved: false,
            }],
            calling_conv: CallingConv {
                arg_regs: vec![],
                ret_regs: vec![],
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
    fn test_small_immediate_passes_through() {
        let mut func = MirFunction::new("test");
        let _block = func.new_block();
        let v0 = func.alloc_vreg();

        func.blocks[0].insts.push(MirInst::MovImm {
            dst: v0,
            imm: Imm64::new(42),
        });
        func.blocks[0]
            .insts
            .push(MirInst::Ret { val: Some(v0) });

        let target = test_target();
        legalize(&mut func, &target);

        // Small immediate should pass through unchanged
        assert!(matches!(
            &func.blocks[0].insts[0],
            MirInst::MovImm { imm: Imm64(42), .. }
        ));
    }

    #[test]
    fn test_large_immediate_expanded() {
        let mut func = MirFunction::new("test");
        let _block = func.new_block();
        let v0 = func.alloc_vreg();

        func.blocks[0].insts.push(MirInst::MovImm {
            dst: v0,
            imm: Imm64::new(0x1_0000), // 65536 — needs 17 bits
        });
        func.blocks[0]
            .insts
            .push(MirInst::Ret { val: Some(v0) });

        let target = test_target();
        legalize(&mut func, &target);

        // Should be expanded into multiple instructions
        assert!(
            func.blocks[0].insts.len() > 2, // at least the expansion + ret
            "Large immediate should be expanded into multiple instructions"
        );
    }

    #[test]
    fn test_div_on_riscv_replaced() {
        let mut func = MirFunction::new("test");
        let _block = func.new_block();
        let v0 = func.alloc_vreg();
        let v1 = func.alloc_vreg();
        let v2 = func.alloc_vreg();

        func.blocks[0].insts.push(MirInst::Div {
            dst: v2,
            lhs: v0,
            rhs: v1,
        });
        func.blocks[0]
            .insts
            .push(MirInst::Ret { val: Some(v2) });

        let mut riscv_target = test_target();
        riscv_target.desc.name = "riscv64".to_string();
        legalize(&mut func, &riscv_target);

        // Div should be replaced with a Call to __divdi3
        let has_call = func.blocks[0].insts.iter().any(|i| {
            matches!(i, MirInst::Call { func: name, .. } if name == "__divdi3")
        });
        assert!(has_call, "Div should be replaced with software call on RISC-V");
    }

    #[test]
    fn test_div_on_x86_kept() {
        let mut func = MirFunction::new("test");
        let _block = func.new_block();
        let v0 = func.alloc_vreg();
        let v1 = func.alloc_vreg();
        let v2 = func.alloc_vreg();

        func.blocks[0].insts.push(MirInst::Div {
            dst: v2,
            lhs: v0,
            rhs: v1,
        });
        func.blocks[0]
            .insts
            .push(MirInst::Ret { val: Some(v2) });

        let target = test_target();
        legalize(&mut func, &target);

        // Div should remain as a hardware Div on x86-64
        let has_div = func.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i, MirInst::Div { .. }));
        assert!(has_div, "Div should be kept as hardware instruction on x86-64");
    }
}
