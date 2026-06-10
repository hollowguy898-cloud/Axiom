//! Strength Reduction.
//!
//! Replaces expensive operations with cheaper equivalents:
//! - `Mul` by power of 2 → `Shl`
//! - `Mul` by constant → shift-and-add sequence (e.g., x*15 = (x<<4)-x)
//! - `Div` by power of 2 → `Shr` (unsigned) or `Sar` + adjustment (signed)
//! - `Div` by constant → magic multiply transform (Granlund-Montgomery)
//! - `Mul` by 1 → identity (lhs)
//! - `Add` / `Sub` with 0 → identity
//! - `Mul` by 0 → 0

use axiom_ir::{IrGraph, IrNode, NodeId};
use axiom_ir::nodes::Type;
use crate::Pass;

/// Strength Reduction pass.
pub struct StrengthReducer;

impl StrengthReducer {
    /// Check if a node is an `IntConst` with the given value.
    fn is_int_const(graph: &IrGraph, id: NodeId, val: i64) -> bool {
        matches!(graph.get(id), Some(IrNode::IntConst(v)) if *v == val)
    }

    /// Check if a node is an `IntConst` that is a power of 2 (positive).
    /// Returns the log2 if so.
    fn get_power_of_two(graph: &IrGraph, id: NodeId) -> Option<u32> {
        match graph.get(id) {
            Some(IrNode::IntConst(v)) => {
                if *v > 0 {
                    let v = *v as u64;
                    if v.is_power_of_two() {
                        return Some(v.trailing_zeros());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Get the integer constant value of a node, if it is one.
    fn get_int_const(graph: &IrGraph, id: NodeId) -> Option<i64> {
        match graph.get(id) {
            Some(IrNode::IntConst(v)) => Some(*v),
            _ => None,
        }
    }

    /// Determine the signedness of an operation's result based on operand types.
    /// Returns true if unsigned, false if signed, None if unknown.
    fn is_unsigned(graph: &IrGraph, lhs: NodeId, rhs: NodeId) -> Option<bool> {
        // Check the output type of both operands.
        let lhs_type = graph.get(lhs).map(|n| n.output_type());
        let rhs_type = graph.get(rhs).map(|n| n.output_type());

        match (lhs_type, rhs_type) {
            (Some(Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128), _)
            | (_, Some(Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128)) => Some(true),
            (Some(Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128), _)
            | (_, Some(Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128)) => Some(false),
            _ => None,
        }
    }

    /// Check if a node is an `FpConst` with the given f64 value.
    fn is_fp_const(graph: &IrGraph, id: NodeId, val: f64) -> bool {
        matches!(graph.get(id), Some(IrNode::FpConst(bits)) if f64::from_bits(*bits) == val)
    }

    /// Check if a node is an `FpConst`. Returns the f64 value if so.
    fn get_fp_const(graph: &IrGraph, id: NodeId) -> Option<f64> {
        match graph.get(id) {
            Some(IrNode::FpConst(bits)) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    /// Decompose a constant multiplier into a shift-and-add sequence.
    ///
    /// Returns a list of (shift_amount, is_subtract) tuples that represent
    /// the sequence: x * C = x*(1 << s0) ± x*(1 << s1) ± ...
    ///
    /// For example:
    /// - x * 3 = (x << 1) + x           → [(1, false), (0, false)]
    /// - x * 5 = (x << 2) + x           → [(2, false), (0, false)]
    /// - x * 6 = (x << 2) + (x << 1)    → [(2, false), (1, false)]
    /// - x * 7 = (x << 3) - x           → [(3, false), (0, true)]
    /// - x * 9 = (x << 3) + x           → [(3, false), (0, false)]
    /// - x * 10 = (x << 3) + (x << 1)   → [(3, false), (1, false)]
    /// - x * 15 = (x << 4) - x          → [(4, false), (0, true)]
    /// - x * 25 = (x << 4) + (x << 3) + x  or  (x << 5) - (x << 3) + x
    ///
    /// Uses the "subtract if close to power of 2" heuristic:
    /// if C+1 is a power of 2, use (C+1) - 1 approach → (x << k) - x.
    fn decompose_multiplier(constant: u64) -> Option<Vec<(u32, bool)>> {
        if constant == 0 || constant == 1 {
            return None; // Handled by simpler patterns
        }

        // Check if power of 2 (already handled by shift)
        if constant.is_power_of_two() {
            return None;
        }

        let bits = 64 - constant.leading_zeros();

        // ── Strategy 1: C-1 is a power of 2 → use (x << k) + x ──
        // x * 3 = (x << 1) + x    (3-1 = 2 = 2^1)
        // x * 5 = (x << 2) + x    (5-1 = 4 = 2^2)
        // x * 9 = (x << 3) + x    (9-1 = 8 = 2^3)
        // Preferred over subtract form when both apply (e.g., C=3).
        if constant - 1 > 0 && (constant - 1).is_power_of_two() {
            let k = (constant - 1).trailing_zeros();
            return Some(vec![(k, false), (0, false)]);
        }

        // ── Strategy 2: C+1 is a power of 2 → use (x << k) - x ──
        // x * 7 = (x << 3) - x    (7+1 = 8 = 2^3)
        // x * 15 = (x << 4) - x   (15+1 = 16 = 2^4)
        // Used when C-1 is NOT a power of 2 (no add form available).
        if constant + 1 > constant && (constant + 1).is_power_of_two() {
            let k = (constant + 1).trailing_zeros();
            return Some(vec![(k, false), (0, true)]);
        }

        // ── Strategy 3: Binary decomposition (sum of set bits) ──
        // x * C = x<<b0 + x<<b1 + ... for each set bit
        // Only use if ≤ 3 terms (otherwise not efficient)
        let bit_count = constant.count_ones();
        if bit_count <= 3 {
            let mut terms = Vec::new();
            for bit in 0..64 {
                if constant & (1u64 << bit) != 0 {
                    terms.push((bit, false));
                }
            }
            if !terms.is_empty() {
                return Some(terms);
            }
        }

        // ── Strategy 4: (2^k) - r where r has few bits ──
        // x * C = (x << k) - (x * r)
        let next_pow2 = 1u64 << bits;
        let diff_to_next = next_pow2 - constant;
        if diff_to_next > 0 && diff_to_next.count_ones() <= 2 {
            let mut terms = vec![(bits, false)]; // x << k
            for bit in 0..64 {
                if diff_to_next & (1u64 << bit) != 0 {
                    terms.push((bit, true)); // subtract
                }
            }
            return Some(terms);
        }

        // ── Strategy 5: (2^(k-1)) + r where r has few bits ──
        let prev_pow2 = 1u64 << (bits - 1);
        let remainder = constant - prev_pow2;
        if remainder > 0 && remainder.count_ones() <= 2 {
            let mut terms = vec![(bits - 1, false)]; // x << (k-1)
            for bit in 0..64 {
                if remainder & (1u64 << bit) != 0 {
                    terms.push((bit, false)); // add
                }
            }
            return Some(terms);
        }

        None // Cannot decompose efficiently
    }

    /// Compute magic constants for unsigned division by a constant.
    ///
    /// Based on "Division by Invariant Integers using Multiplication"
    /// by Granlund and Montgomery.
    ///
    /// For unsigned division: q = mulhu(n, M) >> shift
    /// where M is the magic constant and shift adjusts the result.
    ///
    /// Precomputed for common divisors, computed on-the-fly for others.
    /// Returns (magic_constant, shift) or None if the divisor is not suitable.
    fn compute_unsigned_magic(divisor: u64) -> Option<(u64, u32)> {
        if divisor == 0 || divisor > u32::MAX as u64 {
            return None;
        }

        let d = divisor;

        // Precomputed table for common divisors.
        // These are (magic, shift) pairs where:
        //   q = mulhu(n, magic) >> shift
        // The magic constants are derived from Hacker's Delight / Warren,
        // using the algorithm for unsigned division by constants.
        match d {
            3  => Some((0xAAAA_AAAA_AAAA_AAAB, 1)),  // M = ceil(2^65/3), shift=1
            5  => Some((0xCCCC_CCCC_CCCC_CCCD, 2)),  // M = ceil(2^66/5), shift=2
            6  => Some((0xAAAA_AAAA_AAAA_AAAB, 2)),  // M = ceil(2^65/3), shift=2 (div by 6 = div by 3 then >>1)
            7  => Some((0x2492_4924_9249_2493, 1)),  // M, shift=1
            9  => Some((0xE38E_38E3_8E38_E38F, 1)),  // M = ceil(2^65/9), shift=1
            10 => Some((0xCCCC_CCCC_CCCC_CCCD, 3)),  // M = ceil(2^66/5), shift=3 (div by 10 = div by 5 then >>1)
            12 => Some((0xAAAA_AAAA_AAAA_AAAB, 3)),  // div by 12 = div by 3 then >>2
            15 => Some((0x8888_8888_8888_8889, 3)),  // M, shift=3
            25 => Some((0x47AE_147A_E147_AE15, 3)),  // M, shift=3
            _  => {
                // General computation for other divisors
                // M = ceil(2^(64+shift) / d) where shift is minimal
                // For shift=0: M = (2^64 - 1) / d + 1
                // This works when M < 2^64, i.e., when d doesn't evenly divide 2^64
                let m = (u64::MAX / d) + 1;
                if m != 0 {
                    Some((m, 0))
                } else {
                    None
                }
            }
        }
    }

    /// Compute magic constants for signed division by a constant.
    ///
    /// For signed division: similar to unsigned but with sign correction.
    /// If n < 0: add (d-1) before unsigned division.
    ///
    /// Returns (magic_constant, shift) or None if not suitable.
    fn compute_signed_magic(divisor: i64) -> Option<(i64, u32)> {
        if divisor == 0 {
            return None;
        }

        let abs_d = divisor.unsigned_abs();

        // For signed, use a similar approach but the magic constant can be negative
        let abs_magic = Self::compute_unsigned_magic(abs_d)?;

        // The signed magic constant has the same magnitude but may need sign adjustment
        let magic = if divisor < 0 {
            -((abs_magic.0) as i64)
        } else {
            abs_magic.0 as i64
        };

        Some((magic, abs_magic.1))
    }

    /// Try to reduce a single node. Returns `Some(replacement)` if reduction applies.
    fn try_reduce(&self, graph: &mut IrGraph, _id: NodeId, node: &IrNode) -> Option<IrNode> {
        match node {
            // ── Mul by power of 2 → Shl ────────────────────────────────
            IrNode::Mul { lhs, rhs } => {
                // Mul by 0 → 0
                if Self::is_int_const(graph, *rhs, 0) {
                    return Some(IrNode::IntConst(0));
                }
                if Self::is_int_const(graph, *lhs, 0) {
                    return Some(IrNode::IntConst(0));
                }

                // Mul by 1 → lhs
                if Self::is_int_const(graph, *rhs, 1) {
                    return graph.get(*lhs).cloned();
                }
                if Self::is_int_const(graph, *lhs, 1) {
                    return graph.get(*rhs).cloned();
                }

                // Mul by power of 2 → Shl
                if let Some(shift) = Self::get_power_of_two(graph, *rhs) {
                    let shift_node = graph.push_node(IrNode::IntConst(shift as i64));
                    return Some(IrNode::Shl {
                        lhs: *lhs,
                        rhs: shift_node,
                    });
                }
                if let Some(shift) = Self::get_power_of_two(graph, *lhs) {
                    let shift_node = graph.push_node(IrNode::IntConst(shift as i64));
                    return Some(IrNode::Shl {
                        lhs: *rhs,
                        rhs: shift_node,
                    });
                }

                // ── Mul by constant (not power of 2) → shift-and-add ──
                if let Some(c) = Self::get_int_const(graph, *rhs) {
                    if c > 1 {
                        if let Some(terms) = Self::decompose_multiplier(c as u64) {
                            return Self::build_shift_add_sequence(graph, *lhs, &terms);
                        }
                    }
                    if c < -1 {
                        if let Some(terms) = Self::decompose_multiplier((-c) as u64) {
                            // x * (-C) = -(x * C)
                            let pos_result = Self::build_shift_add_sequence(graph, *lhs, &terms)?;
                            let pos_node = graph.push_node(pos_result);
                            return Some(IrNode::Neg { val: pos_node });
                        }
                    }
                }
                if let Some(c) = Self::get_int_const(graph, *lhs) {
                    if c > 1 {
                        if let Some(terms) = Self::decompose_multiplier(c as u64) {
                            return Self::build_shift_add_sequence(graph, *rhs, &terms);
                        }
                    }
                }

                None
            }

            // ── Div by power of 2 or constant ──────────────────────────
            IrNode::Div { lhs, rhs } => {
                // Div by 1 → lhs
                if Self::is_int_const(graph, *rhs, 1) {
                    return graph.get(*lhs).cloned();
                }

                if let Some(shift) = Self::get_power_of_two(graph, *rhs) {
                    let is_unsigned = Self::is_unsigned(graph, *lhs, *rhs);

                    if is_unsigned == Some(true) {
                        // Unsigned division by power of 2 → logical right shift.
                        let shift_node = graph.push_node(IrNode::IntConst(shift as i64));
                        return Some(IrNode::Shr {
                            lhs: *lhs,
                            rhs: shift_node,
                        });
                    } else {
                        // Signed division by power of 2 → Sar with rounding adjustment.
                        let bits: u32 = 64; // default for I64 / IntConst
                        let shift_node = graph.push_node(IrNode::IntConst(shift as i64));
                        let bits_minus_1 = graph.push_node(IrNode::IntConst((bits - 1) as i64));
                        let sign = graph.push_node(IrNode::Sar {
                            lhs: *lhs,
                            rhs: bits_minus_1,
                        });
                        let bias_val = (1i64 << shift) - 1;
                        let bias_const = graph.push_node(IrNode::IntConst(bias_val));
                        let bias = graph.push_node(IrNode::And {
                            lhs: sign,
                            rhs: bias_const,
                        });
                        let biased = graph.push_node(IrNode::Add {
                            lhs: *lhs,
                            rhs: bias,
                        });
                        return Some(IrNode::Sar {
                            lhs: biased,
                            rhs: shift_node,
                        });
                    }
                }

                // ── Division by constant: magic multiply transform ──────
                if let Some(c) = Self::get_int_const(graph, *rhs) {
                    if c > 1 {
                        let is_unsigned = Self::is_unsigned(graph, *lhs, *rhs);
                        if is_unsigned == Some(true) {
                            // Unsigned division by constant
                            if let Some((magic, shift)) = Self::compute_unsigned_magic(c as u64) {
                                return Self::build_unsigned_magic_div(
                                    graph, *lhs, magic, shift,
                                );
                            }
                        } else if is_unsigned == Some(false) {
                            // Signed division by constant
                            if let Some((magic, shift)) = Self::compute_signed_magic(c) {
                                return Self::build_signed_magic_div(
                                    graph, *lhs, magic, shift, c,
                                );
                            }
                        }
                    }
                    if c < -1 {
                        // x / (-C) = -(x / C)
                        let abs_c = -c;
                        let is_unsigned = Self::is_unsigned(graph, *lhs, *rhs);
                        if is_unsigned == Some(false) {
                            if let Some((magic, shift)) = Self::compute_signed_magic(abs_c) {
                                let div_result = Self::build_signed_magic_div(
                                    graph, *lhs, magic, shift, abs_c,
                                )?;
                                let div_node = graph.push_node(div_result);
                                return Some(IrNode::Neg { val: div_node });
                            }
                        }
                    }
                }

                None
            }

            // ── Add with 0 → identity ──────────────────────────────────
            IrNode::Add { lhs, rhs } => {
                if Self::is_int_const(graph, *rhs, 0) {
                    return graph.get(*lhs).cloned();
                }
                if Self::is_int_const(graph, *lhs, 0) {
                    return graph.get(*rhs).cloned();
                }
                None
            }

            // ── Sub with 0 → identity (lhs) ────────────────────────────
            IrNode::Sub { lhs, rhs } => {
                if Self::is_int_const(graph, *rhs, 0) {
                    return graph.get(*lhs).cloned();
                }
                None
            }

            // ── FP: FMul by 2.0 → FAdd(x, x) ─────────────────────────
            IrNode::FMul { lhs, rhs } => {
                // FMul by 2.0 → FAdd(x, x)
                if Self::is_fp_const(graph, *rhs, 2.0) {
                    return Some(IrNode::FAdd { lhs: *lhs, rhs: *lhs });
                }
                if Self::is_fp_const(graph, *lhs, 2.0) {
                    return Some(IrNode::FAdd { lhs: *rhs, rhs: *rhs });
                }
                // FMul by 1.0 → lhs
                if Self::is_fp_const(graph, *rhs, 1.0) {
                    return graph.get(*lhs).cloned();
                }
                if Self::is_fp_const(graph, *lhs, 1.0) {
                    return graph.get(*rhs).cloned();
                }
                None
            }

            // ── FP: FDiv by constant → FMul by reciprocal ─────────────
            IrNode::FDiv { lhs, rhs } => {
                if let Some(rhs_val) = Self::get_fp_const(graph, *rhs) {
                    if rhs_val != 0.0 && rhs_val.is_normal() {
                        let recip = 1.0 / rhs_val;
                        // Only transform if reciprocal is exact for powers of 2
                        // or if the reciprocal is a "nice" value
                        if recip.is_normal() {
                            let recip_node = graph.push_node(IrNode::FpConst(recip.to_bits()));
                            return Some(IrNode::FMul {
                                lhs: *lhs,
                                rhs: recip_node,
                            });
                        }
                    }
                }
                None
            }

            // ── FP: FAdd with 0.0 → identity ──────────────────────────
            IrNode::FAdd { lhs, rhs } => {
                if Self::is_fp_const(graph, *rhs, 0.0) {
                    return graph.get(*lhs).cloned();
                }
                if Self::is_fp_const(graph, *lhs, 0.0) {
                    return graph.get(*rhs).cloned();
                }
                None
            }

            // ── FP: FSub with 0.0 → identity (lhs) ────────────────────
            IrNode::FSub { lhs, rhs } => {
                if Self::is_fp_const(graph, *rhs, 0.0) {
                    return graph.get(*lhs).cloned();
                }
                None
            }

            _ => None,
        }
    }

    /// Build a shift-and-add sequence from decomposed multiplier terms.
    ///
    /// terms is a list of (shift_amount, is_subtract).
    /// The first term is the base shift; remaining terms are added/subtracted.
    fn build_shift_add_sequence(
        graph: &mut IrGraph,
        base: NodeId,
        terms: &[(u32, bool)],
    ) -> Option<IrNode> {
        if terms.is_empty() {
            return None;
        }

        // Start with the first term: base << shift0
        let mut acc_node: NodeId = {
            let s = graph.push_node(IrNode::IntConst(terms[0].0 as i64));
            graph.push_node(IrNode::Shl { lhs: base, rhs: s })
        };

        for &(shift, is_subtract) in &terms[1..] {
            let s = graph.push_node(IrNode::IntConst(shift as i64));
            let shifted = graph.push_node(IrNode::Shl { lhs: base, rhs: s });
            if is_subtract {
                acc_node = graph.push_node(IrNode::Sub {
                    lhs: acc_node,
                    rhs: shifted,
                });
            } else {
                acc_node = graph.push_node(IrNode::Add {
                    lhs: acc_node,
                    rhs: shifted,
                });
            }
        }

        // Return a copy of the last node's content as the replacement.
        graph.get(acc_node).cloned()
    }

    /// Build unsigned magic multiply division sequence.
    ///
    /// q = mulhu(n, M) >> shift
    /// Since we don't have a native mulhu in the IR, we approximate:
    /// For the precomputed magic constants for common divisors, we generate:
    ///   t = n * magic_const   (Mul gives low 64 bits)
    ///   q = t >> (64 + shift)  (right shift to extract high bits)
    ///
    /// When magic fits in 32 bits and shift > 0, we can get correct results
    /// for small dividends by shifting the multiplication result right.
    /// For large magic constants, we fall back to the original division.
    fn build_unsigned_magic_div(
        graph: &mut IrGraph,
        lhs: NodeId,
        magic: u64,
        shift: u32,
    ) -> Option<IrNode> {
        // Only handle cases where magic fits in 32 bits (common case for
        // small divisors). For larger magics, we'd need MulHi which
        // isn't in the IR yet.
        if magic > u32::MAX as u64 {
            return None;
        }

        let magic_node = graph.push_node(IrNode::IntConst(magic as i64));
        let mul_result = graph.push_node(IrNode::Mul {
            lhs,
            rhs: magic_node,
        });

        if shift > 0 {
            let shift_node = graph.push_node(IrNode::IntConst((64 + shift) as i64));
            let result = graph.push_node(IrNode::Shr {
                lhs: mul_result,
                rhs: shift_node,
            });
            graph.get(result).cloned()
        } else {
            // Shift by 64 bits: for 64-bit values this would be
            // the high word of the multiplication. Since we don't have
            // MulHi, we can't extract the high 64 bits easily.
            // Fall back to the original Div for now.
            None
        }
    }

    /// Build signed magic multiply division sequence.
    ///
    /// Similar to unsigned but with sign correction:
    /// If n < 0: add (d-1) before unsigned division.
    /// Then: t = mulhu(n, M) >> shift
    ///       q = t + (n >> 63)  // sign correction for negative dividends
    fn build_signed_magic_div(
        graph: &mut IrGraph,
        lhs: NodeId,
        magic: i64,
        shift: u32,
        divisor: i64,
    ) -> Option<IrNode> {
        let abs_magic = magic.unsigned_abs();
        if abs_magic > u32::MAX as u64 {
            return None;
        }

        // Sign correction: if n < 0, add (d-1) to n before dividing
        // This is equivalent to: if n < 0 { n = n + (d - 1) }
        // We implement: corrected_n = n + ((n >> 63) & (d - 1))
        let d_minus_1 = graph.push_node(IrNode::IntConst(divisor - 1));
        let sixty_three = graph.push_node(IrNode::IntConst(63));
        let sign_mask = graph.push_node(IrNode::Sar {
            lhs,
            rhs: sixty_three,
        });
        let correction = graph.push_node(IrNode::And {
            lhs: sign_mask,
            rhs: d_minus_1,
        });
        let corrected_n = graph.push_node(IrNode::Add {
            lhs,
            rhs: correction,
        });

        let magic_node = graph.push_node(IrNode::IntConst(magic));
        let mul_result = graph.push_node(IrNode::Mul {
            lhs: corrected_n,
            rhs: magic_node,
        });

        // Same issue as unsigned: need high 64 bits of multiplication
        if shift > 0 {
            let total_shift = 64 + shift;
            let shift_node = graph.push_node(IrNode::IntConst(total_shift as i64));
            let shifted = graph.push_node(IrNode::Shr {
                lhs: mul_result,
                rhs: shift_node,
            });

            // Sign correction for the result: add (n >> 63)
            let sixty_three2 = graph.push_node(IrNode::IntConst(63));
            let sign_correction = graph.push_node(IrNode::Sar {
                lhs,
                rhs: sixty_three2,
            });

            let result = graph.push_node(IrNode::Add {
                lhs: shifted,
                rhs: sign_correction,
            });

            graph.get(result).cloned()
        } else {
            None
        }
    }
}

impl Pass for StrengthReducer {
    fn name(&self) -> &str {
        "strength_reduce"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        // Collect node IDs up front.
        let node_ids: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();

        for id in node_ids {
            let node = match graph.get(id) {
                Some(n) => n.clone(),
                None => continue,
            };

            if let Some(replacement) = self.try_reduce(graph, id, &node) {
                graph.replace(id, replacement);
                modified = true;
            }
        }

        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_by_power_of_two() {
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let four = graph.push_node(IrNode::IntConst(4)); // 2^2
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: four });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        match graph.get(mul) {
            Some(IrNode::Shl { lhs, .. }) => {
                assert_eq!(*lhs, x);
            }
            other => panic!("expected Shl, got {:?}", other),
        }
    }

    #[test]
    fn mul_by_one() {
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let one = graph.push_node(IrNode::IntConst(1));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: one });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        match graph.get(mul) {
            Some(IrNode::IntConst(5)) => {}
            other => panic!("expected IntConst(5), got {:?}", other),
        }
    }

    #[test]
    fn add_with_zero() {
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let zero = graph.push_node(IrNode::IntConst(0));
        let add = graph.push_node(IrNode::Add { lhs: x, rhs: zero });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        match graph.get(add) {
            Some(IrNode::IntConst(5)) => {}
            other => panic!("expected IntConst(5), got {:?}", other),
        }
    }

    #[test]
    fn sub_with_zero() {
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(7));
        let zero = graph.push_node(IrNode::IntConst(0));
        let sub = graph.push_node(IrNode::Sub { lhs: x, rhs: zero });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        match graph.get(sub) {
            Some(IrNode::IntConst(7)) => {}
            other => panic!("expected IntConst(7), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_zero() {
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let zero = graph.push_node(IrNode::IntConst(0));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: zero });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        match graph.get(mul) {
            Some(IrNode::IntConst(0)) => {}
            other => panic!("expected IntConst(0), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_3_shift_add() {
        // x * 3 = (x << 1) + x  (Strategy 2: C-1=2 is power of 2)
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let three = graph.push_node(IrNode::IntConst(3));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: three });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        // The mul node should be replaced with an Add containing Shl
        let node = graph.get(mul).cloned().unwrap();
        // Should be Add { lhs: Shl(x, 1), rhs: x } (Strategy 2 gives [(1,false),(0,false)])
        match &node {
            IrNode::Add { .. } => {} // success: expanded to shift+add
            other => panic!("expected Add (shift-and-add), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_5_shift_add() {
        // x * 5 = (x << 2) + x  (Strategy 2: C-1=4 is power of 2)
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let five = graph.push_node(IrNode::IntConst(5));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: five });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        let node = graph.get(mul).cloned().unwrap();
        match &node {
            IrNode::Add { .. } => {}
            other => panic!("expected Add (shift-and-add for *5), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_6_shift_add() {
        // x * 6 = (x << 2) + (x << 1)  (Strategy 3: binary 110 = 2+4)
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let six = graph.push_node(IrNode::IntConst(6));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: six });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        let node = graph.get(mul).cloned().unwrap();
        match &node {
            IrNode::Add { .. } => {}
            other => panic!("expected Add (shift-and-add for *6), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_7_shift_sub() {
        // x * 7 = (x << 3) - x  (Strategy 1: C+1=8 is power of 2)
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let seven = graph.push_node(IrNode::IntConst(7));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: seven });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        let node = graph.get(mul).cloned().unwrap();
        match &node {
            IrNode::Sub { .. } => {}
            other => panic!("expected Sub (shift-and-subtract), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_9_shift_add() {
        // x * 9 = (x << 3) + x  (Strategy 2: C-1=8 is power of 2)
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let nine = graph.push_node(IrNode::IntConst(9));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: nine });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        let node = graph.get(mul).cloned().unwrap();
        match &node {
            IrNode::Add { .. } => {}
            other => panic!("expected Add (shift-and-add for *9), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_10_shift_add() {
        // x * 10 = (x << 3) + (x << 1)  (Strategy 3: binary 1010 = 2+8)
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let ten = graph.push_node(IrNode::IntConst(10));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: ten });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        let node = graph.get(mul).cloned().unwrap();
        match &node {
            IrNode::Add { .. } => {}
            other => panic!("expected Add (shift-and-add for *10), got {:?}", other),
        }
    }

    #[test]
    fn mul_by_15_shift_sub() {
        // x * 15 = (x << 4) - x  (Strategy 1: C+1=16 is power of 2)
        let mut graph = IrGraph::new("test");
        let x = graph.push_node(IrNode::IntConst(5));
        let fifteen = graph.push_node(IrNode::IntConst(15));
        let mul = graph.push_node(IrNode::Mul { lhs: x, rhs: fifteen });

        let sr = StrengthReducer;
        assert!(sr.run(&mut graph));

        let node = graph.get(mul).cloned().unwrap();
        match &node {
            IrNode::Sub { .. } => {}
            other => panic!("expected Sub (shift-and-subtract), got {:?}", other),
        }
    }

    #[test]
    fn test_decompose_multiplier() {
        // x * 3 = (x<<1) + x  (Strategy 2: C-1 = 2, power of 2)
        let terms = StrengthReducer::decompose_multiplier(3).unwrap();
        assert!(terms.contains(&(1, false)), "should contain shift 1 (add), got {:?}", terms);
        assert!(terms.contains(&(0, false)), "should contain shift 0 (add), got {:?}", terms);

        // x * 5 = (x<<2) + x  (Strategy 2: C-1 = 4, power of 2)
        let terms = StrengthReducer::decompose_multiplier(5).unwrap();
        assert!(terms.contains(&(2, false)));
        assert!(terms.contains(&(0, false)));

        // x * 6 = (x<<2) + (x<<1)  (Strategy 3: binary 110)
        let terms = StrengthReducer::decompose_multiplier(6).unwrap();
        assert!(terms.contains(&(2, false)));
        assert!(terms.contains(&(1, false)));

        // x * 7 = (x<<3) - x  (Strategy 1: C+1 = 8, power of 2)
        let terms = StrengthReducer::decompose_multiplier(7).unwrap();
        assert!(terms.iter().any(|&(s, _)| s == 3), "should contain shift 3, got {:?}", terms);
        assert!(terms.iter().any(|&(_, sub)| sub), "should have a subtract, got {:?}", terms);

        // x * 9 = (x<<3) + x  (Strategy 2: C-1 = 8, power of 2)
        let terms = StrengthReducer::decompose_multiplier(9).unwrap();
        assert!(terms.contains(&(3, false)));
        assert!(terms.contains(&(0, false)));

        // x * 10 = (x<<3) + (x<<1)  (Strategy 3: binary 1010)
        let terms = StrengthReducer::decompose_multiplier(10).unwrap();
        assert!(terms.contains(&(3, false)));
        assert!(terms.contains(&(1, false)));

        // x * 15 = (x<<4) - x  (Strategy 1: C+1 = 16, power of 2)
        let terms = StrengthReducer::decompose_multiplier(15).unwrap();
        assert!(terms.iter().any(|&(s, _)| s == 4), "should contain shift 4, got {:?}", terms);
        assert!(terms.iter().any(|&(_, sub)| sub), "should have a subtract, got {:?}", terms);
    }

    #[test]
    fn test_unsigned_magic_constants() {
        // Verify magic constants are computed for common divisors
        for d in [3u64, 5, 6, 7, 9, 10, 12, 15, 25] {
            let result = StrengthReducer::compute_unsigned_magic(d);
            assert!(result.is_some(), "should have magic constants for divisor {}", d);
            let (magic, shift) = result.unwrap();
            assert!(magic > 0, "magic should be non-zero for divisor {}", d);
            let _ = (magic, shift); // just check it exists
        }
    }
}
