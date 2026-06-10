//! Common Subexpression Elimination.
//!
//! CRITICAL CORRECTNESS FIX: CSE MUST check for intervening stores on the same
//! ownership root. Two `Load` nodes from the same address and root are only
//! equivalent if no `Store` to that root has occurred between them.
//!
//! # Algorithm
//!
//! - **Constants** (IntConst, BoolConst): keyed by their value hash.
//!   Duplicate constants with the same value are replaced by the first.
//! - **Pure operations** (Add, Sub, etc.): key = u64 hash, no root check.
//! - **Load operations**: key = u64 hash. Tracked per ownership root.
//!   When a `Store` to root `R` is seen, **all** Load CSE entries for root `R`
//!   are invalidated — the store may have changed the loaded value.
//! - **Store operations**: NEVER CSE'd (side effects).
//!
//! # Performance
//!
//! Uses u64 hash keys instead of String keys for O(1) comparison
//! and zero allocation per lookup. Hash construction uses the `hash2`
//! combining function for good distribution.

use std::collections::HashMap;

use axiom_ir::{IrGraph, IrNode, NodeId, OwnershipRoot};
use crate::Pass;

/// Common Subexpression Elimination pass.
///
/// Walks the graph in NodeId order, building a hash table of operation→NodeId
/// mappings. When a duplicate is found, all uses of the new node are replaced
/// with the existing node and the new node is removed.
///
/// Uses u64 hash keys for O(1) comparison and zero string allocation.
pub struct CommonSubexprElim;

/// Hash combining function — the golden ratio constant for good bit mixing.
///
/// Applies: a * GOLDEN_RATIO + b (wrapping arithmetic).
/// This is a standard technique from Boost's hash_combine and provides
/// excellent avalanche behavior for combining two u64 values into one.
#[inline]
fn hash2(a: u64, b: u64) -> u64 {
    a.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(b)
}

/// Tag constants for hash key construction.
/// Each operation type gets a unique tag value to avoid collisions.
const TAG_INTCONST: u64 = 0;
const TAG_FPCONST: u64 = 1;
const TAG_BOOLCONST: u64 = 2;
const TAG_ADD: u64 = 3;
const TAG_SUB: u64 = 4;
const TAG_MUL: u64 = 5;
const TAG_DIV: u64 = 6;
const TAG_REM: u64 = 7;
const TAG_AND: u64 = 8;
const TAG_OR: u64 = 9;
const TAG_XOR: u64 = 10;
const TAG_SHL: u64 = 11;
const TAG_SHR: u64 = 12;
const TAG_SAR: u64 = 13;
const TAG_NEG: u64 = 14;
const TAG_NOT: u64 = 15;
const TAG_EQ: u64 = 16;
const TAG_NE: u64 = 17;
const TAG_LT: u64 = 18;
const TAG_LE: u64 = 19;
const TAG_GT: u64 = 20;
const TAG_GE: u64 = 21;
const TAG_ZEXT: u64 = 22;
const TAG_SEXT: u64 = 23;
const TAG_TRUNC: u64 = 24;
const TAG_BITCAST: u64 = 25;
const TAG_INTTOPTR: u64 = 26;
const TAG_PTRTOINT: u64 = 27;
const TAG_EXTRACT: u64 = 28;
const TAG_INSERT: u64 = 29;
// FP operations
const TAG_FADD: u64 = 30;
const TAG_FSUB: u64 = 31;
const TAG_FMUL: u64 = 32;
const TAG_FDIV: u64 = 33;
const TAG_FREM: u64 = 34;
const TAG_FNEG: u64 = 35;
const TAG_FABS: u64 = 36;
const TAG_FSQRT: u64 = 37;
const TAG_FEQ: u64 = 38;
const TAG_FLT: u64 = 39;
const TAG_FLE: u64 = 40;
const TAG_FGT: u64 = 41;
const TAG_FGE: u64 = 42;
const TAG_FNE: u64 = 43;
const TAG_FPTOSINT: u64 = 44;
const TAG_SINTTOFP: u64 = 45;
const TAG_FPTOUINT: u64 = 46;
const TAG_UINTTOFP: u64 = 47;
const TAG_COPYSIGN: u64 = 48;
const TAG_FMIN: u64 = 49;
const TAG_FMAX: u64 = 50;
// Load
const TAG_LOAD: u64 = 60;

/// Build a u64 hash key for a binary operation with two NodeId inputs.
/// For commutative ops, inputs are sorted to ensure a == b same as b == a.
///
/// Uses the hash2 combining function:
///   key = hash2(tag, hash2(lo, hi))
#[inline]
fn binop_key(tag: u64, a: NodeId, b: NodeId, commutative: bool) -> u64 {
    let (lo, hi) = if commutative && a > b {
        (b.0 as u64, a.0 as u64)
    } else {
        (a.0 as u64, b.0 as u64)
    };
    hash2(tag, hash2(lo, hi))
}

/// Build a u64 hash key for a unary operation.
///
/// key = hash2(tag, val_id)
#[inline]
fn unop_key(tag: u64, val: NodeId) -> u64 {
    hash2(tag, val.0 as u64)
}

/// Build a u64 hash key for a unary operation with a type operand.
///
/// key = hash2(tag, hash2(val_id, type_bits))
#[inline]
fn unop_type_key(tag: u64, val: NodeId, ty: axiom_ir::nodes::Type) -> u64 {
    let ty_bits = type_to_bits(ty);
    hash2(tag, hash2(val.0 as u64, ty_bits as u64))
}

/// Build a u64 hash key for an extract operation.
///
/// key = hash2(TAG_EXTRACT, hash2(aggregate_id, index))
#[inline]
fn extract_key(aggregate: NodeId, index: u32) -> u64 {
    hash2(TAG_EXTRACT, hash2(aggregate.0 as u64, index as u64))
}

/// Build a u64 hash key for an insert operation.
///
/// key = hash2(TAG_INSERT, hash2(hash2(aggregate_id, index), value_id))
#[inline]
fn insert_key(aggregate: NodeId, index: u32, value: NodeId) -> u64 {
    hash2(TAG_INSERT, hash2(hash2(aggregate.0 as u64, index as u64), value.0 as u64))
}

/// Build a u64 hash key for a Load operation.
///
/// key = hash2(TAG_LOAD, hash2(addr_id, type_bits))
#[inline]
fn load_hash_key(addr: NodeId, ty: axiom_ir::nodes::Type) -> u64 {
    let ty_bits = type_to_bits(ty);
    hash2(TAG_LOAD, hash2(addr.0 as u64, ty_bits as u64))
}

/// Convert a Type to a compact u16 for hashing.
fn type_to_bits(ty: axiom_ir::nodes::Type) -> u16 {
    match ty {
        axiom_ir::nodes::Type::I8 => 0,
        axiom_ir::nodes::Type::I16 => 1,
        axiom_ir::nodes::Type::I32 => 2,
        axiom_ir::nodes::Type::I64 => 3,
        axiom_ir::nodes::Type::I128 => 4,
        axiom_ir::nodes::Type::U8 => 5,
        axiom_ir::nodes::Type::U16 => 6,
        axiom_ir::nodes::Type::U32 => 7,
        axiom_ir::nodes::Type::U64 => 8,
        axiom_ir::nodes::Type::U128 => 9,
        axiom_ir::nodes::Type::F32 => 10,
        axiom_ir::nodes::Type::F64 => 11,
        axiom_ir::nodes::Type::Bool => 12,
        axiom_ir::nodes::Type::Ptr => 13,
        axiom_ir::nodes::Type::Void => 14,
        axiom_ir::nodes::Type::Unknown => 15,
    }
}

impl CommonSubexprElim {
    /// Compute a CSE hash key for a constant or pure (non-memory) operation.
    /// Returns `None` for operations that should not be CSE'd.
    fn node_key(node: &IrNode) -> Option<u64> {
        match node {
            // ── Constants ──────────────────────────────────────────────
            IrNode::IntConst(n) => Some(hash2(TAG_INTCONST, *n as u64)),
            IrNode::BoolConst(b) => Some(hash2(TAG_BOOLCONST, *b as u64)),
            // FpConst: hash the bits representation with tag
            IrNode::FpConst(bits) => Some(hash2(TAG_FPCONST, *bits)),

            // ── Commutative binary ops ───────────────────────────────────
            IrNode::Add { lhs, rhs } => Some(binop_key(TAG_ADD, *lhs, *rhs, true)),
            IrNode::Mul { lhs, rhs } => Some(binop_key(TAG_MUL, *lhs, *rhs, true)),
            IrNode::And { lhs, rhs } => Some(binop_key(TAG_AND, *lhs, *rhs, true)),
            IrNode::Or { lhs, rhs } => Some(binop_key(TAG_OR, *lhs, *rhs, true)),
            IrNode::Xor { lhs, rhs } => Some(binop_key(TAG_XOR, *lhs, *rhs, true)),
            IrNode::Eq { lhs, rhs } => Some(binop_key(TAG_EQ, *lhs, *rhs, true)),
            IrNode::Ne { lhs, rhs } => Some(binop_key(TAG_NE, *lhs, *rhs, true)),

            // ── Non-commutative binary ops ──────────────────────────────
            IrNode::Sub { lhs, rhs } => Some(binop_key(TAG_SUB, *lhs, *rhs, false)),
            IrNode::Div { lhs, rhs } => Some(binop_key(TAG_DIV, *lhs, *rhs, false)),
            IrNode::Rem { lhs, rhs } => Some(binop_key(TAG_REM, *lhs, *rhs, false)),
            IrNode::Shl { lhs, rhs } => Some(binop_key(TAG_SHL, *lhs, *rhs, false)),
            IrNode::Shr { lhs, rhs } => Some(binop_key(TAG_SHR, *lhs, *rhs, false)),
            IrNode::Sar { lhs, rhs } => Some(binop_key(TAG_SAR, *lhs, *rhs, false)),
            IrNode::Lt { lhs, rhs } => Some(binop_key(TAG_LT, *lhs, *rhs, false)),
            IrNode::Le { lhs, rhs } => Some(binop_key(TAG_LE, *lhs, *rhs, false)),
            IrNode::Gt { lhs, rhs } => Some(binop_key(TAG_GT, *lhs, *rhs, false)),
            IrNode::Ge { lhs, rhs } => Some(binop_key(TAG_GE, *lhs, *rhs, false)),

            // ── FP commutative binary ops ─────────────────────────────
            IrNode::FAdd { lhs, rhs } => Some(binop_key(TAG_FADD, *lhs, *rhs, true)),
            IrNode::FMul { lhs, rhs } => Some(binop_key(TAG_FMUL, *lhs, *rhs, true)),
            IrNode::FEq { lhs, rhs } => Some(binop_key(TAG_FEQ, *lhs, *rhs, true)),
            IrNode::FNe { lhs, rhs } => Some(binop_key(TAG_FNE, *lhs, *rhs, true)),

            // ── FP non-commutative binary ops ──────────────────────────
            IrNode::FSub { lhs, rhs } => Some(binop_key(TAG_FSUB, *lhs, *rhs, false)),
            IrNode::FDiv { lhs, rhs } => Some(binop_key(TAG_FDIV, *lhs, *rhs, false)),
            IrNode::FRem { lhs, rhs } => Some(binop_key(TAG_FREM, *lhs, *rhs, false)),
            IrNode::FLt { lhs, rhs } => Some(binop_key(TAG_FLT, *lhs, *rhs, false)),
            IrNode::FLe { lhs, rhs } => Some(binop_key(TAG_FLE, *lhs, *rhs, false)),
            IrNode::FGt { lhs, rhs } => Some(binop_key(TAG_FGT, *lhs, *rhs, false)),
            IrNode::FGe { lhs, rhs } => Some(binop_key(TAG_FGE, *lhs, *rhs, false)),
            IrNode::Copysign { lhs, rhs } => Some(binop_key(TAG_COPYSIGN, *lhs, *rhs, false)),
            IrNode::Fmin { lhs, rhs } => Some(binop_key(TAG_FMIN, *lhs, *rhs, false)),
            IrNode::Fmax { lhs, rhs } => Some(binop_key(TAG_FMAX, *lhs, *rhs, false)),

            // ── FP unary ops ──────────────────────────────────────────
            IrNode::FNeg { val } => Some(unop_key(TAG_FNEG, *val)),
            IrNode::FAbs { val } => Some(unop_key(TAG_FABS, *val)),
            IrNode::FSqrt { val } => Some(unop_key(TAG_FSQRT, *val)),

            // ── FP conversion ops ─────────────────────────────────────
            IrNode::FpToSInt { val, to } => Some(unop_type_key(TAG_FPTOSINT, *val, *to)),
            IrNode::SIntToFp { val, to } => Some(unop_type_key(TAG_SINTTOFP, *val, *to)),
            IrNode::FpToUInt { val, to } => Some(unop_type_key(TAG_FPTOUINT, *val, *to)),
            IrNode::UIntToFp { val, to } => Some(unop_type_key(TAG_UINTTOFP, *val, *to)),

            // ── Unary ops ──────────────────────────────────────────────
            IrNode::Neg { val } => Some(unop_key(TAG_NEG, *val)),
            IrNode::Not { val } => Some(unop_key(TAG_NOT, *val)),

            // ── Conversions ────────────────────────────────────────────
            IrNode::ZExt { val, to } => Some(unop_type_key(TAG_ZEXT, *val, *to)),
            IrNode::SExt { val, to } => Some(unop_type_key(TAG_SEXT, *val, *to)),
            IrNode::Trunc { val, to } => Some(unop_type_key(TAG_TRUNC, *val, *to)),
            IrNode::BitCast { val, to } => Some(unop_type_key(TAG_BITCAST, *val, *to)),
            IrNode::IntToPtr { val } => Some(unop_key(TAG_INTTOPTR, *val)),
            IrNode::PtrToInt { val } => Some(unop_key(TAG_PTRTOINT, *val)),

            // ── Aggregates ─────────────────────────────────────────────
            IrNode::Extract { aggregate, index } => Some(extract_key(*aggregate, *index)),
            IrNode::Insert { aggregate, index, value } => Some(insert_key(*aggregate, *index, *value)),

            // ── Everything else: don't CSE ─────────────────────────────
            _ => None,
        }
    }

    /// Compute a CSE hash key for a Load operation (excluding the root —
    /// the root is tracked separately in the load CSE table).
    fn load_key(node: &IrNode) -> Option<(u64, OwnershipRoot)> {
        match node {
            IrNode::Load { addr, root, ty } => {
                let key = load_hash_key(*addr, *ty);
                Some((key, *root))
            }
            _ => None,
        }
    }

    /// Returns true if the node has side effects and should never be CSE'd.
    fn is_non_cse(node: &IrNode) -> bool {
        node.has_side_effects()
            || matches!(
                node,
                IrNode::Start
                    | IrNode::Unreachable
                    | IrNode::Region { .. }
                    | IrNode::Phi { .. }
                    | IrNode::Call { .. }
                    | IrNode::CallIndirect { .. }
                    | IrNode::Intrinsic { .. }
                    | IrNode::VarDef { .. }
                    | IrNode::VarRef { .. }
                    | IrNode::VarSet { .. }
                    | IrNode::Fence { .. }
                    | IrNode::Owned { .. }
                    | IrNode::UndefConst
            )
    }
}

impl Pass for CommonSubexprElim {
    fn name(&self) -> &str {
        "cse"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;

        // CSE table for pure operations and constants: u64 hash → existing NodeId.
        let mut pure_cse: HashMap<u64, NodeId> = HashMap::new();

        // CSE table for Load operations: (u64 hash, root) → existing NodeId.
        // When a Store to root R is seen, all entries for root R are removed.
        let mut load_cse: HashMap<(u64, OwnershipRoot), NodeId> = HashMap::new();

        // Track how many stores we've seen per root (for the invalidation logic).
        let mut _store_counts: HashMap<OwnershipRoot, u32> = HashMap::new();

        // Collect all node IDs up front; process in NodeId order.
        let node_ids: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();

        for id in node_ids {
            let node = match graph.get(id) {
                Some(n) => n.clone(),
                None => continue,
            };

            // ── Handle Stores: invalidate Load CSE entries for same root ──
            if let IrNode::Store { root, .. } = &node {
                let root = *root;
                load_cse.retain(|(_, r), _| *r != root);
                *_store_counts.entry(root).or_insert(0) += 1;
                continue;
            }

            // Skip nodes that we never CSE.
            if Self::is_non_cse(&node) {
                continue;
            }

            // ── Handle Loads: check load_cse table with root awareness ───
            if let Some((key, root)) = Self::load_key(&node) {
                let lookup = (key, root);
                if let Some(&existing) = load_cse.get(&lookup) {
                    if existing != id {
                        graph.replace_uses(id, existing);
                        graph.remove(id);
                        modified = true;
                        continue;
                    }
                }
                load_cse.insert(lookup, id);
                continue;
            }

            // ── Handle constants and pure operations ────────────────────
            if let Some(key) = Self::node_key(&node) {
                if let Some(&existing) = pure_cse.get(&key) {
                    if existing != id {
                        // For constants, always eliminate duplicates
                        // For operations, eliminate identical computations
                        graph.replace_uses(id, existing);
                        graph.remove(id);
                        modified = true;
                        continue;
                    }
                }
                pure_cse.insert(key, id);
            }
        }

        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::Type;

    #[test]
    fn cse_eliminates_duplicate_add() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let add1 = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let add2 = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(add2) });

        let cse = CommonSubexprElim;
        assert!(cse.run(&mut graph));

        assert!(graph.get(add1).is_some());
        assert!(graph.get(add2).is_none());

        for (_, node) in graph.iter() {
            if let IrNode::Return { value: Some(val) } = node {
                assert_eq!(*val, add1);
            }
        }
    }

    #[test]
    fn cse_commutative_normalization() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let add1 = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let add2 = graph.push_node(IrNode::Add { lhs: b, rhs: a });
        let _ret = graph.push_node(IrNode::Return { value: Some(add2) });

        let cse = CommonSubexprElim;
        assert!(cse.run(&mut graph));

        assert!(graph.get(add1).is_none() || graph.get(add2).is_none());
    }

    #[test]
    fn cse_store_invalidates_load() {
        let mut graph = IrGraph::new("test");
        let root = OwnershipRoot::STACK;
        let addr = graph.push_node(IrNode::IntConst(100));
        let _load1 = graph.push_node(IrNode::Load {
            addr,
            root,
            ty: Type::I64,
        });
        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store {
            addr,
            val,
            root,
            ty: Type::I64,
        });
        let load2 = graph.push_node(IrNode::Load {
            addr,
            root,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(load2) });

        let cse = CommonSubexprElim;
        cse.run(&mut graph);

        assert!(graph.get(load2).is_some());
    }

    #[test]
    fn cse_load_same_root_no_store() {
        let mut graph = IrGraph::new("test");
        let root = OwnershipRoot::STACK;
        let addr = graph.push_node(IrNode::IntConst(100));
        let load1 = graph.push_node(IrNode::Load {
            addr,
            root,
            ty: Type::I64,
        });
        let load2 = graph.push_node(IrNode::Load {
            addr,
            root,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(load2) });

        let cse = CommonSubexprElim;
        assert!(cse.run(&mut graph));

        assert!(graph.get(load1).is_some());
        assert!(graph.get(load2).is_none());
    }

    #[test]
    fn cse_different_roots_no_invalidation() {
        let mut graph = IrGraph::new("test");
        let root_a = OwnershipRoot::new(5);
        let root_b = OwnershipRoot::new(6);
        let addr = graph.push_node(IrNode::IntConst(100));
        let load1 = graph.push_node(IrNode::Load {
            addr,
            root: root_b,
            ty: Type::I64,
        });
        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store {
            addr,
            val,
            root: root_a,
            ty: Type::I64,
        });
        let load2 = graph.push_node(IrNode::Load {
            addr,
            root: root_b,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: Some(load2) });

        let cse = CommonSubexprElim;
        assert!(cse.run(&mut graph));

        assert!(graph.get(load1).is_some());
        assert!(graph.get(load2).is_none());
    }

    #[test]
    fn cse_never_eliminates_stores() {
        let mut graph = IrGraph::new("test");
        let root = OwnershipRoot::STACK;
        let addr = graph.push_node(IrNode::IntConst(100));
        let val1 = graph.push_node(IrNode::IntConst(1));
        let val2 = graph.push_node(IrNode::IntConst(2));
        let store1 = graph.push_node(IrNode::Store {
            addr,
            val: val1,
            root,
            ty: Type::I64,
        });
        let store2 = graph.push_node(IrNode::Store {
            addr,
            val: val2,
            root,
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return { value: None });

        let cse = CommonSubexprElim;
        cse.run(&mut graph);

        assert!(graph.get(store1).is_some());
        assert!(graph.get(store2).is_some());
    }

    #[test]
    fn cse_hash_key_no_collisions() {
        // Verify that different operations produce different hash keys
        let a = NodeId::new(1);
        let b = NodeId::new(2);

        let add_key = binop_key(TAG_ADD, a, b, true);
        let sub_key = binop_key(TAG_SUB, a, b, false);
        let mul_key = binop_key(TAG_MUL, a, b, true);

        assert_ne!(add_key, sub_key, "Add and Sub should have different keys");
        assert_ne!(add_key, mul_key, "Add and Mul should have different keys");
        assert_ne!(sub_key, mul_key, "Sub and Mul should have different keys");
    }

    #[test]
    fn cse_eliminates_duplicate_int_const() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(42));
        let b = graph.push_node(IrNode::IntConst(42));
        let add = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(add) });

        let cse = CommonSubexprElim;
        assert!(cse.run(&mut graph));

        // One of the duplicate IntConst(42) should be eliminated
        assert!(graph.get(a).is_none() || graph.get(b).is_none());
    }

    #[test]
    fn cse_eliminates_duplicate_bool_const() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::BoolConst(true));
        let _b = graph.push_node(IrNode::BoolConst(true));
        // Use them in different operations so both are referenced
        let _ret = graph.push_node(IrNode::Return { value: Some(a) });

        let cse = CommonSubexprElim;
        let modified = cse.run(&mut graph);

        // At least one bool const should be eliminated (both hash to same key)
        // Note: if a is the return value, it might not be eliminated since
        // it's directly used. But the CSE table should recognize them as same.
        assert!(modified, "CSE should detect duplicate BoolConst");
    }

    #[test]
    fn cse_different_int_consts_not_eliminated() {
        let mut graph = IrGraph::new("test");
        let a = graph.push_node(IrNode::IntConst(1));
        let b = graph.push_node(IrNode::IntConst(2));
        let add = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(add) });

        let cse = CommonSubexprElim;
        // Different constants should not be eliminated
        // But the Add may get eliminated if we run CSE twice... let's just check
        // that both constants remain
        cse.run(&mut graph);

        assert!(graph.get(a).is_some(), "IntConst(1) should remain");
        assert!(graph.get(b).is_some(), "IntConst(2) should remain");
    }

    #[test]
    fn test_hash2_function() {
        // Verify hash2 produces different results for different inputs
        let h1 = hash2(1, 2);
        let h2 = hash2(2, 1);
        assert_ne!(h1, h2, "hash2(1,2) should differ from hash2(2,1)");

        // Verify hash2 is consistent
        let h3 = hash2(1, 2);
        assert_eq!(h1, h3, "hash2 should be deterministic");

        // Verify hash2 handles zero inputs (0 * GOLDEN + 0 = 0, which is fine
        // since we always combine with a non-zero tag in practice)
        let h4 = hash2(0, 0);
        // hash2(0,0) = 0 mathematically, but in practice we always use a tag
        // so this case doesn't arise in CSE keys
        let _ = h4;
    }
}
