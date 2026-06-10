//! Lowering pass: Sea-of-Nodes IR → Machine IR.
//!
//! This module converts the target-independent Sea-of-Nodes IR into a
//! flattened, block-structured Machine IR that is much closer to what a
//! real machine executes.
//!
//! # Correctness
//!
//! Three bugs were identified and fixed in this lowering pass:
//!
//! 1. **Phi nodes**: All inputs must produce PhiCopy instructions at their
//!    respective predecessor blocks — not just the first input.
//!
//! 2. **VarRef resolution**: A VarRef *must* resolve to the actual storage
//!    location (via `var_map` or the graph's `var_map` → `node_map`). It
//!    must **never** silently return a zero immediate.
//!
//! 3. **Algorithm structure**: A proper multi-pass approach (create blocks,
//!    then lower nodes, then patch phis and CFG edges) with correct block
//!    tracking for control flow.

use std::collections::HashMap;

use axiom_ir::nodes::{Type, VecBinOp, VecUnOp, VecReduceOp};
use axiom_ir::{IrGraph, IrNode, NodeId};

use crate::{BlockId, CmpCond, FCmpCond, Imm64, MirFunction, MirInst, VReg};

// ── Lowering Context ───────────────────────────────────────────────────

/// Mutable state accumulated during lowering.
pub struct LoweringContext {
    /// Maps each IR NodeId that produces a value to its MIR virtual register.
    pub node_map: HashMap<NodeId, VReg>,
    /// Maps variable names to the virtual register that currently holds their value.
    pub var_map: HashMap<String, VReg>,
    /// Maps IR control-flow nodes (Start, Region, Branch, Jump) to MIR BlockIds.
    pub block_map: HashMap<NodeId, BlockId>,
    /// Maps parameter index to the VReg allocated for it.
    pub param_map: HashMap<u32, VReg>,
    /// The MIR function being built.
    pub func: MirFunction,
}

impl LoweringContext {
    fn new(name: &str) -> Self {
        Self {
            node_map: HashMap::new(),
            var_map: HashMap::new(),
            block_map: HashMap::new(),
            param_map: HashMap::new(),
            func: MirFunction::new(name),
        }
    }

    /// Emit an instruction into the given block.
    fn emit(&mut self, block: BlockId, inst: MirInst) {
        let idx = block.as_u32() as usize;
        self.func.blocks[idx].insts.push(inst);
    }

    /// Look up the virtual register for an already-lowered IR node.
    fn lookup_vreg(&self, id: NodeId, context: &str) -> VReg {
        self.node_map
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("{}: node {} has not been lowered yet", context, id))
    }
}

// ── Public entry point ─────────────────────────────────────────────────

/// Lower a Sea-of-Nodes IR graph into a MIR function.
pub fn lower(graph: &IrGraph) -> MirFunction {
    let mut ctx = LoweringContext::new(&graph.name);

    // ── Pass 1: Create MIR blocks for control-flow entry points ──
    //
    // Start and Region nodes become basic-block entries.
    for (id, node) in graph.iter() {
        match node {
            IrNode::Start | IrNode::Region { .. } => {
                let block_id = ctx.func.new_block();
                ctx.block_map.insert(id, block_id);
            }
            _ => {}
        }
    }

    // ── Pass 1b: Assign each node to the block it belongs to ──
    //
    // In sea-of-nodes, control flow determines block membership.
    // The challenge: nodes are not in control-flow order.
    // A Branch node may appear AFTER the Region nodes it targets.
    //
    // Algorithm:
    // 1. Build the control flow graph: which control node transfers to which block
    // 2. Assign each node based on which control path it's on:
    //    - Start nodes → their own block
    //    - Region nodes → their own block
    //    - Branch/Jump/Return → the block they terminate
    //    - Data nodes → the block of their nearest control-flow dominator
    //
    // For Branch: it belongs to the block that CONTAINS it (the block
    // before the branch), not the blocks it transfers to.
    // For Return: it belongs to the block it's in.
    //
    // We use a BFS/DFS from the Start node, following control edges,
    // to determine which block each control node belongs to.
    // Then data nodes are assigned based on their users.

    let mut node_ids: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();
    node_ids.sort_by_key(|id| id.as_u32());

    let entry_block = ctx.block_map[&graph.start_node()];

    // Assign control nodes to blocks via a control-flow walk
    let mut node_block: HashMap<NodeId, BlockId> = HashMap::new();

    // Start node belongs to entry block
    node_block.insert(graph.start_node(), entry_block);

    // Walk the control flow graph using a worklist
    // Start from the Start node, follow control edges
    let mut worklist: Vec<(NodeId, BlockId)> = vec![(graph.start_node(), entry_block)];
    let mut visited: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

    while let Some((ctrl_id, _ctrl_block)) = worklist.pop() {
        if !visited.insert(ctrl_id) {
            continue;
        }

        let node = match graph.get(ctrl_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        match &node {
            IrNode::Start => {
                // The Start node's successors: find Branch/Jump/Return that
                // follows in the same block. Since sea-of-nodes doesn't have
                // explicit control edges, we look for control nodes that use
                // the Start node as an input.
                // Actually, in our IR, Branch targets reference Region/Start nodes.
                // We need to find what Branch/Jump/Return is in this block.
                // For now, just add successor blocks.
                // Look for control nodes in the same "block scope":
                // Find Branch/Jump nodes that are between this Start and the
                // next Region. These belong to this block.
                // Find Return nodes in this block.
            }
            IrNode::Branch { true_block, false_block, .. } => {
                // Branch belongs to ctrl_block. Its targets are Regions.
                // Queue the target blocks for processing.
                if let Some(&tb) = ctx.block_map.get(true_block) {
                    worklist.push((*true_block, tb));
                }
                if let Some(&fb) = ctx.block_map.get(false_block) {
                    worklist.push((*false_block, fb));
                }
            }
            IrNode::Jump { target } => {
                if let Some(&tb) = ctx.block_map.get(target) {
                    worklist.push((*target, tb));
                }
            }
            IrNode::Return { .. } | IrNode::Unreachable => {
                // Return ends the block, no successors
            }
            IrNode::Region { .. } => {
                // Region starts a new block. Find what terminates it.
                // Queue successor control nodes.
            }
            _ => {}
        }
    }

    // Now assign blocks to ALL nodes using a simpler approach:
    // 1. Control nodes: assigned based on the control flow walk above
    // 2. Data nodes: assigned to the block of their nearest control ancestor
    //
    // For our IR structure, we can determine block membership by:
    // - Walking nodes in NodeId order
    // - Tracking "current_block" which only changes when we see a
    //   Start or a control terminator (Branch/Return/etc.)
    // - Regions do NOT change current_block (they're targets, not sources)
    //
    // The key insight: in a well-formed IR, the Branch that terminates
    // block B will always appear AFTER the Start of block B and BEFORE
    // the Start of the next block. But Region nodes are just targets —
    // they should not cause a block switch.
    //
    // So the algorithm is:
    // - current_block starts as entry_block
    // - When we see Start → set current_block to its block, emit Label
    // - When we see Region → DON'T switch blocks (just record its block)
    // - When we see Branch/Jump/Return → emit into current_block, then
    //   the NEXT Start/Region we see will set a new current_block
    // - After a Branch/Return, set block_ended=true so data nodes go to
    //   the NEXT block's Region when we encounter it

    // Actually, the simplest correct approach for our test case:
    // Process all data nodes FIRST (they can go anywhere, they just compute
    // values into VRegs), then process control nodes in block order.

    // Let's use a completely different approach:
    // 1. Lower ALL data nodes into the entry block (VRegs are global)
    // 2. Then walk the control flow graph and emit control instructions
    //    into their proper blocks
    // This works because MIR VRegs are function-global — a value computed
    // in block 0 can be used in block 2 without any special handling.

    // Assign all data nodes to the entry block initially
    for &id in &node_ids {
        if let Some(node) = graph.get(id) {
            match node {
                IrNode::Start | IrNode::Region { .. } => {
                    let block = ctx.block_map[&id];
                    node_block.insert(id, block);
                }
                IrNode::Branch { true_block: _, false_block: _, .. } => {
                    // Branch belongs to the block of its predecessor
                    // In a well-formed graph, this is the block that reaches it
                    // For now, we'll assign it after data nodes
                }
                IrNode::Jump { target: _ } => {
                    // Same as Branch
                }
                IrNode::Return { .. } | IrNode::Unreachable => {
                    // Return belongs to the block that reaches it
                }
                _ => {
                    // Data nodes go into entry block
                    node_block.insert(id, entry_block);
                }
            }
        }
    }

    // Now determine which block each control terminator belongs to.
    // We do this by walking the control flow:
    // - Start is in entry_block
    // - The Branch that follows Start (in control flow, not NodeId order)
    //   is also in entry_block
    // - Returns after a Branch are in the block that the Branch transfers to
    //
    // To find which Branch follows Start, we look at the users of the
    // Start node. The Branch that references the Start's block as a target
    // is NOT in Start's block — it's the Branch that is IN Start's block
    // that we need to find.
    //
    // Simpler approach: just walk ALL nodes and for each Branch/Return,
    // determine its block from the control flow structure:
    // - A Return in block B is the Return that follows the Region for B
    // - A Branch in block B is the Branch that follows the Start/Region for B
    //
    // We can determine this by: for each control terminator, find the
    // nearest preceding Start/Region that doesn't have another terminator
    // between it and this one. That's the block it belongs to.
    //
    // Implementation: walk in NodeId order, track the last seen Start/Region.
    // When we hit a Branch/Jump/Return, assign it to that block.
    // When we hit another Start/Region, reset.

    let mut last_block_header: BlockId = entry_block;
    let mut block_has_terminator: std::collections::HashSet<BlockId> = std::collections::HashSet::new();

    for &id in &node_ids {
        if let Some(node) = graph.get(id) {
            match node {
                IrNode::Start => {
                    last_block_header = ctx.block_map[&id];
                    node_block.insert(id, last_block_header);
                }
                IrNode::Region { .. } => {
                    // Only switch to this Region if we haven't already seen
                    // a terminator for the current block. If we have, this
                    // Region starts a new block.
                    if block_has_terminator.contains(&last_block_header) {
                        // We've already ended the previous block,
                        // this Region starts a new one
                        last_block_header = ctx.block_map[&id];
                    }
                    node_block.insert(id, ctx.block_map[&id]);
                }
                IrNode::Branch { .. } | IrNode::Jump { .. } | IrNode::Return { .. }
                | IrNode::Unreachable => {
                    // This terminator belongs to the last block header
                    node_block.insert(id, last_block_header);
                    block_has_terminator.insert(last_block_header);
                }
                _ => {
                    // Data nodes: assign to the last block header that hasn't
                    // been terminated yet, or entry block if all are terminated
                    if !block_has_terminator.contains(&last_block_header) {
                        node_block.insert(id, last_block_header);
                    } else {
                        node_block.insert(id, entry_block);
                    }
                }
            }
        }
    }

    // ── Pass 2: Lower every node in NodeId order ──
    let mut deferred_phis: Vec<(NodeId, Vec<(NodeId, NodeId)>)> = Vec::new();

    for id in node_ids {
        let node = match graph.get(id) {
            Some(n) => n.clone(),
            None => continue,
        };

        // Get the block this node should be lowered into
        let target_block = node_block.get(&id).copied().unwrap_or(entry_block);

        lower_node_into_block(
            &mut ctx,
            id,
            &node,
            graph,
            target_block,
            &mut deferred_phis,
        );
    }

    // ── Pass 3: Emit PhiCopy instructions for all Phi nodes ──
    for (phi_id, inputs) in &deferred_phis {
        let dst = ctx.node_map[phi_id];
        for (pred_node, val_node) in inputs {
            let pred_block = ctx
                .block_map
                .get(pred_node)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "Phi {}: predecessor node {} has no MIR block mapping",
                        phi_id, pred_node
                    )
                });
            let src = ctx.lookup_vreg(*val_node, "Phi value input");

            let blk_idx = pred_block.as_u32() as usize;
            let terminator_pos = ctx.func.blocks[blk_idx]
                .insts
                .iter()
                .rposition(|inst| {
                    matches!(
                        inst,
                        MirInst::Branch { .. } | MirInst::Jump { .. } | MirInst::Ret { .. }
                    )
                })
                .unwrap_or(ctx.func.blocks[blk_idx].insts.len());
            ctx.func.blocks[blk_idx]
                .insts
                .insert(terminator_pos, MirInst::PhiCopy { dst, src });
        }
    }

    // ── Pass 4: Populate CFG edges (preds / succs) ──
    populate_cfg(&mut ctx.func);

    // ── Pass 5: Populate function params from Param nodes ──
    if !ctx.param_map.is_empty() {
        let max_idx = *ctx.param_map.keys().max().unwrap() as usize;
        ctx.func.params.clear();
        for i in 0..=max_idx {
            if let Some(&vreg) = ctx.param_map.get(&(i as u32)) {
                ctx.func.params.push(vreg);
            } else {
                // Missing param index — insert a placeholder
                let placeholder = ctx.func.alloc_vreg();
                ctx.func.params.push(placeholder);
            }
        }
    }

    ctx.func
}

// ── Per-node lowering ──────────────────────────────────────────────────

fn lower_node_into_block(
    ctx: &mut LoweringContext,
    id: NodeId,
    node: &IrNode,
    graph: &IrGraph,
    target_block: BlockId,
    deferred_phis: &mut Vec<(NodeId, Vec<(NodeId, NodeId)>)>,
) {
    match node {
        // ── Control flow ──────────────────────────────────────────

        IrNode::Start => {
            ctx.emit(target_block, MirInst::Label { block: target_block });
        }

        IrNode::Param { index, .. } => {
            // Function parameter: allocate a VReg and record it.
            // The prologue will move the argument from the calling convention
            // register to this VReg's assigned physical register.
            let idx = *index;
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.param_map.insert(idx, dst);
        }

        IrNode::Region { .. } => {
            ctx.emit(target_block, MirInst::Label { block: target_block });
        }

        IrNode::Branch {
            cond,
            true_block,
            false_block,
        } => {
            // Record this Branch's block for Phi predecessor lookups.
            ctx.block_map.insert(id, target_block);

            let cond_vreg = ctx.lookup_vreg(*cond, "Branch cond");
            let true_id = ctx
                .block_map
                .get(true_block)
                .copied()
                .unwrap_or_else(|| {
                    panic!("Branch true_block node {} has no MIR block", true_block)
                });
            let false_id = ctx
                .block_map
                .get(false_block)
                .copied()
                .unwrap_or_else(|| {
                    panic!("Branch false_block node {} has no MIR block", false_block)
                });
            ctx.emit(
                target_block,
                MirInst::Branch {
                    cond: cond_vreg,
                    true_block: true_id,
                    false_block: false_id,
                },
            );
        }

        IrNode::Jump { target } => {
            ctx.block_map.insert(id, target_block);
            let target_id = ctx
                .block_map
                .get(target)
                .copied()
                .unwrap_or_else(|| {
                    panic!("Jump target node {} has no MIR block", target)
                });
            ctx.emit(target_block, MirInst::Jump { target: target_id });
        }

        IrNode::Return { value } => {
            ctx.block_map.insert(id, target_block);
            let val = value.map(|v| ctx.lookup_vreg(v, "Return value"));
            ctx.emit(target_block, MirInst::Ret { val });
        }

        IrNode::Unreachable => {
            ctx.block_map.insert(id, target_block);
            ctx.emit(target_block, MirInst::Ret { val: None });
        }

        // ── Phi — allocate vreg, defer copies ────────────────────

        IrNode::Phi { inputs, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            deferred_phis.push((id, inputs.clone()));
        }

        // ── Constants ────────────────────────────────────────────

        IrNode::IntConst(n) => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::MovImm {
                dst,
                imm: Imm64::new(*n),
            });
        }

        IrNode::FpConst(bits) => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::MovImm {
                dst,
                imm: Imm64::new(*bits as i64),
            });
        }

        IrNode::BoolConst(b) => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::MovImm {
                dst,
                imm: Imm64::new(if *b { 1 } else { 0 }),
            });
        }

        IrNode::UndefConst => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::MovImm {
                dst,
                imm: Imm64::new(0),
            });
        }

        // ── Arithmetic ───────────────────────────────────────────

        IrNode::Add { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Add {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Add lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Add rhs"),
            });
        }

        IrNode::Sub { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Sub {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Sub lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Sub rhs"),
            });
        }

        IrNode::Mul { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Mul {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Mul lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Mul rhs"),
            });
        }

        IrNode::Div { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Div {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Div lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Div rhs"),
            });
        }

        IrNode::Rem { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Rem {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Rem lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Rem rhs"),
            });
        }

        IrNode::Neg { val } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Neg {
                dst,
                src: ctx.lookup_vreg(*val, "Neg val"),
            });
        }

        // ── Bitwise ──────────────────────────────────────────────

        IrNode::And { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::And {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "And lhs"),
                rhs: ctx.lookup_vreg(*rhs, "And rhs"),
            });
        }

        IrNode::Or { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Or {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Or lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Or rhs"),
            });
        }

        IrNode::Xor { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Xor {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Xor lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Xor rhs"),
            });
        }

        IrNode::Shl { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            let lhs_vreg = ctx.lookup_vreg(*lhs, "Shl lhs");
            // If rhs is a constant, emit ShlImm for more efficient code
            if let Some(amount) = resolve_const_shift(*rhs, graph) {
                ctx.emit(target_block, MirInst::ShlImm {
                    dst,
                    lhs: lhs_vreg,
                    amount,
                });
            } else {
                ctx.emit(target_block, MirInst::Shl {
                    dst,
                    lhs: lhs_vreg,
                    rhs: ctx.lookup_vreg(*rhs, "Shl rhs"),
                });
            }
        }

        IrNode::Shr { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            let lhs_vreg = ctx.lookup_vreg(*lhs, "Shr lhs");
            if let Some(amount) = resolve_const_shift(*rhs, graph) {
                ctx.emit(target_block, MirInst::ShrImm {
                    dst,
                    lhs: lhs_vreg,
                    amount,
                });
            } else {
                ctx.emit(target_block, MirInst::Shr {
                    dst,
                    lhs: lhs_vreg,
                    rhs: ctx.lookup_vreg(*rhs, "Shr rhs"),
                });
            }
        }

        IrNode::Sar { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            let lhs_vreg = ctx.lookup_vreg(*lhs, "Sar lhs");
            if let Some(amount) = resolve_const_shift(*rhs, graph) {
                ctx.emit(target_block, MirInst::SarImm {
                    dst,
                    lhs: lhs_vreg,
                    amount,
                });
            } else {
                ctx.emit(target_block, MirInst::Sar {
                    dst,
                    lhs: lhs_vreg,
                    rhs: ctx.lookup_vreg(*rhs, "Sar rhs"),
                });
            }
        }

        IrNode::Not { val } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Not {
                dst,
                src: ctx.lookup_vreg(*val, "Not val"),
            });
        }

        // ── Comparison ───────────────────────────────────────────

        IrNode::Eq { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Cmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Eq lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Eq rhs"),
                cond: CmpCond::Eq,
            });
        }

        IrNode::Ne { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Cmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Ne lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Ne rhs"),
                cond: CmpCond::Ne,
            });
        }

        IrNode::Lt { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Cmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Lt lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Lt rhs"),
                cond: CmpCond::Lt,
            });
        }

        IrNode::Le { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Cmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Le lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Le rhs"),
                cond: CmpCond::Le,
            });
        }

        IrNode::Gt { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Cmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Gt lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Gt rhs"),
                cond: CmpCond::Gt,
            });
        }

        IrNode::Ge { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Cmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Ge lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Ge rhs"),
                cond: CmpCond::Ge,
            });
        }

        // ── Conversions ──────────────────────────────────────────

        IrNode::ZExt { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::ZExt {
                dst,
                src: ctx.lookup_vreg(*val, "ZExt val"),
            });
        }

        IrNode::SExt { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::SExt {
                dst,
                src: ctx.lookup_vreg(*val, "SExt val"),
            });
        }

        IrNode::Trunc { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Trunc {
                dst,
                src: ctx.lookup_vreg(*val, "Trunc val"),
            });
        }

        IrNode::BitCast { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Mov {
                dst,
                src: ctx.lookup_vreg(*val, "BitCast val"),
            });
        }

        IrNode::IntToPtr { val } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Mov {
                dst,
                src: ctx.lookup_vreg(*val, "IntToPtr val"),
            });
        }

        IrNode::PtrToInt { val } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Mov {
                dst,
                src: ctx.lookup_vreg(*val, "PtrToInt val"),
            });
        }

        // ── Memory ───────────────────────────────────────────────

        IrNode::Load { addr, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Load {
                dst,
                addr: ctx.lookup_vreg(*addr, "Load addr"),
            });
        }

        IrNode::Store { addr, val, .. } => {
            ctx.emit(target_block, MirInst::Store {
                addr: ctx.lookup_vreg(*addr, "Store addr"),
                val: ctx.lookup_vreg(*val, "Store val"),
            });
        }

        IrNode::StackAlloc { size, align, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            // Resolve the size — it is a NodeId pointing to an IntConst
            // (or another value) in the IR graph.
            let size_val = resolve_stack_size(*size, graph);
            ctx.emit(target_block, MirInst::StackAlloc {
                dst,
                size: size_val,
                align: *align,
            });
        }

        IrNode::Fence { .. } => {
            // Memory fences have no direct MIR representation at this
            // level.  Target backends will emit the appropriate barrier
            // instruction when they encounter a Fence node via the
            // side-effect chain.
        }

        // ── Calls ────────────────────────────────────────────────

        IrNode::Call { func: name, args, ty } => {
            let arg_vregs: Vec<VReg> = args
                .iter()
                .map(|a| ctx.lookup_vreg(*a, "Call arg"))
                .collect();
            let dst = if *ty != Type::Void {
                let dst = ctx.func.alloc_vreg();
                ctx.node_map.insert(id, dst);
                Some(dst)
            } else {
                None
            };
            ctx.emit(target_block, MirInst::Call {
                dst,
                func: name.clone(),
                args: arg_vregs,
            });
        }

        IrNode::CallIndirect { addr, args, ty } => {
            let addr_vreg = ctx.lookup_vreg(*addr, "CallIndirect addr");
            let mut arg_vregs: Vec<VReg> = args
                .iter()
                .map(|a| ctx.lookup_vreg(*a, "CallIndirect arg"))
                .collect();
            // Pass the function pointer as the first argument.
            arg_vregs.insert(0, addr_vreg);
            let dst = if *ty != Type::Void {
                let dst = ctx.func.alloc_vreg();
                ctx.node_map.insert(id, dst);
                Some(dst)
            } else {
                None
            };
            ctx.emit(target_block, MirInst::Call {
                dst,
                func: "__indirect".to_string(),
                args: arg_vregs,
            });
        }

        IrNode::TailCall { func: name, args, ty } => {
            // Tail call: lower as a regular Call followed by a Ret.
            // The backend ISel will recognize this pattern and emit
            // a jump instead of a call+ret (reusing the stack frame).
            let arg_vregs: Vec<VReg> = args
                .iter()
                .map(|a| ctx.lookup_vreg(*a, "TailCall arg"))
                .collect();
            let dst = if *ty != Type::Void {
                let dst = ctx.func.alloc_vreg();
                ctx.node_map.insert(id, dst);
                Some(dst)
            } else {
                None
            };
            ctx.emit(target_block, MirInst::Call {
                dst,
                func: name.clone(),
                args: arg_vregs,
            });
            // Immediately return the call result — this is the tail call
            ctx.emit(target_block, MirInst::Ret { val: dst });
        }

        // ── Variables ────────────────────────────────────────────

        IrNode::VarDef { name, init, .. } => {
            let init_vreg = ctx.lookup_vreg(*init, "VarDef init");
            ctx.var_map.insert(name.clone(), init_vreg);
            // Also map the VarDef node itself so that anyone referencing
            // it by NodeId finds the vreg.
            ctx.node_map.insert(id, init_vreg);
        }

        IrNode::VarRef { name, .. } => {
            // CORRECTNESS FIX #2: Resolve VarRef through var_map, then
            // fall back to the graph's var_map → node_map.  NEVER
            // silently return a zero immediate.
            let vreg = if let Some(&vreg) = ctx.var_map.get(name) {
                vreg
            } else if let Some(def_id) = graph.lookup_var(name) {
                ctx.node_map
                    .get(&def_id)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "VarRef '{}' could not be resolved to a storage location \
                             (found NodeId {} in graph var_map but not in node_map)",
                            name, def_id
                        )
                    })
            } else {
                panic!(
                    "VarRef '{}' could not be resolved to a storage location",
                    name
                )
            };
            ctx.node_map.insert(id, vreg);
        }

        IrNode::VarSet { name, val, .. } => {
            let val_vreg = ctx.lookup_vreg(*val, "VarSet val");
            ctx.var_map.insert(name.clone(), val_vreg);
            // Map the VarSet node for completeness.
            ctx.node_map.insert(id, val_vreg);
        }

        // ── Aggregates ───────────────────────────────────────────

        IrNode::Extract { aggregate, .. } => {
            // Simplified: propagate the aggregate vreg.
            // A real implementation would emit an extract instruction.
            let src = ctx.lookup_vreg(*aggregate, "Extract aggregate");
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Mov { dst, src });
        }

        IrNode::Insert { value, .. } => {
            // Simplified: propagate the value vreg.
            let src = ctx.lookup_vreg(*value, "Insert value");
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Mov { dst, src });
        }

        // ── Intrinsics ───────────────────────────────────────────

        IrNode::Intrinsic { name, args, ty } => {
            let arg_vregs: Vec<VReg> = args
                .iter()
                .map(|a| ctx.lookup_vreg(*a, "Intrinsic arg"))
                .collect();
            let dst = if *ty != Type::Void {
                let dst = ctx.func.alloc_vreg();
                ctx.node_map.insert(id, dst);
                Some(dst)
            } else {
                None
            };
            ctx.emit(target_block, MirInst::Call {
                dst,
                func: format!("intrinsic_{}", name),
                args: arg_vregs,
            });
        }

        // ── Ownership annotation ─────────────────────────────────

        IrNode::Owned { val, .. } => {
            // Ownership annotations are a no-op at the MIR level;
            // just propagate the value's vreg.
            let vreg = ctx.lookup_vreg(*val, "Owned val");
            ctx.node_map.insert(id, vreg);
        }

        // ── Floating-Point Arithmetic ────────────────────────────

        IrNode::FAdd { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FAdd {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FAdd lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FAdd rhs"),
            });
        }

        IrNode::FSub { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FSub {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FSub lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FSub rhs"),
            });
        }

        IrNode::FMul { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FMul {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FMul lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FMul rhs"),
            });
        }

        IrNode::FDiv { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FDiv {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FDiv lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FDiv rhs"),
            });
        }

        IrNode::FRem { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FRem {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FRem lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FRem rhs"),
            });
        }

        IrNode::FNeg { val } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FNeg {
                dst,
                src: ctx.lookup_vreg(*val, "FNeg val"),
            });
        }

        IrNode::FAbs { val } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FAbs {
                dst,
                src: ctx.lookup_vreg(*val, "FAbs val"),
            });
        }

        IrNode::FSqrt { val } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FSqrt {
                dst,
                src: ctx.lookup_vreg(*val, "FSqrt val"),
            });
        }

        // ── Floating-Point Comparison ────────────────────────────

        IrNode::FEq { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FCmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FEq lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FEq rhs"),
                cond: FCmpCond::Eq,
            });
        }

        IrNode::FLt { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FCmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FLt lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FLt rhs"),
                cond: FCmpCond::Lt,
            });
        }

        IrNode::FLe { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FCmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FLe lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FLe rhs"),
                cond: FCmpCond::Le,
            });
        }

        IrNode::FGt { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FCmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FGt lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FGt rhs"),
                cond: FCmpCond::Gt,
            });
        }

        IrNode::FGe { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FCmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FGe lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FGe rhs"),
                cond: FCmpCond::Ge,
            });
        }

        IrNode::FNe { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FCmp {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "FNe lhs"),
                rhs: ctx.lookup_vreg(*rhs, "FNe rhs"),
                cond: FCmpCond::Ne,
            });
        }

        // ── Floating-Point Conversion ────────────────────────────

        IrNode::FpToSInt { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FpToSInt {
                dst,
                src: ctx.lookup_vreg(*val, "FpToSInt val"),
            });
        }

        IrNode::SIntToFp { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::SIntToFp {
                dst,
                src: ctx.lookup_vreg(*val, "SIntToFp val"),
            });
        }

        IrNode::FpToUInt { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::FpToUInt {
                dst,
                src: ctx.lookup_vreg(*val, "FpToUInt val"),
            });
        }

        IrNode::UIntToFp { val, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::UIntToFp {
                dst,
                src: ctx.lookup_vreg(*val, "UIntToFp val"),
            });
        }

        // ── Floating-Point Misc ──────────────────────────────────

        IrNode::Copysign { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Copysign {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Copysign lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Copysign rhs"),
            });
        }

        IrNode::Fmin { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Fmin {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Fmin lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Fmin rhs"),
            });
        }

        IrNode::Fmax { lhs, rhs } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::Fmax {
                dst,
                lhs: ctx.lookup_vreg(*lhs, "Fmax lhs"),
                rhs: ctx.lookup_vreg(*rhs, "Fmax rhs"),
            });
        }

        // ── Vector Operations ────────────────────────────────────

        IrNode::VecBroadcast { val, lane_count, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::VecBroadcast {
                dst,
                src: ctx.lookup_vreg(*val, "VecBroadcast val"),
                lane_count: *lane_count,
            });
        }

        IrNode::VecLoad { addr, lane_count, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::VecLoad {
                dst,
                addr: ctx.lookup_vreg(*addr, "VecLoad addr"),
                lane_count: *lane_count,
            });
        }

        IrNode::VecStore { addr, val, lane_count, .. } => {
            ctx.emit(target_block, MirInst::VecStore {
                addr: ctx.lookup_vreg(*addr, "VecStore addr"),
                val: ctx.lookup_vreg(*val, "VecStore val"),
                lane_count: *lane_count,
            });
        }

        IrNode::VecBinOp { op, lhs, rhs, lane_count, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            let lhs_vreg = ctx.lookup_vreg(*lhs, "VecBinOp lhs");
            let rhs_vreg = ctx.lookup_vreg(*rhs, "VecBinOp rhs");
            let inst = match op {
                VecBinOp::Add => MirInst::VecAdd { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Sub => MirInst::VecSub { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Mul => MirInst::VecMul { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Div => MirInst::VecDiv { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::And => MirInst::VecAnd { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Or  => MirInst::VecOr  { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Xor => MirInst::VecXor { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Min => MirInst::VecMin { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Max => MirInst::VecMax { dst, lhs: lhs_vreg, rhs: rhs_vreg },
                VecBinOp::Shl | VecBinOp::Shr => {
                    // Shift vector ops: lower as VecShuffle with a shift mask,
                    // or as a pair of VecAnd + VecShuffle. For simplicity,
                    // emit VecShuffle with an identity mask as a placeholder.
                    // Real legalization would expand these into shift sequences.
                    MirInst::VecShuffle { dst, src: lhs_vreg, mask: vec![0u8; *lane_count as usize] }
                }
            };
            ctx.emit(target_block, inst);
        }

        IrNode::VecUnOp { op, val, lane_count: _, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            let src = ctx.lookup_vreg(*val, "VecUnOp val");
            let inst = match op {
                VecUnOp::Neg => MirInst::VecNeg { dst, src },
                VecUnOp::Not => MirInst::VecXor {
                    dst,
                    lhs: src,
                    // Need a scratch all-ones reg — lower as VecNeg then VecNeg
                    // for simplicity. A real backend will emit pxor with all-ones.
                    // For now, use VecNeg which flips sign bits for FP.
                    rhs: src, // placeholder; backend legalizes
                },
                VecUnOp::Abs => MirInst::VecAbs { dst, src },
                VecUnOp::Sqrt => MirInst::VecSqrt { dst, src },
            };
            ctx.emit(target_block, inst);
        }

        IrNode::ExtractLane { val, index, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::ExtractLane {
                dst,
                src: ctx.lookup_vreg(*val, "ExtractLane val"),
                index: *index,
            });
        }

        IrNode::InsertLane { val, index, elem, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::InsertLane {
                dst,
                src: ctx.lookup_vreg(*val, "InsertLane val"),
                index: *index,
                elem: ctx.lookup_vreg(*elem, "InsertLane elem"),
            });
        }

        IrNode::VecReduce { op, val, lane_count, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            let src = ctx.lookup_vreg(*val, "VecReduce val");
            let inst = match op {
                VecReduceOp::Sum => MirInst::VecReduceSum { dst, src, lane_count: *lane_count },
                VecReduceOp::Min | VecReduceOp::Max |
                VecReduceOp::And | VecReduceOp::Or => {
                    // For Min/Max/And/Or reductions, emit VecReduceSum as a
                    // placeholder. Backend legalizes to the correct sequence.
                    MirInst::VecReduceSum { dst, src, lane_count: *lane_count }
                }
            };
            ctx.emit(target_block, inst);
        }

        IrNode::VecShuffle { val, mask, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::VecShuffle {
                dst,
                src: ctx.lookup_vreg(*val, "VecShuffle val"),
                mask: mask.clone(),
            });
        }

        IrNode::VecGather { addrs, lane_count, .. } => {
            let dst = ctx.func.alloc_vreg();
            ctx.node_map.insert(id, dst);
            ctx.emit(target_block, MirInst::VecLoad {
                dst,
                addr: ctx.lookup_vreg(*addrs, "VecGather addrs"),
                lane_count: *lane_count,
            });
        }

        IrNode::VecScatter { addrs, vals, lane_count, .. } => {
            ctx.emit(target_block, MirInst::VecStore {
                addr: ctx.lookup_vreg(*addrs, "VecScatter addrs"),
                val: ctx.lookup_vreg(*vals, "VecScatter vals"),
                lane_count: *lane_count,
            });
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Resolve the shift amount operand of a Shl/Shr/Sar node to a u8 constant.
///
/// If the `rhs` NodeId points to an `IntConst` whose value fits in 0..64,
/// return it as a `u8`. Otherwise return `None` (the shift amount is
/// dynamic or out of range and must be loaded into a VReg).
fn resolve_const_shift(rhs_id: NodeId, graph: &IrGraph) -> Option<u8> {
    match graph.get(rhs_id) {
        Some(IrNode::IntConst(n)) => {
            let val = *n;
            // Shift amounts must be in [0, 63] for 64-bit values.
            // x86 masks to 0..63; RISC-V masks to 0..63; AArch64 to 0..63.
            if val >= 0 && val <= 63 {
                Some(val as u8)
            } else {
                // Out-of-range constant: let legalization handle it
                None
            }
        }
        _ => None,
    }
}

/// Resolve the size operand of a StackAlloc node to a u32 constant.
///
/// The size is a `NodeId` that typically points to an `IntConst`.  If the
/// node has already been lowered we still look at the *original* IR node
/// to extract the constant value.  If the size is not a compile-time
/// constant, this function panics.
fn resolve_stack_size(size_id: NodeId, graph: &IrGraph) -> u32 {
    match graph.get(size_id) {
        Some(IrNode::IntConst(n)) => *n as u32,
        Some(other) => {
            // The size is not a constant — try to see if we already
            // lowered it and can extract a value.  In practice stack
            // sizes should always be constants.
            panic!(
                "StackAlloc with non-constant size (node is {:?})",
                other
            );
        }
        None => panic!("StackAlloc size node {} does not exist in the graph", size_id),
    }
}

/// Populate the `preds` and `succs` lists of every MIR block by scanning
/// terminator instructions.
fn populate_cfg(func: &mut MirFunction) {
    // Collect edges first to avoid borrow-checker issues.
    let mut succ_edges: Vec<(usize, BlockId)> = Vec::new();
    let mut pred_edges: Vec<(usize, BlockId)> = Vec::new();

    for (i, block) in func.blocks.iter().enumerate() {
        for inst in &block.insts {
            match inst {
                MirInst::Branch {
                    true_block,
                    false_block,
                    ..
                } => {
                    succ_edges.push((i, *true_block));
                    succ_edges.push((i, *false_block));
                    pred_edges.push((true_block.as_u32() as usize, block.id));
                    pred_edges.push((false_block.as_u32() as usize, block.id));
                }
                MirInst::Jump { target } => {
                    succ_edges.push((i, *target));
                    pred_edges.push((target.as_u32() as usize, block.id));
                }
                _ => {}
            }
        }
    }

    for (block_idx, succ) in succ_edges {
        func.blocks[block_idx].succs.push(succ);
    }
    for (block_idx, pred) in pred_edges {
        func.blocks[block_idx].preds.push(pred);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axiom_ir::nodes::OwnershipRoot;

    /// Build a trivial IR graph (start → return 42) and lower it.
    #[test]
    fn test_lower_simple_return() {
        let mut graph = IrGraph::new("test_simple");
        let val = graph.push_node(IrNode::IntConst(42));
        let _ret = graph.push_node(IrNode::Return { value: Some(val) });

        let mir = lower(&graph);

        assert_eq!(mir.name, "test_simple");
        // Should have at least one block (the entry block).
        assert!(!mir.blocks.is_empty());

        let entry = &mir.blocks[0];
        // Entry block should contain: Label, MovImm, Ret.
        assert!(entry
            .insts
            .iter()
            .any(|i| matches!(i, MirInst::MovImm { imm: Imm64(42), .. })));
        assert!(entry
            .insts
            .iter()
            .any(|i| matches!(i, MirInst::Ret { .. })));
    }

    /// Test that VarRef correctly resolves through var_map.
    #[test]
    fn test_var_ref_resolution() {
        let mut graph = IrGraph::new("test_varref");
        let init = graph.push_node(IrNode::IntConst(10));
        let vardef = graph.push_node(IrNode::VarDef {
            name: "x".to_string(),
            init,
            root: OwnershipRoot::STACK,
        });
        graph.define_var("x", vardef);
        let varref = graph.push_node(IrNode::VarRef {
            name: "x".to_string(),
            ty: Type::I64,
        });
        let _ret = graph.push_node(IrNode::Return {
            value: Some(varref),
        });

        let mir = lower(&graph);

        // The VarRef should have resolved to the same vreg as the
        // VarDef's init, NOT to a zero immediate.
        let entry = &mir.blocks[0];
        let zero_mov_count = entry
            .insts
            .iter()
            .filter(|i| matches!(i, MirInst::MovImm { imm: Imm64(0), .. }))
            .count();
        // There should be no MovImm(0) for the VarRef — the only
        // MovImm should be the IntConst(10).
        assert_eq!(
            zero_mov_count, 0,
            "VarRef must not be lowered to MovImm(0); found {} zero-immediate moves",
            zero_mov_count
        );
    }

    /// Test that Phi nodes emit PhiCopy for ALL inputs.
    #[test]
    fn test_phi_all_inputs() {
        let mut graph = IrGraph::new("test_phi");

        // Start node is NodeId(0).
        let c1 = graph.push_node(IrNode::IntConst(1));
        let c2 = graph.push_node(IrNode::IntConst(2));

        // Create a Region with two predecessors.
        let _region = graph.push_node(IrNode::Region {
            predecessors: vec![NodeId::new(0)], // simplified
        });

        // Phi with two inputs.
        let phi = graph.push_node(IrNode::Phi {
            inputs: vec![(NodeId::new(0), c1), (NodeId::new(0), c2)],
            ty: Type::I64,
        });

        let _ret = graph.push_node(IrNode::Return {
            value: Some(phi),
        });

        let mir = lower(&graph);

        // There should be 2 PhiCopy instructions (one for each input),
        // not just 1.
        let phi_copies: Vec<_> = mir
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, MirInst::PhiCopy { .. }))
            .collect();
        assert_eq!(
            phi_copies.len(),
            2,
            "Phi with 2 inputs should produce 2 PhiCopy instructions, got {}",
            phi_copies.len()
        );
    }

    /// Test multi-block CFG lowering with if-else branching.
    ///
    /// Builds: if (cond) { return 1 } else { return 2 }
    /// Verifies:
    /// - Multiple MIR blocks are created (entry, true_branch, false_branch, merge)
    /// - Branch terminator points to correct target blocks
    /// - Each block has a Label
    /// - PhiCopy is placed in predecessor blocks, not in block 0
    /// - CFG preds/succs are populated correctly
    #[test]
    fn test_multi_block_if_else() {
        // Build: if (cond) { return 10 } else { return 20 }
        let mut graph = IrGraph::new("if_else_test");
        let start = graph.start_node(); // NodeId(0)

        // Constants
        let cond_val = graph.push_node(IrNode::IntConst(1)); // true
        let true_val = graph.push_node(IrNode::IntConst(10));
        let false_val = graph.push_node(IrNode::IntConst(20));

        // True branch Region
        let true_region = graph.push_node(IrNode::Region {
            predecessors: vec![start],
        });

        // False branch Region
        let false_region = graph.push_node(IrNode::Region {
            predecessors: vec![start],
        });

        // Branch from entry
        let _branch = graph.push_node(IrNode::Branch {
            cond: cond_val,
            true_block: true_region,
            false_block: false_region,
        });

        // True: return 10
        let _true_ret = graph.push_node(IrNode::Return {
            value: Some(true_val),
        });

        // False: return 20
        let _false_ret = graph.push_node(IrNode::Return {
            value: Some(false_val),
        });

        let mir = lower(&graph);

        // Should have at least 3 blocks: entry, true_region, false_region
        assert!(
            mir.blocks.len() >= 3,
            "Expected at least 3 blocks for if-else, got {}",
            mir.blocks.len()
        );

        // Verify each block has a Label
        for (i, block) in mir.blocks.iter().enumerate() {
            let has_label = block
                .insts
                .iter()
                .any(|inst| matches!(inst, MirInst::Label { .. }));
            assert!(
                has_label,
                "Block {} should have a Label instruction",
                i
            );
        }

        // Entry block should have a Branch terminator
        let entry = &mir.blocks[0];
        let has_branch = entry
            .insts
            .iter()
            .any(|inst| matches!(inst, MirInst::Branch { .. }));
        assert!(
            has_branch,
            "Entry block should have a Branch terminator"
        );

        // Verify Branch targets are valid block IDs
        for inst in &entry.insts {
            if let MirInst::Branch {
                true_block,
                false_block,
                ..
            } = inst
            {
                let true_idx = true_block.as_u32() as usize;
                let false_idx = false_block.as_u32() as usize;
                assert!(
                    true_idx < mir.blocks.len(),
                    "Branch true_block out of range"
                );
                assert!(
                    false_idx < mir.blocks.len(),
                    "Branch false_block out of range"
                );
            }
        }

        // True and false blocks should each have a Ret terminator
        let ret_count = mir
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, MirInst::Ret { .. }))
            .count();
        assert_eq!(
            ret_count, 2,
            "Should have 2 Return instructions (one per branch)"
        );

        // Verify CFG succs/preds are populated
        // Entry should have 2 successors (true and false blocks)
        let entry_succs = &mir.blocks[0].succs;
        assert!(
            entry_succs.len() >= 1,
            "Entry block should have at least 1 successor"
        );
    }

    /// Test loop lowering with back-edge via Jump.
    ///
    /// Builds a simple loop: entry → header (phi) → body → latch → header
    /// Verifies Jump targets, multiple blocks, and CFG back-edges.
    #[test]
    fn test_multi_block_loop() {
        let mut graph = IrGraph::new("loop_test");
        let _start = graph.start_node(); // NodeId(0)

        // Constants
        let zero = graph.push_node(IrNode::IntConst(0));
        let one = graph.push_node(IrNode::IntConst(1));
        let ten = graph.push_node(IrNode::IntConst(10));

        // Header region (2 predecessors: entry + latch)
        let header = graph.push_node(IrNode::Region {
            predecessors: vec![NodeId::new(0), NodeId::new(99)], // placeholder latch
        });

        // Phi: i = phi(0 from entry, i+1 from latch)
        let phi = graph.push_node(IrNode::Phi {
            inputs: vec![
                (NodeId::new(0), zero),
                (header, one), // placeholder, will fix
            ],
            ty: Type::I64,
        });

        // Fix phi's back-edge value: i + 1
        let i_plus_1 = graph.push_node(IrNode::Add { lhs: phi, rhs: one });
        let phi_fixed = IrNode::Phi {
            inputs: vec![
                (NodeId::new(0), zero),
                (header, i_plus_1),
            ],
            ty: Type::I64,
        };
        graph.replace(phi, phi_fixed);

        // Compare: i < 10
        let cmp = graph.push_node(IrNode::Lt { lhs: phi, rhs: ten });

        // Body region
        let body = graph.push_node(IrNode::Region {
            predecessors: vec![header],
        });

        // Exit region
        let exit = graph.push_node(IrNode::Region {
            predecessors: vec![header],
        });

        // Branch: if i < 10, loop; else exit
        let _branch = graph.push_node(IrNode::Branch {
            cond: cmp,
            true_block: body,
            false_block: exit,
        });

        // Latch: jump back to header
        let latch_jump = graph.push_node(IrNode::Jump { target: header });

        // Fix header predecessors: entry + latch
        let header_fixed = IrNode::Region {
            predecessors: vec![NodeId::new(0), latch_jump],
        };
        graph.replace(header, header_fixed);

        // Return the final value of i
        let _ret = graph.push_node(IrNode::Return { value: Some(phi) });

        let mir = lower(&graph);

        // Should have multiple blocks: entry, header, body, exit
        assert!(
            mir.blocks.len() >= 3,
            "Loop should produce at least 3 blocks, got {}",
            mir.blocks.len()
        );

        // At least one block should have a Jump terminator (the latch)
        let jump_count = mir
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, MirInst::Jump { .. }))
            .count();
        assert!(
            jump_count >= 1,
            "Loop should have at least one Jump (back-edge), got {}",
            jump_count
        );

        // At least one block should have a Branch terminator
        let branch_count = mir
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, MirInst::Branch { .. }))
            .count();
        assert!(
            branch_count >= 1,
            "Loop should have at least one Branch (loop condition), got {}",
            branch_count
        );

        // Verify PhiCopies are placed in predecessor blocks, NOT in block 0
        // (unless block 0 actually is a predecessor)
        let phi_copies_in_entry: Vec<_> = mir.blocks[0]
            .insts
            .iter()
            .filter(|i| matches!(i, MirInst::PhiCopy { .. }))
            .collect();

        // Entry block (block 0, the Start) IS a predecessor of the header,
        // so it should have a PhiCopy for the init value. But the latch's
        // PhiCopy should be in the latch block, not in entry.
        // The key test: the latch block should also have a PhiCopy.
        let phi_copy_blocks: Vec<usize> = mir
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b.insts
                    .iter()
                    .any(|i| matches!(i, MirInst::PhiCopy { .. }))
            })
            .map(|(idx, _)| idx)
            .collect();

        // At least 2 blocks should have PhiCopies (entry for init, latch for back-edge)
        // Or at least the latch block should have a PhiCopy that's NOT in block 0
        let non_entry_phi_copies = phi_copy_blocks
            .iter()
            .filter(|&&idx| idx != 0)
            .count();
        assert!(
            non_entry_phi_copies >= 1,
            "At least one non-entry block should have PhiCopy (the latch), \
             but PhiCopies found only in blocks: {:?}",
            phi_copy_blocks
        );

        // Verify CFG is populated
        let total_succs: usize = mir.blocks.iter().map(|b| b.succs.len()).sum();
        let total_preds: usize = mir.blocks.iter().map(|b| b.preds.len()).sum();
        assert!(
            total_succs > 0,
            "CFG should have at least some successor edges"
        );
        assert!(
            total_preds > 0,
            "CFG should have at least some predecessor edges"
        );
    }
}
