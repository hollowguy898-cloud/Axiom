//! Axiom Block Layout — Profile-Guided Basic Block Layout.
//!
//! This crate implements the Pettis-Hansen algorithm for basic block layout,
//! which orders basic blocks to maximize fall-through edges and minimize
//! branch mispredictions. It also provides a synthetic profile generator
//! that uses static heuristics when profile data is unavailable.
//!
//! # Core Algorithm
//!
//! The Pettis-Hansen algorithm works as follows:
//! 1. Each basic block starts as its own chain.
//! 2. All edges are sorted by execution frequency (descending).
//! 3. For each edge, if the source block is at the tail of one chain and
//!    the destination is at the head of another, the chains are merged
//!    (enabling a fall-through in the final layout).
//! 4. The final block order is derived from the concatenated chains.
//!
//! # Synthetic Profiles
//!
//! When profile data is not available, `synthesize_profile` creates a
//! synthetic profile using static heuristics:
//! - Loop back edges are considered hot.
//! - Early exit edges (to return blocks) are considered cold.
//! - Normal successor edges get moderate weight.

use axiom_mir::{BlockId, MirBlock, MirFunction, MirInst};
use std::collections::HashMap;

// ── Edge Profile ────────────────────────────────────────────────────────

/// Edge profile data: maps (source_block, dest_block) → execution count.
#[derive(Debug, Clone)]
pub struct EdgeProfile {
    /// Execution counts for edges (source, destination) → count.
    pub edges: HashMap<(BlockId, BlockId), u64>,
    /// Execution counts for individual blocks.
    pub block_counts: HashMap<BlockId, u64>,
}

impl EdgeProfile {
    /// Create a new, empty profile.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            block_counts: HashMap::new(),
        }
    }

    /// Record an edge with its execution count.
    pub fn record_edge(&mut self, from: BlockId, to: BlockId, count: u64) {
        *self.edges.entry((from, to)).or_insert(0) += count;
        *self.block_counts.entry(from).or_insert(0) += count;
        *self.block_counts.entry(to).or_insert(0) += count;
    }

    /// Get the weight (execution count) of a specific edge.
    pub fn edge_weight(&self, from: BlockId, to: BlockId) -> u64 {
        self.edges.get(&(from, to)).copied().unwrap_or(0)
    }

    /// Get the weight (execution count) of a specific block.
    pub fn block_weight(&self, block: BlockId) -> u64 {
        self.block_counts.get(&block).copied().unwrap_or(0)
    }

    /// Total weight of all edges (used for improvement estimation).
    pub fn total_edge_weight(&self) -> u64 {
        self.edges.values().sum()
    }
}

impl Default for EdgeProfile {
    fn default() -> Self {
        Self::new()
    }
}

// ── Layout Result ───────────────────────────────────────────────────────

/// Result of block layout optimization.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Ordered list of block IDs in the optimal layout.
    pub block_order: Vec<BlockId>,
    /// Estimated I-cache miss reduction percentage.
    pub estimated_improvement: f64,
}

// ── Chain (for Pettis-Hansen) ───────────────────────────────────────────

/// A chain of blocks that will be laid out contiguously.
#[derive(Debug, Clone)]
struct Chain {
    /// Blocks in this chain, in order.
    blocks: Vec<BlockId>,
}

impl Chain {
    fn new(block: BlockId) -> Self {
        Self {
            blocks: vec![block],
        }
    }

    /// The first block in the chain (head).
    fn head(&self) -> BlockId {
        self.blocks[0]
    }

    /// The last block in the chain (tail).
    fn tail(&self) -> BlockId {
        *self.blocks.last().unwrap()
    }

    /// Merge another chain onto the end of this one.
    #[allow(dead_code)]
    fn merge(&mut self, other: Chain) {
        self.blocks.extend(other.blocks);
    }

    /// Check if this chain contains a specific block.
    #[allow(dead_code)]
    fn contains(&self, block: BlockId) -> bool {
        self.blocks.contains(&block)
    }
}

// ── Pettis-Hansen Algorithm ─────────────────────────────────────────────

/// Pettis-Hansen block layout algorithm.
///
/// Orders basic blocks to maximize fall-through edges (minimize branch
/// mispredictions). The algorithm greedily merges chains of blocks by
/// considering edges in order of decreasing frequency.
pub fn pettis_hansen_layout(func: &MirFunction, profile: &EdgeProfile) -> LayoutResult {
    let blocks: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();

    // Edge case: 0 or 1 blocks
    if blocks.len() <= 1 {
        let total = profile.total_edge_weight() as f64;
        return LayoutResult {
            block_order: blocks,
            estimated_improvement: if total > 0.0 { 100.0 } else { 0.0 },
        };
    }

    // 1. Initialize: each block is its own chain
    let mut chains: Vec<Chain> = blocks.iter().map(|&b| Chain::new(b)).collect();

    // Map from block → chain index
    let mut block_to_chain: HashMap<BlockId, usize> = HashMap::new();
    for (i, chain) in chains.iter().enumerate() {
        for &block in &chain.blocks {
            block_to_chain.insert(block, i);
        }
    }

    // 2. Sort edges by frequency (descending)
    let mut sorted_edges: Vec<((BlockId, BlockId), u64)> = profile
        .edges
        .iter()
        .map(|(&edge, &weight)| (edge, weight))
        .collect();
    sorted_edges.sort_by(|a, b| b.1.cmp(&a.1));

    // 3. For each edge, try to merge chains
    let mut chain_alive: Vec<bool> = vec![true; chains.len()];

    for ((src, dst), _weight) in &sorted_edges {
        // Both blocks must be in alive chains
        let src_chain_idx = match block_to_chain.get(src) {
            Some(&idx) if chain_alive[idx] => idx,
            _ => continue,
        };
        let dst_chain_idx = match block_to_chain.get(dst) {
            Some(&idx) if chain_alive[idx] => idx,
            _ => continue,
        };

        // Must be different chains
        if src_chain_idx == dst_chain_idx {
            continue;
        }

        // src must be at the tail of its chain, dst must be at the head of its chain
        if chains[src_chain_idx].tail() != *src {
            continue;
        }
        if chains[dst_chain_idx].head() != *dst {
            continue;
        }

        // Merge: append dst_chain onto src_chain
        let dst_blocks = chains[dst_chain_idx].blocks.clone();
        chains[src_chain_idx].blocks.extend(dst_blocks);

        // Update block_to_chain for all blocks in the merged-away chain
        for &block in &chains[dst_chain_idx].blocks {
            block_to_chain.insert(block, src_chain_idx);
        }

        // Mark the dst chain as dead
        chain_alive[dst_chain_idx] = false;
    }

    // 4. Collect the final chain order
    // We want to ensure the entry block (block 0) comes first.
    // Find the chain containing the entry block and put it first.
    let entry_chain_idx = block_to_chain.get(&BlockId::new(0)).copied().unwrap_or(0);

    let mut final_order = Vec::new();

    // Add the entry chain first
    if chain_alive[entry_chain_idx] {
        final_order.extend(&chains[entry_chain_idx].blocks);
        chain_alive[entry_chain_idx] = false;
    }

    // Add remaining chains
    // For chains not connected to the entry, we order by the weight of their
    // highest-weight incoming edge (hotter chains first), breaking ties by
    // block ID order.
    let mut remaining: Vec<(u64, BlockId, usize)> = Vec::new();
    for (i, alive) in chain_alive.iter().enumerate() {
        if *alive {
            // Use the max edge weight involving this chain as priority
            let max_weight = chains[i]
                .blocks
                .iter()
                .map(|&b| profile.block_weight(b))
                .max()
                .unwrap_or(0);
            remaining.push((max_weight, chains[i].head(), i));
        }
    }
    remaining.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.as_u32().cmp(&b.1.as_u32())));

    for (_, _, idx) in remaining {
        final_order.extend(&chains[idx].blocks);
    }

    // 5. Estimate improvement
    // Count how many of the original edges are now fall-through
    let total_edge_weight = profile.total_edge_weight() as f64;
    let mut fall_through_weight: u64 = 0;
    for i in 0..final_order.len().saturating_sub(1) {
        let from = final_order[i];
        let to = final_order[i + 1];
        fall_through_weight += profile.edge_weight(from, to);
    }

    let estimated_improvement = if total_edge_weight > 0.0 {
        (fall_through_weight as f64 / total_edge_weight) * 100.0
    } else {
        0.0
    };

    LayoutResult {
        block_order: final_order,
        estimated_improvement,
    }
}

// ── Synthetic Profile ───────────────────────────────────────────────────

/// Create a synthetic edge profile from static heuristics.
///
/// When profile-guided optimization data is not available, this function
/// uses static heuristics to estimate edge frequencies:
/// - **Loop back edges** (a successor that dominates this block) are hot.
/// - **Early exit edges** (to blocks that return) are cold.
/// - **Fall-through edges** (the first successor) get moderate weight.
/// - **Other edges** get baseline weight.
///
/// The function also uses loop nesting depth to amplify weights for
/// blocks deeper in loop nests.
pub fn synthesize_profile(func: &MirFunction) -> EdgeProfile {
    let mut profile = EdgeProfile::new();

    // Build block map for quick lookup
    let _block_map: HashMap<BlockId, &MirBlock> =
        func.blocks.iter().map(|b| (b.id, b)).collect();

    // Compute simple loop detection: a back edge is an edge from block B
    // to block A where A appears before B in the block order AND A
    // is a successor of B. We use a simple heuristic: if a successor
    // of a block has a lower BlockId, it's likely a loop back edge.
    // More sophisticated: compute dominators. For now, we use the simpler
    // heuristic that a successor with a smaller block ID is a back edge.

    // Identify return blocks (blocks that end with Ret)
    let mut return_blocks: HashMap<BlockId, bool> = HashMap::new();
    for block in &func.blocks {
        let is_return = block.insts.iter().any(|i| matches!(i, MirInst::Ret { .. }));
        return_blocks.insert(block.id, is_return);
    }

    // Identify likely loop headers: blocks that are targets of back edges
    let mut loop_headers: HashMap<BlockId, u32> = HashMap::new();
    for block in &func.blocks {
        for &succ in &block.succs {
            if succ.as_u32() <= block.id.as_u32() {
                // This is a back edge (succ has lower or equal ID → likely loop)
                *loop_headers.entry(succ).or_insert(0) += 1;
            }
        }
    }

    // Compute loop nesting depth (simplified)
    // A block is at depth = number of loop headers that dominate it.
    // Simplified: if block ID >= header ID, it might be in that loop.
    let mut loop_depth: HashMap<BlockId, u32> = HashMap::new();
    for block in &func.blocks {
        let mut depth = 0u32;
        for (&header, _) in &loop_headers {
            if header.as_u32() <= block.id.as_u32() {
                depth += 1;
            }
        }
        // Don't count the block itself as being in its own loop more than once
        if loop_headers.contains_key(&block.id) {
            depth = depth.saturating_sub(1);
        }
        loop_depth.insert(block.id, depth);
    }

    // Assign edge weights based on heuristics
    let base_weight: u64 = 10;
    let loop_back_weight: u64 = 1000;
    let early_exit_weight: u64 = 1;
    let hot_successor_weight: u64 = 100;
    let cold_successor_weight: u64 = 5;

    for block in &func.blocks {
        let depth = loop_depth.get(&block.id).copied().unwrap_or(0);
        let depth_multiplier = 10u64.saturating_pow(depth);

        if block.succs.is_empty() {
            continue;
        }

        for (i, &succ) in block.succs.iter().enumerate() {
            let weight = if succ.as_u32() <= block.id.as_u32() {
                // Loop back edge — very hot
                loop_back_weight * depth_multiplier
            } else if return_blocks.get(&succ).copied().unwrap_or(false)
                && block.succs.len() > 1
            {
                // Early exit to a return block — cold
                early_exit_weight
            } else if i == 0 {
                // First successor (likely fall-through or taken branch) — moderate
                hot_successor_weight * depth_multiplier
            } else {
                // Other successors — less hot
                cold_successor_weight * depth_multiplier
            };

            profile.record_edge(block.id, succ, weight.max(base_weight));
        }

        // If no successors were recorded (e.g., unreachable), give the block
        // a base weight
        if block.succs.is_empty() {
            *profile.block_counts.entry(block.id).or_insert(0) += base_weight;
        }
    }

    profile
}

// ── Layout Application ──────────────────────────────────────────────────

/// Apply a block layout to reorder blocks in a MirFunction.
///
/// This reorders the `blocks` vector in the MirFunction according to the
/// given layout result. Block IDs remain the same; only their order in
/// the function changes.
///
/// Returns `true` if the block order was changed, `false` if it was
/// already in the optimal order.
pub fn layout_function(func: &mut MirFunction, result: &LayoutResult) -> bool {
    let original_order: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();

    // Check if already in the right order
    if original_order == result.block_order {
        return false;
    }

    // Build a map from BlockId to the block data
    let mut block_map: HashMap<BlockId, MirBlock> = HashMap::new();
    for block in func.blocks.drain(..) {
        block_map.insert(block.id, block);
    }

    // Re-insert blocks in the new order
    for &block_id in &result.block_order {
        if let Some(block) = block_map.remove(&block_id) {
            func.blocks.push(block);
        }
    }

    // Any remaining blocks (shouldn't happen if layout is correct)
    for (_, block) in block_map {
        func.blocks.push(block);
    }

    true
}

// ── Convenience: Layout a Function with Synthetic Profile ───────────────

/// Layout a function using a synthetic profile.
///
/// This is a convenience function that creates a synthetic profile and
/// applies the Pettis-Hansen algorithm to the given function.
pub fn layout_with_synthetic_profile(func: &mut MirFunction) -> LayoutResult {
    let profile = synthesize_profile(func);
    let result = pettis_hansen_layout(func, &profile);
    layout_function(func, &result);
    result
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_function() -> MirFunction {
        let mut func = MirFunction::new("test");
        let bb0 = func.new_block(); // BlockId(0) — entry
        let bb1 = func.new_block(); // BlockId(1) — loop body
        let bb2 = func.new_block(); // BlockId(2) — exit

        // bb0 → bb1
        func.blocks[0].succs = vec![bb1];
        func.blocks[1].preds = vec![bb0];

        // bb1 → bb1 (loop back), bb1 → bb2 (exit)
        func.blocks[1].succs = vec![bb1, bb2];
        func.blocks[1].preds = vec![bb0, bb1];
        func.blocks[2].preds = vec![bb1];

        func
    }

    #[test]
    fn edge_profile_record_and_query() {
        let mut profile = EdgeProfile::new();
        let b0 = BlockId::new(0);
        let b1 = BlockId::new(1);

        profile.record_edge(b0, b1, 100);
        assert_eq!(profile.edge_weight(b0, b1), 100);
        assert_eq!(profile.edge_weight(b1, b0), 0);
        assert!(profile.block_weight(b0) > 0);
        assert!(profile.block_weight(b1) > 0);
    }

    #[test]
    fn pettis_hansen_simple() {
        let func = make_simple_function();
        let mut profile = EdgeProfile::new();

        let b0 = BlockId::new(0);
        let b1 = BlockId::new(1);
        let b2 = BlockId::new(2);

        // bb0→bb1: 100, bb1→bb1: 1000 (loop back), bb1→bb2: 50
        profile.record_edge(b0, b1, 100);
        profile.record_edge(b1, b1, 1000);
        profile.record_edge(b1, b2, 50);

        let result = pettis_hansen_layout(&func, &profile);

        // Entry block should come first
        assert_eq!(result.block_order[0], b0);
        // All blocks should be present
        assert_eq!(result.block_order.len(), 3);
        assert!(result.estimated_improvement >= 0.0);
    }

    #[test]
    fn synthesize_profile_loop_heuristic() {
        let func = make_simple_function();
        let profile = synthesize_profile(&func);

        let b0 = BlockId::new(0);
        let b1 = BlockId::new(1);
        let b2 = BlockId::new(2);

        // Loop back edge (b1→b1) should be hot
        let back_weight = profile.edge_weight(b1, b1);
        // Exit edge (b1→b2) should be relatively cold
        let exit_weight = profile.edge_weight(b1, b2);
        // Forward edge (b0→b1) should be moderate
        let forward_weight = profile.edge_weight(b0, b1);

        assert!(back_weight > exit_weight, "Loop back edge should be hotter than exit edge");
        assert!(forward_weight > exit_weight, "Forward edge should be hotter than exit edge");
    }

    #[test]
    fn layout_function_reorders_blocks() {
        let mut func = make_simple_function();

        // Create a layout that puts blocks in reverse order
        let result = LayoutResult {
            block_order: vec![BlockId::new(0), BlockId::new(2), BlockId::new(1)],
            estimated_improvement: 50.0,
        };

        let changed = layout_function(&mut func, &result);
        assert!(changed, "Should have reordered blocks");

        // Verify order
        assert_eq!(func.blocks[0].id, BlockId::new(0));
        assert_eq!(func.blocks[1].id, BlockId::new(2));
        assert_eq!(func.blocks[2].id, BlockId::new(1));
    }

    #[test]
    fn layout_function_no_change_if_already_optimal() {
        let mut func = make_simple_function();
        let original_order: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();

        let result = LayoutResult {
            block_order: original_order,
            estimated_improvement: 0.0,
        };

        let changed = layout_function(&mut func, &result);
        assert!(!changed, "Should not have changed already optimal order");
    }

    #[test]
    fn layout_with_synthetic_profile_works() {
        let mut func = make_simple_function();
        let result = layout_with_synthetic_profile(&mut func);

        // Entry block should still be first
        assert_eq!(func.blocks[0].id, BlockId::new(0));
        assert_eq!(func.blocks.len(), 3);
        assert!(result.estimated_improvement >= 0.0);
    }

    #[test]
    fn empty_function_layout() {
        let func = MirFunction::new("empty");
        let profile = EdgeProfile::new();
        let result = pettis_hansen_layout(&func, &profile);

        assert!(result.block_order.is_empty());
        assert_eq!(result.estimated_improvement, 0.0);
    }

    #[test]
    fn single_block_layout() {
        let mut func = MirFunction::new("single");
        let _bb = func.new_block();

        let profile = EdgeProfile::new();
        let result = pettis_hansen_layout(&func, &profile);

        assert_eq!(result.block_order.len(), 1);
    }

    #[test]
    fn diamond_control_flow() {
        let mut func = MirFunction::new("diamond");
        let bb0 = func.new_block(); // entry
        let bb1 = func.new_block(); // left
        let bb2 = func.new_block(); // right
        let bb3 = func.new_block(); // merge

        func.blocks[0].succs = vec![bb1, bb2];
        func.blocks[1].preds = vec![bb0];
        func.blocks[2].preds = vec![bb0];
        func.blocks[1].succs = vec![bb3];
        func.blocks[2].succs = vec![bb3];
        func.blocks[3].preds = vec![bb1, bb2];

        let mut profile = EdgeProfile::new();
        profile.record_edge(bb0, bb1, 80); // mostly take left
        profile.record_edge(bb0, bb2, 20); // rarely take right
        profile.record_edge(bb1, bb3, 80);
        profile.record_edge(bb2, bb3, 20);

        let result = pettis_hansen_layout(&func, &profile);

        assert_eq!(result.block_order[0], bb0);
        // The hotter path should be laid out for fall-through
        // bb0→bb1 should be fall-through (bb1 immediately after bb0)
        let pos_bb0 = result.block_order.iter().position(|&b| b == bb0).unwrap();
        let pos_bb1 = result.block_order.iter().position(|&b| b == bb1).unwrap();
        assert_eq!(pos_bb1, pos_bb0 + 1, "Hot path should be fall-through");
    }
}
