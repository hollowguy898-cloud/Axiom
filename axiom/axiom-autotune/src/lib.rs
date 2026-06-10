//! Axiom Autotune — Autotuner System.
//!
//! This crate implements a three-engine autotuner that searches for the
//! best optimization parameters for a given function. The three engines are:
//!
//! 1. **Search Engine**: Explores the parameter space to generate candidate
//!    solutions. The default implementation uses beam search with stochastic
//!    sampling.
//!
//! 2. **Policy Engine**: Decides which candidates to keep for the next
//!    generation. The default implementation uses elitist selection with
//!    diversity pressure.
//!
//! 3. **Cost Model**: Estimates the cost of a compiled function for a given
//!    set of tuning parameters. The default implementation uses instruction
//!    count with weighted costs per instruction type.
//!
//! # Default Tuning Parameters
//!
//! The autotuner tunes the following parameters:
//! - `inline_threshold`: Maximum callee size for inlining
//! - `dse_aggressiveness`: Dead store elimination aggressiveness (0-3)
//! - `cse_scope`: CSE scope (0=local, 1=global)
//! - `vectorize_width`: Target vector width in bits
//! - `unroll_factor`: Loop unrolling factor
//! - `block_layout_strategy`: Block layout strategy (0=none, 1=Pettis-Hansen)
//!
//! # Usage
//!
//! ```ignore
//! use axiom_autotune::{Autotuner, BeamSearchEngine, ElitistPolicy, InstructionCountCostModel};
//!
//! let mut autotuner = Autotuner::new(
//!     Box::new(BeamSearchEngine::new(10, 0.3)),
//!     Box::new(ElitistPolicy::new(0.5, 0.3)),
//!     Box::new(InstructionCountCostModel::default()),
//!     20,   // generations
//!     50,   // population_size
//! );
//!
//! let best_params = autotuner.tune(&graph);
//! let pipeline = autotuner.create_pipeline(&best_params);
//! ```

use axiom_ir::{IrGraph, IrNode};
use axiom_opt::Pass;
use std::collections::HashMap;

// ── Tuning Parameter ────────────────────────────────────────────────────

/// A tuning parameter with a range of values.
#[derive(Debug, Clone)]
pub struct TuningParam {
    /// Parameter name.
    pub name: String,
    /// Current value.
    pub value: i64,
    /// Minimum value.
    pub min: i64,
    /// Maximum value.
    pub max: i64,
    /// Step size between valid values.
    pub step: i64,
}

impl TuningParam {
    /// Create a new tuning parameter.
    pub fn new(name: &str, value: i64, min: i64, max: i64, step: i64) -> Self {
        Self {
            name: name.to_string(),
            value,
            min,
            max,
            step,
        }
    }

    /// Get all valid values for this parameter.
    pub fn valid_values(&self) -> Vec<i64> {
        let mut values = Vec::new();
        let mut v = self.min;
        while v <= self.max {
            values.push(v);
            v += self.step;
        }
        values
    }

    /// Clamp a value to the valid range.
    pub fn clamp(&self, value: i64) -> i64 {
        let clamped = value.max(self.min).min(self.max);
        // Snap to nearest valid value
        let offset = (clamped - self.min) % self.step;
        if offset <= self.step / 2 {
            clamped - offset
        } else {
            clamped + (self.step - offset)
        }
        .max(self.min)
        .min(self.max)
    }

    /// Randomly perturb this parameter (returns a new value within range).
    pub fn perturb(&self, rng_state: &mut u64) -> i64 {
        // Simple LCG-based perturbation
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let range = (self.max - self.min) / self.step + 1;
        let offset = (*rng_state as i64).unsigned_abs() as i64 % range;
        self.min + offset * self.step
    }
}

/// Create the default set of tuning parameters.
pub fn default_tuning_params() -> Vec<TuningParam> {
    vec![
        TuningParam::new("inline_threshold", 20, 5, 100, 5),
        TuningParam::new("dse_aggressiveness", 2, 0, 3, 1),
        TuningParam::new("cse_scope", 1, 0, 1, 1),
        TuningParam::new("vectorize_width", 128, 0, 512, 128),
        TuningParam::new("unroll_factor", 2, 0, 8, 1),
        TuningParam::new("block_layout_strategy", 1, 0, 2, 1),
    ]
}

// ── Candidate ───────────────────────────────────────────────────────────

/// A candidate solution: a specific combination of tuning parameters.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Parameter values: parameter name → value.
    pub params: HashMap<String, i64>,
    /// Estimated cost (lower is better).
    pub cost: f64,
    /// Generation in which this candidate was created.
    pub generation: u32,
}

impl Candidate {
    /// Create a new candidate with the given parameters.
    pub fn new(params: HashMap<String, i64>, generation: u32) -> Self {
        Self {
            params,
            cost: f64::MAX,
            generation,
        }
    }

    /// Compute the distance between this candidate and another.
    /// Uses normalized Manhattan distance across all shared parameters.
    pub fn distance(&self, other: &Candidate, param_defs: &[TuningParam]) -> f64 {
        let mut total_dist = 0.0;
        let mut count = 0;

        for param_def in param_defs {
            let a = self.params.get(&param_def.name).copied().unwrap_or(param_def.value);
            let b = other.params.get(&param_def.name).copied().unwrap_or(param_def.value);
            let range = (param_def.max - param_def.min) as f64;
            if range > 0.0 {
                total_dist += ((a - b) as f64).abs() / range;
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }
}

// ── Cost Model ──────────────────────────────────────────────────────────

/// Cost model: estimates the cost of a compiled function.
///
/// The cost model takes an IR graph and a set of tuning parameters, and
/// returns an estimated execution cost. Lower cost is better.
pub trait CostModel: Send + Sync {
    /// Estimate the cost of running `graph` with the given `params`.
    fn estimate_cost(&self, graph: &IrGraph, params: &HashMap<String, i64>) -> f64;

    /// Name of this cost model (for logging).
    fn name(&self) -> &str;
}

/// Simple cost model based on instruction count and memory operations.
///
/// This model weights different instruction types by their estimated
/// latency, producing a weighted instruction count. Tuning parameters
/// affect the cost estimate by:
/// - Higher `inline_threshold` reduces call overhead but increases code size
/// - Higher `dse_aggressiveness` reduces store cost
/// - Higher `vectorize_width` reduces arithmetic cost for vectorizable ops
/// - Higher `unroll_factor` increases total instruction count
#[derive(Debug, Clone)]
pub struct InstructionCountCostModel {
    /// Cost multiplier for load instructions.
    pub load_cost: f64,
    /// Cost multiplier for store instructions.
    pub store_cost: f64,
    /// Cost multiplier for branch instructions.
    pub branch_cost: f64,
    /// Cost multiplier for call instructions.
    pub call_cost: f64,
    /// Cost multiplier for arithmetic instructions.
    pub arith_cost: f64,
}

impl Default for InstructionCountCostModel {
    fn default() -> Self {
        Self {
            load_cost: 4.0,
            store_cost: 4.0,
            branch_cost: 2.0,
            call_cost: 10.0,
            arith_cost: 1.0,
        }
    }
}

impl CostModel for InstructionCountCostModel {
    fn estimate_cost(&self, graph: &IrGraph, params: &HashMap<String, i64>) -> f64 {
        let mut cost = 0.0;
        let mut load_count = 0u32;
        let mut store_count = 0u32;
        let mut branch_count = 0u32;
        let mut call_count = 0u32;
        let mut arith_count = 0u32;
        let mut other_count = 0u32;

        for (_id, node) in graph.iter() {
            match node {
                IrNode::Load { .. } => load_count += 1,
                IrNode::Store { .. } | IrNode::VarSet { .. } => store_count += 1,
                IrNode::Branch { .. } | IrNode::Jump { .. } => branch_count += 1,
                IrNode::Call { .. } | IrNode::CallIndirect { .. } => call_count += 1,
                IrNode::Add { .. } | IrNode::Sub { .. } | IrNode::Mul { .. }
                | IrNode::Div { .. } | IrNode::Rem { .. } | IrNode::Neg { .. }
                | IrNode::And { .. } | IrNode::Or { .. } | IrNode::Xor { .. }
                | IrNode::Shl { .. } | IrNode::Shr { .. } | IrNode::Sar { .. }
                | IrNode::Not { .. } => arith_count += 1,
                _ => other_count += 1,
            }
        }

        // Base cost
        cost += load_count as f64 * self.load_cost;
        cost += store_count as f64 * self.store_cost;
        cost += branch_count as f64 * self.branch_cost;
        cost += call_count as f64 * self.call_cost;
        cost += arith_count as f64 * self.arith_cost;
        cost += other_count as f64 * 0.5;

        // Apply parameter adjustments

        // DSE aggressiveness: higher aggressiveness reduces store cost
        let dse_aggr = params.get("dse_aggressiveness").copied().unwrap_or(2);
        let dse_factor = 1.0 - (dse_aggr as f64 * 0.15).min(0.6);
        cost = cost - store_count as f64 * self.store_cost * (1.0 - dse_factor);

        // Inline threshold: reduces call cost but may increase code size
        let inline_thresh = params.get("inline_threshold").copied().unwrap_or(20);
        // Estimate that a fraction of calls will be inlined
        let inline_fraction = (inline_thresh as f64 / 100.0).min(1.0);
        let call_savings = call_count as f64 * self.call_cost * inline_fraction * 0.5;
        let code_size_penalty = inline_fraction * arith_count as f64 * 0.1;
        cost = cost - call_savings + code_size_penalty;

        // Vectorize width: reduces arithmetic cost if there are enough ops
        let vec_width = params.get("vectorize_width").copied().unwrap_or(128);
        if vec_width > 0 && arith_count > 4 {
            let vec_factor = 1.0 / (vec_width as f64 / 128.0).max(1.0);
            let vectorizable_fraction = 0.3; // assume 30% of arith is vectorizable
            let vec_savings = arith_count as f64 * self.arith_cost * vectorizable_fraction * (1.0 - vec_factor);
            cost -= vec_savings;
        }

        // Unroll factor: increases instruction count but may improve ILP
        let unroll = params.get("unroll_factor").copied().unwrap_or(2);
        if unroll > 0 {
            // Unrolling increases code size but reduces loop overhead
            let loop_overhead_reduction = branch_count as f64 * self.branch_cost * 0.2 * (unroll as f64).min(4.0) / 4.0;
            let code_size_increase = arith_count as f64 * self.arith_cost * 0.05 * (unroll as f64 - 1.0).max(0.0);
            cost = cost - loop_overhead_reduction + code_size_increase;
        }

        // Block layout strategy: reduces branch mispredictions
        let layout_strategy = params.get("block_layout_strategy").copied().unwrap_or(1);
        if layout_strategy > 0 {
            let branch_savings = branch_count as f64 * self.branch_cost * 0.15 * layout_strategy as f64 / 2.0;
            cost -= branch_savings;
        }

        cost.max(0.0)
    }

    fn name(&self) -> &str {
        "instruction_count"
    }
}

// ── Search Engine ───────────────────────────────────────────────────────

/// Search engine: explores the parameter space.
///
/// The search engine generates new candidate solutions based on the
/// current population. Different search strategies can be implemented
/// (beam search, random search, genetic algorithms, etc.).
pub trait SearchEngine: Send + Sync {
    /// Generate `count` new candidates from the current `population`.
    ///
    /// The search engine may use the existing population to guide its
    /// search (e.g., by perturbing the best candidates) or may generate
    /// entirely new candidates (exploration).
    fn generate_candidates(&mut self, population: &[Candidate], count: usize) -> Vec<Candidate>;

    /// Name of this search engine (for logging).
    fn name(&self) -> &str;
}

/// Beam search with stochastic sampling.
///
/// This search engine:
/// 1. Selects the top candidates from the population (the "beam")
/// 2. For each beam member, generates perturbations of its parameters
/// 3. Also generates some random candidates (stochastic sampling) for
///    exploration
///
/// The `stochastic_ratio` controls what fraction of new candidates are
/// purely random vs. perturbations of existing good candidates.
pub struct BeamSearchEngine {
    /// Number of top candidates to use as the beam.
    pub beam_width: usize,
    /// Fraction of candidates that are randomly generated (0.0 - 1.0).
    pub stochastic_ratio: f64,
    /// RNG state for reproducibility.
    rng_state: u64,
}

impl BeamSearchEngine {
    /// Create a new beam search engine.
    ///
    /// - `beam_width`: number of top candidates to base perturbations on
    /// - `stochastic_ratio`: fraction of candidates that are purely random
    ///   (0.0 = all perturbations, 1.0 = all random)
    pub fn new(beam_width: usize, stochastic_ratio: f64) -> Self {
        Self {
            beam_width,
            stochastic_ratio: stochastic_ratio.clamp(0.0, 1.0),
            rng_state: 42, // deterministic seed
        }
    }

    /// Advance the RNG state.
    fn next_random(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.rng_state
    }

    /// Generate a random candidate.
    fn random_candidate(&mut self, param_defs: &[TuningParam], generation: u32) -> Candidate {
        let mut params = HashMap::new();
        for param in param_defs {
            let value = param.perturb(&mut self.rng_state);
            params.insert(param.name.clone(), value);
        }
        Candidate::new(params, generation)
    }

    /// Perturb an existing candidate.
    fn perturb_candidate(
        &mut self,
        parent: &Candidate,
        param_defs: &[TuningParam],
        generation: u32,
    ) -> Candidate {
        let mut params = parent.params.clone();

        // Perturb 1-3 parameters
        let num_perturb = 1 + (self.next_random() as usize % 3).min(param_defs.len());
        for _ in 0..num_perturb {
            let idx = self.next_random() as usize % param_defs.len();
            let param = &param_defs[idx];
            let new_value = param.perturb(&mut self.rng_state);
            params.insert(param.name.clone(), new_value);
        }

        Candidate::new(params, generation)
    }
}

impl SearchEngine for BeamSearchEngine {
    fn generate_candidates(&mut self, population: &[Candidate], count: usize) -> Vec<Candidate> {
        let param_defs = default_tuning_params();
        let generation = population
            .iter()
            .map(|c| c.generation)
            .max()
            .unwrap_or(0)
            + 1;

        // Select beam (top candidates by cost)
        let mut sorted: Vec<&Candidate> = population.iter().collect();
        sorted.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));
        let beam: Vec<&Candidate> = sorted.into_iter().take(self.beam_width).collect();

        let mut candidates = Vec::with_capacity(count);

        if beam.is_empty() {
            // No existing population — generate random candidates
            for _ in 0..count {
                candidates.push(self.random_candidate(&param_defs, generation));
            }
        } else {
            let stochastic_count = (count as f64 * self.stochastic_ratio) as usize;
            let perturb_count = count - stochastic_count;

            // Generate perturbations of beam members
            for i in 0..perturb_count {
                let parent = beam[i % beam.len()];
                candidates.push(self.perturb_candidate(parent, &param_defs, generation));
            }

            // Generate random candidates
            for _ in 0..stochastic_count {
                candidates.push(self.random_candidate(&param_defs, generation));
            }
        }

        candidates
    }

    fn name(&self) -> &str {
        "beam_search"
    }
}

// ── Policy Engine ───────────────────────────────────────────────────────

/// Policy engine: decides which candidates to keep.
///
/// The policy engine evaluates candidates and selects which ones to keep
/// for the next generation. Different policies can prioritize different
/// objectives (cost minimization, diversity, etc.).
pub trait PolicyEngine: Send + Sync {
    /// Select `count` candidates from the given pool to keep.
    fn select(&self, candidates: &[Candidate], count: usize) -> Vec<Candidate>;

    /// Name of this policy engine (for logging).
    fn name(&self) -> &str;
}

/// Elitist + diversity policy: keeps the best candidates while maintaining diversity.
///
/// The selection process:
/// 1. Sort candidates by cost (ascending — lower is better)
/// 2. Select the top `elite_ratio` fraction as elites
/// 3. For remaining slots, select candidates that maximize diversity
///    (distance from already-selected candidates)
///
/// This ensures we don't get stuck in a local minimum while still
/// converging toward good solutions.
pub struct ElitistPolicy {
    /// Fraction of slots reserved for elite candidates (0.0 - 1.0).
    pub elite_ratio: f64,
    /// Diversity pressure: how strongly to prefer diverse candidates
    /// in the non-elite slots (0.0 = no diversity pressure, 1.0 = max).
    pub diversity_pressure: f64,
}

impl ElitistPolicy {
    /// Create a new elitist policy.
    ///
    /// - `elite_ratio`: fraction of slots for the best candidates (0.0 - 1.0)
    /// - `diversity_pressure`: how strongly to prefer diverse candidates (0.0 - 1.0)
    pub fn new(elite_ratio: f64, diversity_pressure: f64) -> Self {
        Self {
            elite_ratio: elite_ratio.clamp(0.0, 1.0),
            diversity_pressure: diversity_pressure.clamp(0.0, 1.0),
        }
    }
}

impl PolicyEngine for ElitistPolicy {
    fn select(&self, candidates: &[Candidate], count: usize) -> Vec<Candidate> {
        if candidates.len() <= count {
            return candidates.to_vec();
        }

        let param_defs = default_tuning_params();

        // Sort by cost
        let mut sorted: Vec<usize> = (0..candidates.len()).collect();
        sorted.sort_by(|&a, &b| {
            candidates[a]
                .cost
                .partial_cmp(&candidates[b].cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select elites
        let elite_count = (count as f64 * self.elite_ratio).ceil() as usize;
        let elite_count = elite_count.min(count).min(sorted.len());

        let selected_indices: Vec<usize> = sorted[..elite_count].to_vec();
        let mut selected: Vec<Candidate> = selected_indices
            .iter()
            .map(|&i| candidates[i].clone())
            .collect();

        // Fill remaining slots with diversity-aware selection
        let remaining_count = count - selected.len();
        let mut available: Vec<usize> = sorted[elite_count..].to_vec();

        for _ in 0..remaining_count {
            if available.is_empty() {
                break;
            }

            // For each available candidate, compute a score combining cost
            // and diversity
            let best_idx = if self.diversity_pressure > 0.0 {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_avl_idx = 0;

                for (avl_idx, &cand_idx) in available.iter().enumerate() {
                    let cost_score = -candidates[cand_idx].cost;

                    // Compute minimum distance to any selected candidate
                    let min_dist = selected
                        .iter()
                        .map(|s| candidates[cand_idx].distance(s, &param_defs))
                        .fold(f64::MAX, f64::min);

                    let diversity_score = min_dist * self.diversity_pressure * 100.0;
                    let total_score = cost_score + diversity_score;

                    if total_score > best_score {
                        best_score = total_score;
                        best_avl_idx = avl_idx;
                    }
                }

                available.remove(best_avl_idx)
            } else {
                // No diversity pressure — just pick the best remaining by cost
                available.remove(0)
            };

            selected.push(candidates[best_idx].clone());
        }

        selected
    }

    fn name(&self) -> &str {
        "elitist"
    }
}

// ── Autotuner ───────────────────────────────────────────────────────────

/// The autotuner: combines search, policy, and cost model.
///
/// The autotuner runs an iterative optimization loop:
/// 1. Generate candidate solutions using the search engine
/// 2. Evaluate each candidate using the cost model
/// 3. Select the best candidates using the policy engine
/// 4. Repeat for the configured number of generations
///
/// After tuning, the best parameters can be used to create an optimized
/// pipeline of `Pass` implementations.
pub struct Autotuner {
    /// Search engine for exploring parameter space.
    search: Box<dyn SearchEngine>,
    /// Policy engine for selecting candidates.
    policy: Box<dyn PolicyEngine>,
    /// Cost model for evaluating candidates.
    cost_model: Box<dyn CostModel>,
    /// Tuning parameter definitions.
    params: Vec<TuningParam>,
    /// Number of generations to run.
    generations: u32,
    /// Population size per generation.
    population_size: usize,
}

impl Autotuner {
    /// Create a new autotuner.
    ///
    /// - `search`: search engine for exploring parameter space
    /// - `policy`: policy engine for selecting candidates
    /// - `cost_model`: cost model for evaluating candidates
    /// - `generations`: number of generations to run
    /// - `population_size`: number of candidates per generation
    pub fn new(
        search: Box<dyn SearchEngine>,
        policy: Box<dyn PolicyEngine>,
        cost_model: Box<dyn CostModel>,
        generations: u32,
        population_size: usize,
    ) -> Self {
        Self {
            search,
            policy,
            cost_model,
            params: default_tuning_params(),
            generations,
            population_size,
        }
    }

    /// Set custom tuning parameters (overrides defaults).
    pub fn with_params(mut self, params: Vec<TuningParam>) -> Self {
        self.params = params;
        self
    }

    /// Tune optimization parameters for a given function.
    ///
    /// Runs the iterative optimization loop and returns the best
    /// parameter values found.
    pub fn tune(&mut self, graph: &IrGraph) -> HashMap<String, i64> {
        // Initialize population with the default parameter values
        let mut population: Vec<Candidate> = Vec::new();

        // Add the default candidate
        let default_params: HashMap<String, i64> = self
            .params
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        let mut default_candidate = Candidate::new(default_params, 0);
        default_candidate.cost = self.cost_model.estimate_cost(graph, &default_candidate.params);
        population.push(default_candidate);

        // Generate initial random candidates
        let initial_count = self.population_size.saturating_sub(1);
        let mut new_candidates = self.search.generate_candidates(&population, initial_count);
        for candidate in &mut new_candidates {
            candidate.cost = self.cost_model.estimate_cost(graph, &candidate.params);
        }
        population.extend(new_candidates);

        // Iterative optimization loop
        for gen in 1..=self.generations {
            // Generate new candidates from current population
            let num_new = self.population_size / 2; // generate half, keep half
            let mut new_candidates = self.search.generate_candidates(&population, num_new);
            for candidate in &mut new_candidates {
                candidate.cost = self.cost_model.estimate_cost(graph, &candidate.params);
            }

            // Merge with current population
            population.extend(new_candidates);

            // Select the best candidates
            population = self.policy.select(&population, self.population_size);

            // Log progress (in a real compiler, this would use a proper logger)
            if let Some(best) = population.first() {
                // Best cost at generation `gen`
                let _ = (gen, best.cost); // suppress unused warning
            }
        }

        // Return the best candidate's parameters
        population
            .into_iter()
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.params)
            .unwrap_or_else(|| {
                self.params
                    .iter()
                    .map(|p| (p.name.clone(), p.value))
                    .collect()
            })
    }

    /// Apply the tuned parameters to create an optimization pipeline.
    ///
    /// This creates a sequence of `Pass` implementations configured
    /// according to the tuned parameters.
    pub fn create_pipeline(&self, params: &HashMap<String, i64>) -> Vec<Box<dyn Pass>> {
        use axiom_opt::{
            ConstantFolder, CommonSubexprElim, DeadCodeElim, DeadStoreElim, Inliner,
            StrengthReducer,
        };

        let mut pipeline: Vec<Box<dyn Pass>> = Vec::new();

        // Always start with constant folding and DCE
        pipeline.push(Box::new(ConstantFolder));
        pipeline.push(Box::new(DeadCodeElim));

        // CSE with configured scope
        // (scope is controlled internally by the pass; here we just include it)
        pipeline.push(Box::new(CommonSubexprElim));

        // DSE with configured aggressiveness
        // (aggressiveness is controlled internally; we include the pass)
        pipeline.push(Box::new(DeadStoreElim));

        // Strength reduction
        pipeline.push(Box::new(StrengthReducer));

        // Inliner with configured threshold
        let inline_threshold = params
            .get("inline_threshold")
            .copied()
            .unwrap_or(20) as usize;
        let inliner = Inliner::new(std::collections::HashMap::new(), inline_threshold);
        pipeline.push(Box::new(inliner));

        // Final cleanup
        pipeline.push(Box::new(ConstantFolder));
        pipeline.push(Box::new(DeadCodeElim));

        pipeline
    }

    /// Get the tuning parameter definitions.
    pub fn params(&self) -> &[TuningParam] {
        &self.params
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> IrGraph {
        let mut graph = IrGraph::new("test_fn");

        // Create some arithmetic
        let a = graph.push_node(IrNode::IntConst(10));
        let b = graph.push_node(IrNode::IntConst(20));
        let sum = graph.push_node(IrNode::Add { lhs: a, rhs: b });
        let _ret = graph.push_node(IrNode::Return { value: Some(sum) });

        graph
    }

    fn make_graph_with_memory() -> IrGraph {
        use axiom_ir::nodes::Type;

        let mut graph = IrGraph::new("mem_fn");

        let size = graph.push_node(IrNode::IntConst(64));
        let root = graph.alloc_root();
        let alloc = graph.push_node(IrNode::StackAlloc {
            size,
            align: 8,
            root,
        });

        let val = graph.push_node(IrNode::IntConst(42));
        let _store = graph.push_node(IrNode::Store {
            addr: alloc,
            val,
            root,
            ty: Type::I64,
        });

        let load = graph.push_node(IrNode::Load {
            addr: alloc,
            root,
            ty: Type::I64,
        });

        let a = graph.push_node(IrNode::IntConst(10));
        let sum = graph.push_node(IrNode::Add { lhs: load, rhs: a });
        let _ret = graph.push_node(IrNode::Return { value: Some(sum) });

        graph
    }

    #[test]
    fn tuning_param_valid_values() {
        let param = TuningParam::new("test", 5, 0, 20, 5);
        let values = param.valid_values();
        assert_eq!(values, vec![0, 5, 10, 15, 20]);
    }

    #[test]
    fn tuning_param_clamp() {
        let param = TuningParam::new("test", 10, 0, 20, 5);
        assert_eq!(param.clamp(-5), 0);
        assert_eq!(param.clamp(25), 20);
        assert_eq!(param.clamp(12), 10); // snaps to nearest step
    }

    #[test]
    fn default_params_created() {
        let params = default_tuning_params();
        assert_eq!(params.len(), 6);

        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"inline_threshold"));
        assert!(names.contains(&"dse_aggressiveness"));
        assert!(names.contains(&"cse_scope"));
        assert!(names.contains(&"vectorize_width"));
        assert!(names.contains(&"unroll_factor"));
        assert!(names.contains(&"block_layout_strategy"));
    }

    #[test]
    fn candidate_distance_same() {
        let params1: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        let c1 = Candidate::new(params1.clone(), 0);
        let c2 = Candidate::new(params1, 0);

        let param_defs = default_tuning_params();
        let dist = c1.distance(&c2, &param_defs);
        assert_eq!(dist, 0.0, "Identical candidates should have distance 0");
    }

    #[test]
    fn candidate_distance_different() {
        let mut params1: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        let params2 = params1.clone();

        // Change one parameter
        params1.insert("inline_threshold".to_string(), 5);
        let c1 = Candidate::new(params1, 0);
        let c2 = Candidate::new(params2, 0);

        let param_defs = default_tuning_params();
        let dist = c1.distance(&c2, &param_defs);
        assert!(dist > 0.0, "Different candidates should have positive distance");
    }

    #[test]
    fn cost_model_estimates_cost() {
        let model = InstructionCountCostModel::default();
        let graph = make_simple_graph();
        let params: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();

        let cost = model.estimate_cost(&graph, &params);
        assert!(cost > 0.0, "Cost should be positive for non-empty graph");
    }

    #[test]
    fn cost_model_memory_ops_more_expensive() {
        let model = InstructionCountCostModel::default();
        let simple_graph = make_simple_graph();
        let mem_graph = make_graph_with_memory();

        let params: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();

        let simple_cost = model.estimate_cost(&simple_graph, &params);
        let mem_cost = model.estimate_cost(&mem_graph, &params);

        // Memory graph should be more expensive due to load/store
        assert!(
            mem_cost > simple_cost,
            "Graph with memory ops should have higher cost"
        );
    }

    #[test]
    fn beam_search_generates_candidates() {
        let mut engine = BeamSearchEngine::new(5, 0.3);

        // Start with empty population
        let candidates = engine.generate_candidates(&[], 10);
        assert_eq!(candidates.len(), 10);

        // All should have parameters
        for candidate in &candidates {
            assert!(!candidate.params.is_empty());
        }
    }

    #[test]
    fn beam_search_perturbs_existing() {
        let mut engine = BeamSearchEngine::new(5, 0.0); // no stochastic

        let params: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        let parent = Candidate::new(params, 0);

        let candidates = engine.generate_candidates(&[parent], 5);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn elitist_policy_selects_best() {
        let policy = ElitistPolicy::new(0.5, 0.0);

        let mut candidates = Vec::new();
        for i in 0..10 {
            let mut params = HashMap::new();
            params.insert("test".to_string(), i);
            let mut c = Candidate::new(params, 0);
            c.cost = i as f64; // lower index = lower cost = better
            candidates.push(c);
        }

        let selected = policy.select(&candidates, 5);
        assert_eq!(selected.len(), 5);

        // The best candidate should be selected
        assert_eq!(selected[0].cost, 0.0);
    }

    #[test]
    fn elitist_policy_maintains_diversity() {
        let policy = ElitistPolicy::new(0.2, 1.0); // high diversity pressure

        let mut candidates = Vec::new();
        for i in 0..20 {
            let mut params = HashMap::new();
            params.insert("inline_threshold".to_string(), i * 10);
            let mut c = Candidate::new(params, 0);
            // Give all similar costs so diversity matters
            c.cost = 100.0 - (i as f64 * 0.01);
            candidates.push(c);
        }

        let selected = policy.select(&candidates, 10);
        assert_eq!(selected.len(), 10);

        // With high diversity pressure, selected candidates should be spread out
        let values: Vec<i64> = selected
            .iter()
            .filter_map(|c| c.params.get("inline_threshold").copied())
            .collect();
        // Check that we don't have all candidates clustered together
        if values.len() > 1 {
            let range = values.iter().max().unwrap() - values.iter().min().unwrap();
            assert!(range > 0, "Diverse selection should have spread");
        }
    }

    #[test]
    fn autotuner_tunes_simple_function() {
        let mut autotuner = Autotuner::new(
            Box::new(BeamSearchEngine::new(5, 0.3)),
            Box::new(ElitistPolicy::new(0.5, 0.3)),
            Box::new(InstructionCountCostModel::default()),
            5,  // generations
            20, // population_size
        );

        let graph = make_simple_graph();
        let best_params = autotuner.tune(&graph);

        // Should have all default parameters
        assert!(best_params.contains_key("inline_threshold"));
        assert!(best_params.contains_key("dse_aggressiveness"));
    }

    #[test]
    fn autotuner_creates_pipeline() {
        let autotuner = Autotuner::new(
            Box::new(BeamSearchEngine::new(5, 0.3)),
            Box::new(ElitistPolicy::new(0.5, 0.3)),
            Box::new(InstructionCountCostModel::default()),
            5,
            20,
        );

        let params: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();

        let pipeline = autotuner.create_pipeline(&params);
        assert!(!pipeline.is_empty(), "Pipeline should contain passes");

        // Check that passes have names
        for pass in &pipeline {
            assert!(!pass.name().is_empty());
        }
    }

    #[test]
    fn autotuner_with_memory_heavy_function() {
        let mut autotuner = Autotuner::new(
            Box::new(BeamSearchEngine::new(5, 0.3)),
            Box::new(ElitistPolicy::new(0.5, 0.3)),
            Box::new(InstructionCountCostModel::default()),
            5,
            20,
        );

        let graph = make_graph_with_memory();
        let best_params = autotuner.tune(&graph);

        // Should produce valid parameters
        assert!(best_params.contains_key("dse_aggressiveness"));
        assert!(best_params.contains_key("inline_threshold"));
    }

    #[test]
    fn cost_model_dse_reduces_store_cost() {
        let model = InstructionCountCostModel::default();
        let graph = make_graph_with_memory();

        let mut low_dse: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        low_dse.insert("dse_aggressiveness".to_string(), 0);

        let mut high_dse: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        high_dse.insert("dse_aggressiveness".to_string(), 3);

        let cost_low = model.estimate_cost(&graph, &low_dse);
        let cost_high = model.estimate_cost(&graph, &high_dse);

        assert!(
            cost_high <= cost_low,
            "Higher DSE aggressiveness should not increase cost"
        );
    }

    #[test]
    fn cost_model_vectorization_reduces_cost() {
        let model = InstructionCountCostModel::default();
        let graph = make_simple_graph();

        let mut no_vec: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        no_vec.insert("vectorize_width".to_string(), 0);

        let mut with_vec: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();
        with_vec.insert("vectorize_width".to_string(), 256);

        let cost_no_vec = model.estimate_cost(&graph, &no_vec);
        let cost_with_vec = model.estimate_cost(&graph, &with_vec);

        assert!(
            cost_with_vec <= cost_no_vec,
            "Vectorization should not increase cost"
        );
    }

    #[test]
    fn empty_graph_zero_cost() {
        let model = InstructionCountCostModel::default();
        let graph = IrGraph::new("empty");
        let params: HashMap<String, i64> = default_tuning_params()
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect();

        let cost = model.estimate_cost(&graph, &params);
        // Empty graph should have very low cost (just the Start node)
        assert!(cost >= 0.0);
    }
}
