//! Axiom IR — Sea-of-Nodes intermediate representation.
//!
//! The core principle: every value is a node in a single, target-independent
//! graph. Ownership annotations enable correctness-guaranteed optimizations
//! that every target benefits from.

pub mod builder;
pub mod graph;
pub mod nodes;
pub mod operators;
pub mod vector;

pub use builder::IrBuilder;
pub use graph::IrGraph;
pub use nodes::{IrNode, NodeId, OwnershipRoot, VecBinOp, VecUnOp, VecReduceOp};
pub use operators::Op;
