//! Core agent logic.
//!
//! The agent orchestrates:
//! - Message routing from channels
//! - Job scheduling and execution
//! - Tool invocation with safety
//! - Self-repair for stuck jobs
//! - Proactive heartbeat execution

mod agent_loop;
mod heartbeat;
mod router;
mod scheduler;
mod self_repair;
mod worker;

pub use agent_loop::Agent;
pub use heartbeat::{HeartbeatConfig, HeartbeatResult, HeartbeatRunner, spawn_heartbeat};
pub use router::{MessageIntent, Router};
pub use scheduler::Scheduler;
pub use self_repair::{RepairResult, RepairTask, SelfRepair, StuckJob};
pub use worker::Worker;
