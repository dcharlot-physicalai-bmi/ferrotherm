//! The agent-facing tier for ferrotherm.
//!
//! One set of operations in [`api`], reached two ways: an HTTP server in [`http`] for programs and
//! browsers, and a Model Context Protocol server in [`mcp`] for language models. Both are std-only
//! and depend on nothing outside this workspace, so the binary a user runs against their own models
//! is one they can read end to end.

pub mod api;
pub mod http;
pub mod json;
pub mod mcp;
