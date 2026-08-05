//! Model Context Protocol server over stdio. Point an MCP client at this binary.
fn main() -> std::io::Result<()> {
    ferrotherm_serve::mcp::serve()
}
