Hummingbot MCP Optional Setup

This is optional and only for interactive checks via Codex CLI. The bot itself uses direct REST calls to the Hummingbot API.

Prereqs
- Hummingbot API server running (default http://localhost:8000).
- Docker installed and running.

Codex CLI MCP config
Add to `~/.codex/config.toml`:

```
[mcp_servers.hummingbot]
command = "docker"
args = [
  "run",
  "--rm",
  "-i",
  "-e", "HUMMINGBOT_API_URL=http://host.docker.internal:8000",
  "-v", "hummingbot_mcp:/root/.hummingbot_mcp",
  "hummingbot/hummingbot-mcp:latest"
]
```

Then in Codex CLI:
- `/mcp` to see servers
- `/mcp list` to verify the hummingbot server

Notes
- If your Hummingbot API URL is different, update `HUMMINGBOT_API_URL`.
- MCP is for assistant-driven checks; the autonomous bot runs independently via REST.
