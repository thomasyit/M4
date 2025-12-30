Autonomous trading bot (Sentiment + Risk Gate)

This bot adapts the Coincise blog sentiment analyzer into a trading workflow and applies the risk gate design from the Synternet -> Hummingbot Path A research.

What it does
- Pulls market data (Binance with CoinGecko fallback).
- Asks an LLM for a structured trade setup.
- Converts that into a TradeIntent.
- Applies funding guard, risk-to-stop sizing, and exposure caps.
- Executes via Hummingbot API or DRY_RUN receipts.

Key files
- bot/sentiment_analyzer.py: LLM-based sentiment and trade setup.
- bot/bridge.py: Converts sentiment output into TradeIntent.
- bot/risk_engine.py: Risk gate (funding guard, sizing, caps, quantization).
- bot/executor.py: Executes or records trades.

Quick start (dry run)
1) Copy .env.example to .env and fill in keys.
2) Install deps: pip install -r requirements.txt
3) Run: python -m bot.main --once
4) Dashboard: streamlit run dashboard/app.py

Notes
- DRY_RUN defaults to true.
- Hummingbot API paths are configurable via env (HB_API_PREFIX, HB_RULES_ENDPOINT, HB_ORDERS_ENDPOINT).
- Funding guard can be tightened for Hyperliquid by switching HB_CONNECTOR to hyperliquid_perpetual.
- ANALYSIS_MODE defaults to `dual` and requires agreement between ChatGPT and Synternet before trading. Use `llm` or `synternet` for single-source mode.
- `dual` or `synternet` mode requires `SYNTERNET_API_KEY` and `SYNTERNET_API_URL` in `.env`.
- `SYNTERNET_TIMEOUT_SEC` controls how long dual mode waits for Synternet before rejecting the intent (default 20s).
- `SYNTERNET_HTTP_TIMEOUT` controls the HTTP timeout for Synternet requests (default 20s).
- `SYNTERNET_SKIP_NANSEN=true` uses only the Syntoshi Thinking model and skips the Nansen whale call.
- Entry/stop/TP are computed deterministically from 1h ATR. Configure via `ATR_PERIOD`, `ATR_STOP_MULT`, `ATR_TP_MULT`.
- Regime gate: `REGIME_MODE=trend|range|any` with `REGIME_TREND_THRESHOLD` and `REGIME_VOL_MAX`. Optional 4h EMA confirmation via `REQUIRE_4H_CONFIRM`.
- Funding direction filter: `FUNDING_MAX_LONG_BPS`, `FUNDING_MAX_SHORT_BPS`.
- Risk kill switches: `MAX_DAILY_DD`, `MAX_LOSS_STREAK`, `ALLOW_MULTIPLE_POSITIONS`, `TRADE_STATE_FILE`.
- Execution guards: `MAX_SLIPPAGE_BPS`, `MAX_SPREAD_BPS`.
- Order status polling: `ORDER_MONITOR_ENABLE`, `HB_ORDER_STATUS_ENDPOINT` (used to reconcile TP fills).
- Receipts include `version` from `BOT_VERSION` (default `v2`) for logic tracking.
- Run summaries are written to `RUN_SUMMARY_DIR` even when no trade is executed.
- TP ladder: `ENABLE_TP_LADDER`, `TP_LADDER_PCTS`, `TP_LADDER_R_MULTS` (R-multiple of stop distance).
- Trailing stop: `ENABLE_TRAIL_STOP`, `TRAIL_STOP_ATR_MULT`, `TRAIL_POLL_SEC` (polls Binance price).
- Default schedule runs at 08:01 Asia/Singapore and every 4 hours (6 times/day). Use `--schedule interval` for fixed intervals.
- MCP alternative (optional): see docs/mcp_setup.md for how to wire the Hummingbot MCP server if you want interactive checks via Codex CLI. The bot still uses direct REST calls by default.

Notes
- Local config is stored in `.env` and is intentionally excluded from git.
- Runtime data, logs, and cached OHLCV live under `state/` and `binance_historical/` and are excluded from git.
- Use `.env.example` as a template for required environment variables.
