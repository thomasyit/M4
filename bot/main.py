import argparse
import logging
import os
import time
from datetime import datetime, timedelta

import pytz

from bot.bridge import analyze_and_build_intent_with_summary
from bot.executor import execute_intent
from bot.order_monitor import reconcile_active_trade
from bot.run_logger import log_run_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def _configure_file_logging() -> None:
    log_path = os.getenv("BOT_LOG_FILE", "state/bot.log")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


def run_once(symbol: str) -> int:
    intent, summary = analyze_and_build_intent_with_summary(symbol)
    if not intent:
        logger.info(
            "Run summary: run_id=%s mode=%s chatgpt=%s synternet=%s decision=%s reason=%s",
            summary.get("run_id"),
        summary.get("mode"),
            summary.get("chatgpt_sentiment"),
            summary.get("synternet_sentiment"),
            summary.get("decision"),
            summary.get("reason"),
        )
        recon = reconcile_active_trade()
        if recon:
            logger.info("Reconcile status: %s", recon)
        log_run_summary(summary)
        logger.info("No trade intent produced")
        return 1

    ok, reason = execute_intent(intent)
    summary["decision"] = "executed" if ok else "rejected"
    summary["reason"] = reason
    logger.info(
        "Run summary: run_id=%s mode=%s chatgpt=%s synternet=%s decision=%s reason=%s",
        summary.get("run_id"),
        summary.get("mode"),
        summary.get("chatgpt_sentiment"),
        summary.get("synternet_sentiment"),
        summary.get("decision"),
        summary.get("reason"),
    )
    if ok:
        logger.info("Intent executed: %s", reason)
        recon = reconcile_active_trade()
        if recon:
            logger.info("Reconcile status: %s", recon)
        log_run_summary(summary)
        return 0

    logger.warning("Intent rejected: %s", reason)
    recon = reconcile_active_trade()
    if recon:
        logger.info("Reconcile status: %s", recon)
    log_run_summary(summary)
    return 1


def _next_sgt_run(now: datetime, start_hour: int, start_minute: int, interval_hours: int) -> datetime:
    base = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    if base > now:
        return base
    delta_hours = int((now - base).total_seconds() // 3600) + 1
    next_hours = ((delta_hours + interval_hours - 1) // interval_hours) * interval_hours
    return base + timedelta(hours=next_hours)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous trading bot")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol, e.g. BTCUSDT")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Polling interval in seconds for continuous mode",
    )
    parser.add_argument(
        "--schedule",
        choices=["sgt_4h", "interval"],
        default="sgt_4h",
        help="Schedule mode (default: sgt_4h)",
    )

    args = parser.parse_args()

    _configure_file_logging()

    if args.once:
        raise SystemExit(run_once(args.symbol))

    if args.schedule == "interval":
        interval = args.interval if args.interval > 0 else 3600
        logger.info("Starting interval mode (interval=%ss)", interval)
        while True:
            run_once(args.symbol)
            time.sleep(interval)
    else:
        tz = pytz.timezone(os.getenv("TIMEZONE", "Asia/Singapore"))
        start_hour = int(os.getenv("SCHEDULE_START_HOUR", "8"))
        start_minute = int(os.getenv("SCHEDULE_START_MINUTE", "1"))
        interval_hours = int(os.getenv("SCHEDULE_INTERVAL_HOURS", "4"))
        logger.info(
            "Starting SGT schedule (start=%02d:%02d, interval=%sh)",
            start_hour,
            start_minute,
            interval_hours,
        )
        while True:
            now = datetime.now(tz)
            next_run = _next_sgt_run(now, start_hour, start_minute, interval_hours)
            sleep_for = max((next_run - now).total_seconds(), 0)
            logger.info("Next run at %s", next_run.strftime("%Y-%m-%d %H:%M:%S %Z"))
            time.sleep(sleep_for)
            run_once(args.symbol)


if __name__ == "__main__":
    main()
