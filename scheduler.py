"""Daily Scheduler for AI Sentiment Blog Post Generation.

Runs daily at specified UTC+8 time to generate and publish blog posts.
Includes optional GitHub automation for automated commits and pushes.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, time
from typing import Optional

import schedule
import pytz
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coincise.blog.sentiment_analyzer import SentimentAnalyzer
from coincise.blog.generator import BlogPostGenerator
from coincise.blog.github_utils import GitHubAutomation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class BlogScheduler:
    """Manages daily blog generation scheduling."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        hour: int = 0,
        minute: int = 0,
        timezone: str = "Asia/Singapore",
        auto_push: bool = False
    ):
        """Initialize the scheduler.

        Args:
            symbol: Trading symbol (default: BTCUSDT)
            hour: Hour to run (UTC+8, default 0 = midnight)
            minute: Minute to run (default 0)
            timezone: Timezone string (default Asia/Singapore)
            auto_push: Auto-push to GitHub (default False for local testing)
        """
        self.symbol = symbol
        self.hour = hour
        self.minute = minute
        self.timezone = pytz.timezone(timezone)
        self.auto_push = auto_push

        self.analyzer = SentimentAnalyzer(symbol=symbol)
        self.generator = BlogPostGenerator(symbol=symbol)
        self.github = GitHubAutomation() if auto_push else None

    def generate_blog_post(self) -> bool:
        """Generate a single blog post.

        Returns:
            True if generation successful
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 Blog Generation Started for {self.symbol}")
            logger.info(f"{'='*60}\n")

            # Compile market data
            logger.info("📊 Compiling market data...")
            market_data = self.analyzer.compile_market_data()

            if not market_data:
                logger.error("❌ Failed to compile market data")
                return False

            # Analyze sentiment
            logger.info("\n🤖 Analyzing sentiment...")
            sentiment_result = self.analyzer.analyze_market(market_data)

            if not sentiment_result:
                logger.error("❌ Failed to analyze sentiment")
                return False

            # Generate blog post
            logger.info("\n📝 Generating blog post...")
            blog_content = self.generator.generate_post(sentiment_result, market_data)

            if not blog_content:
                logger.error("❌ Failed to generate blog post")
                return False

            # Save blog post
            logger.info("\n💾 Saving blog post...")
            filepath = self.generator.save_post(blog_content, symbol=self.symbol)

            if not filepath:
                logger.error("❌ Failed to save blog post")
                return False

            logger.info(f"\n✅ Blog post saved: {filepath}")

            # Auto-push to GitHub if enabled
            if self.auto_push and self.github:
                logger.info("\n🔄 Pushing to GitHub...")
                if self.github.commit_and_push(filepath, self.symbol):
                    logger.info("✅ Successfully pushed to GitHub")
                else:
                    logger.warning("⚠️ GitHub push failed, but blog post generated locally")

            logger.info(f"\n{'='*60}")
            logger.info("✅ Blog generation completed successfully!")
            logger.info(f"{'='*60}\n")

            return True

        except Exception as e:
            logger.error(f"❌ Error generating blog post: {e}")
            return False

    def schedule_daily(self) -> None:
        """Schedule daily blog generation.

        Runs at specified hour/minute in specified timezone.
        """
        time_str = f"{self.hour:02d}:{self.minute:02d}"
        logger.info(f"📅 Scheduling daily blog generation at {time_str} {self.timezone}")

        # Schedule the job
        schedule.every().day.at(time_str).do(self.generate_blog_post)

        logger.info(f"✅ Scheduler configured. Press Ctrl+C to stop.\n")

    def run_scheduler(self, run_once: bool = False) -> None:
        """Run the scheduler (blocking).

        Args:
            run_once: If True, generate once and exit (for testing)
        """
        if run_once:
            logger.info("🧪 Running once (test mode)...")
            self.generate_blog_post()
            return

        # Schedule daily
        self.schedule_daily()

        # Keep scheduler running
        try:
            while True:
                schedule.run_pending()
                import time
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("\n\n⛔ Scheduler stopped by user")
            sys.exit(0)

    def get_next_run_time(self) -> datetime:
        """Get the next scheduled run time.

        Returns:
            Next run datetime in local timezone
        """
        now = datetime.now(self.timezone)
        next_run = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)

        # If the scheduled time has passed today, schedule for tomorrow
        if next_run <= now:
            from datetime import timedelta
            next_run = next_run + timedelta(days=1)

        return next_run

    def status(self) -> str:
        """Get scheduler status.

        Returns:
            Human-readable status string
        """
        next_run = self.get_next_run_time()
        time_until = next_run - datetime.now(self.timezone)
        hours, remainder = divmod(int(time_until.total_seconds()), 3600)
        minutes = remainder // 60

        return f"""
📋 SCHEDULER STATUS
{'='*50}
Symbol:         {self.symbol}
Schedule:       Daily at {self.hour:02d}:{self.minute:02d} {self.timezone}
Next Run:       {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}
Time Until:     {hours}h {minutes}m
Auto-Push:      {'✅ Enabled' if self.auto_push else '❌ Disabled'}
Status:         🟢 Ready
{'='*50}
"""


def main():
    """Main entry point."""
    # Configuration from environment
    symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT")
    publish_hour = int(os.getenv("PUBLISH_HOUR", "0"))
    publish_minute = int(os.getenv("PUBLISH_MINUTE", "0"))
    timezone = os.getenv("TIMEZONE", "Asia/Singapore")
    auto_push = os.getenv("AUTO_PUSH", "false").lower() == "true"

    # Create scheduler
    scheduler = BlogScheduler(
        symbol=symbol,
        hour=publish_hour,
        minute=publish_minute,
        timezone=timezone,
        auto_push=auto_push
    )

    # Print status
    logger.info(scheduler.status())

    # Run scheduler (or test mode with --once)
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        logger.info("🧪 Running in test mode (once)...")
        scheduler.run_scheduler(run_once=True)
    else:
        logger.info("🚀 Starting scheduler...")
        scheduler.run_scheduler()


if __name__ == "__main__":
    main()
