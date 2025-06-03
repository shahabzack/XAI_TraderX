import logging
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import os
import subprocess
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('/app/logs/scheduler.log', maxBytes=1000000, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def run_daily_scripts():
    """Run daily_update_stock.py and predict_and_update.py with retries."""

     # ✅ Skip Sundays (weekday == 6)
    if datetime.now().weekday() == 6:
        logger.info("Today is Sunday. Skipping script execution.")
        return
    
    logger.info("Starting daily scripts at %s", datetime.now())
    scripts = ['scripts/daily_update_stock.py', 'scripts/predict_and_update.py']
    max_retries = 3
    retry_delay = 3600  # 1 hour in seconds

    for script in scripts:
        attempts = 0
        success = False
        while attempts < max_retries and not success:
            attempts += 1
            logger.info("Attempt %d/%d: Running %s", attempts, max_retries, script)
            try:
                result = subprocess.run(
                    ['python', script],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("%s completed successfully: %s", script, result.stdout)
                success = True
            except subprocess.CalledProcessError as e:
                logger.error("Attempt %d/%d: Error running %s: %s", attempts, max_retries, script, e.stderr)
                if attempts < max_retries:
                    logger.info("Retrying %s in %d seconds", script, retry_delay)
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached for %s. Giving up.", script)
            except Exception as e:
                logger.error("Attempt %d/%d: Unexpected error running %s: %s", attempts, max_retries, script, str(e))
                if attempts < max_retries:
                    logger.info("Retrying %s in %d seconds", script, retry_delay)
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached for %s. Giving up.", script)

def main():
    logger.info("Starting scheduler")
    
    # Read scheduler time from .env
    update_time = os.getenv("SCHEDULER_UPDATE_TIME", "20:00")  # Default to 8 PM if not set
    try:
        hour, minute = map(int, update_time.split(":"))
    except ValueError:
        logger.error("Invalid SCHEDULER_UPDATE_TIME format in .env: %s. Using default 20:00", update_time)
        hour, minute = 20, 0

    scheduler = BlockingScheduler(timezone='Asia/Kolkata')
    scheduler.add_job(
        run_daily_scripts,
        trigger=CronTrigger(hour=hour, minute=minute),  # Use .env time
        id='daily_scripts',
        name='Run daily stock update and prediction scripts'
    )
    logger.info("Scheduler started, waiting for %s IST", update_time)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
        scheduler.shutdown()

if __name__ == "__main__":
    main()