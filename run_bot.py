from __future__ import annotations

import argparse
import asyncio
import os
import sys

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)
dotenv.load_dotenv()

import logging

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.forecasting.forecast_bots.community.laylapso import (
    LaylapsO1Bot,
)
from forecasting_tools.util.custom_logger import CustomLogger

CustomLogger.setup_logging()

logger = logging.getLogger(__name__)


async def run_forecasts(skip_previous: bool, tournament: int | str) -> None:
    """
    Make a copy of this file called run_bot.py (i.e. remove template) and fill in your bot details.
    This will be run in the workflows
    """
    forecaster = LaylapsO1Bot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previous,
        use_research_summary_to_forecast=False,
        research_used=["perplexity"],
    )
    reports = await forecaster.forecast_on_tournament(
        tournament, return_exceptions=True
    )
    valid_reports = [
        report for report in reports if isinstance(report, ForecastReport)
    ]

    forecaster.log_report_summary(reports)

    total_cost = 0
    for report in valid_reports:
        total_cost += report.price_estimate if report.price_estimate else 0
        await asyncio.sleep(5)
    logger.info(f"Total cost estimated: {total_cost}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run forecasts with specified bot type"
    )
    parser.add_argument(
        "--skip_previous",
        type=str,
        required=True,
        help="Skip previously forecasted questions (True or False)",
    )
    parser.add_argument(
        "--tournament",
        type=str,
        required=True,
        help="Tournament to forecast on",
    )
    args = parser.parse_args()

    try:
        tournament = int(args.tournament)
    except ValueError:
        tournament = str(args.tournament)

    if args.skip_previous == "True":
        skip_previous = True
    elif args.skip_previous == "False":
        skip_previous = False
    else:
        raise ValueError(
            f"Invalid value for skip_previous: {args.skip_previous}. "
            "Must be True or False"
        )

    asyncio.run(run_forecasts(skip_previous, tournament))
