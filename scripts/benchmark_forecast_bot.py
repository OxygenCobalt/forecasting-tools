from __future__ import annotations

import asyncio
import logging

import typeguard

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.benchmarker import Benchmarker
from forecasting_tools.forecasting.forecast_bots.community.laylapso import (
    LaylapsO1Bot,
    LaylapsO3Bot,
)
from forecasting_tools.forecasting.forecast_bots.community.laylapsr import (
    LaylapsPerplexityBot,
)
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot() -> None:
    questions_to_use = 100
    with MonetaryCostManager() as cost_manager:
        bots = [
            LaylapsPerplexityBot(
                research_reports_per_question=1,
                predictions_per_research_report=1,
            ),
            LaylapsO1Bot(
                research_reports_per_question=1,
                predictions_per_research_report=1,
                research_used=["perplexity"],
            ),
            LaylapsO3Bot(
                research_reports_per_question=1,
                predictions_per_research_report=1,
                research_used=["perplexity"],
            ),
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarks = await Benchmarker(
            number_of_questions_to_use=questions_to_use,
            forecast_bots=bots,
            file_path_to_save_reports="logs/forecasts/benchmarks/",
            concurrent_question_batch_size=10,
        ).run_benchmark()
        for i, benchmark in enumerate(benchmarks):
            logger.info(
                f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}"
            )
            logger.info(
                f"- Final Score: {benchmark.average_expected_baseline_score}"
            )
            logger.info(f"- Total Cost: {benchmark.total_cost}")
            logger.info(f"- Time taken: {benchmark.time_taken_in_minutes}")
        logger.info(f"Total Cost: {cost_manager.current_usage}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(benchmark_forecast_bot())
