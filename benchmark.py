from __future__ import annotations

import asyncio
import os
import sys
from typing import Sequence

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)
dotenv.load_dotenv()

import logging

from forecasting_tools.forecasting.forecast_bots.community.laylaps import (
    LaylapsBot,
)
from forecasting_tools.util.custom_logger import CustomLogger

CustomLogger.setup_logging()

logger = logging.getLogger(__name__)

from forecasting_tools import Benchmarker, BenchmarkForBot

# Run benchmark on multiple bots
bots: Sequence[LaylapsBot] = [
    LaylapsBot(
        research_reports_per_question=1, predictions_per_research_report=1
    )
]  # Add your custom bots here
benchmarker = Benchmarker(
    forecast_bots=bots,
    number_of_questions_to_use=100,  # Recommended 100+ for meaningful results
    file_path_to_save_reports="benchmarks/",
)
benchmarks: list[BenchmarkForBot] = asyncio.run(benchmarker.run_benchmark())

# View results
for benchmark in benchmarks[:2]:
    print("--------------------------------")
    print(f"Bot: {benchmark.name}")
    print(
        f"Score: {benchmark.average_inverse_expected_log_score}"
    )  # Lower is better
    print(f"Num reports in benchmark: {len(benchmark.forecast_reports)}")
    print(f"Time: {benchmark.time_taken_in_minutes}min")
    print(f"Cost: ${benchmark.total_cost}")
