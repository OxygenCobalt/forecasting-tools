"""
Run this file as a streamlit app. `streamlit run scripts/benchmark_displayer.py`
As long as your benchmark files (contain 'bench' and end in '.json')
are in the project directory tree, you should be able to view them.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
import typeguard

from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.util import file_manipulation
from front_end.helpers.report_displayer import ReportDisplayer


def get_json_files(directory: str) -> list[str]:
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and "bench" in file:
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    return sorted(json_files)


def display_deviation_scores(reports: list[BinaryReport]) -> None:
    with st.expander("Scores", expanded=False):
        certain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and (r.community_prediction > 0.9 or r.community_prediction < 0.1)
        ]
        uncertain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and 0.1 <= r.community_prediction <= 0.9
        ]
        display_stats_for_report_type(reports, "All Questions")
        display_stats_for_report_type(
            certain_reports,
            "Certain Questions: Community Prediction >90% or <10%",
        )
        display_stats_for_report_type(
            uncertain_reports,
            "Uncertain Questions: Community Prediction 10%-90%",
        )


def display_stats_for_report_type(
    reports: list[BinaryReport], title: str
) -> None:
    average_expected_baseline_score = (
        BinaryReport.calculate_average_expected_baseline_score(reports)
    )
    average_deviation = BinaryReport.calculate_average_deviation_points(
        reports
    )
    st.markdown(
        f"""
        #### {title}
        - Number of Questions: {len(reports)}
        - Expected Baseline Score (lower is better): {average_expected_baseline_score:.4f}
        - Average Deviation: On average, there is a {average_deviation:.2%} percentage point difference between community and bot
        """
    )


def display_questions_and_forecasts(reports: list[BinaryReport]) -> None:
    with st.expander("Questions and Forecasts", expanded=False):
        st.subheader("Question List")
        certain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and (r.community_prediction > 0.9 or r.community_prediction < 0.1)
        ]
        uncertain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and 0.1 <= r.community_prediction <= 0.9
        ]

        display_question_stats_in_list(
            certain_reports,
            "Certain Questions (Community Prediction >90% or <10%)",
        )
        display_question_stats_in_list(
            uncertain_reports,
            "Uncertain Questions (Community Prediction 10%-90%)",
        )


def display_question_stats_in_list(
    report_list: list[BinaryReport], title: str
) -> None:
    st.subheader(title)
    sorted_reports = sorted(
        report_list,
        key=lambda r: (
            r.expected_baseline_score
            if r.expected_baseline_score is not None
            else -1
        ),
        reverse=True,
    )
    for report in sorted_reports:
        deviation = (
            report.expected_baseline_score
            if report.expected_baseline_score is not None
            else -1
        )
        st.write(
            f"- **Δ:** {deviation:.4f} | **🤖:** {report.prediction:.2%} | **👥:** {report.community_prediction:.2%} | **Question:** {report.question.question_text}"
        )


def display_benchmark_list(benchmarks: list[BenchmarkForBot]) -> None:
    if not benchmarks:
        return

    st.markdown("# Select Benchmark")
    benchmark_options = [
        f"{b.name} (Score: {b.average_expected_baseline_score:.4f})"
        for b in benchmarks
    ]
    selected_benchmark_name = st.selectbox(
        "Select a benchmark to view details:", benchmark_options
    )

    st.markdown("---")

    selected_idx = benchmark_options.index(selected_benchmark_name)
    benchmark = benchmarks[selected_idx]

    with st.expander(benchmark.name, expanded=False):
        st.markdown(f"**Description:** {benchmark.description}")
        st.markdown(
            f"**Time Taken (minutes):** {benchmark.time_taken_in_minutes}"
        )
        st.markdown(f"**Total Cost:** {benchmark.total_cost}")
        st.markdown(f"**Git Commit Hash:** {benchmark.git_commit_hash}")
        st.markdown(
            f"**Average Expected Baseline Score:** {benchmark.average_expected_baseline_score:.4f}"
        )
        # Add average deviation score if reports are binary
        if isinstance(benchmark.forecast_reports[0], BinaryReport):
            reports = typeguard.check_type(
                benchmark.forecast_reports, list[BinaryReport]
            )
            average_deviation = (
                BinaryReport.calculate_average_deviation_points(reports)
            )
            st.markdown(
                f"**Average Deviation Score:** {average_deviation:.2%} percentage points"
            )

    with st.expander("Bot Configuration", expanded=False):
        st.markdown("### Bot Configuration")
        for key, value in benchmark.forecast_bot_config.items():
            st.markdown(f"**{key}:** {value}")

    if benchmark.code:
        with st.expander("Forecast Bot Code", expanded=False):
            st.code(benchmark.code, language="python")

    # Display deviation scores and questions for this benchmark
    reports = benchmark.forecast_reports
    if isinstance(reports[0], BinaryReport):
        reports = typeguard.check_type(reports, list[BinaryReport])
        display_deviation_scores(reports)
        display_questions_and_forecasts(reports)
        ReportDisplayer.display_report_list(reports)


def get_benchmark_display_name(benchmark: BenchmarkForBot, index: int) -> str:
    config = benchmark.forecast_bot_config
    reports_per_q = config.get("research_reports_per_question", "1")
    preds_per_r = config.get("predictions_per_research_report", "1")
    return f"{index}: {benchmark.name} ({reports_per_q}x{preds_per_r})"


def add_star_annotations(
    fig: px.bar, df: pd.DataFrame, x_col: str, y_col: str, is_starred_col: str
) -> None:
    """Add star annotations to bars meeting the starred condition."""
    for idx, row in df[df[is_starred_col]].iterrows():
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text="★",
            showarrow=False,
            yshift=10,
            font=dict(size=20),
        )


def display_benchmark_comparison_graphs(
    benchmarks: list[BenchmarkForBot],
) -> None:
    st.markdown("# Benchmark Score Comparisons")
    data_by_benchmark = []

    for index, benchmark in enumerate(benchmarks):
        reports = benchmark.forecast_reports
        reports = typeguard.check_type(reports, list[BinaryReport])
        certain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and (r.community_prediction > 0.9 or r.community_prediction < 0.1)
        ]
        uncertain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and 0.1 <= r.community_prediction <= 0.9
        ]
        certain_reports = typeguard.check_type(
            certain_reports, list[ForecastReport]
        )
        uncertain_reports = typeguard.check_type(
            uncertain_reports, list[ForecastReport]
        )

        benchmark_name = get_benchmark_display_name(benchmark, index)

        data_by_benchmark.extend(
            [
                {
                    "Benchmark": benchmark_name,
                    "Category": "All Questions",
                    "Expected Baseline Score": benchmark.average_expected_baseline_score,
                    "Deviation Score": BinaryReport.calculate_average_deviation_points(
                        reports
                    )
                    * 100,
                },
                {
                    "Benchmark": benchmark_name,
                    "Category": "Certain Questions",
                    "Expected Baseline Score": BinaryReport.calculate_average_expected_baseline_score(
                        certain_reports
                    ),
                    "Deviation Score": BinaryReport.calculate_average_deviation_points(
                        certain_reports
                    )
                    * 100,
                },
                {
                    "Benchmark": benchmark_name,
                    "Category": "Uncertain Questions",
                    "Expected Baseline Score": BinaryReport.calculate_average_expected_baseline_score(
                        uncertain_reports
                    ),
                    "Deviation Score": BinaryReport.calculate_average_deviation_points(
                        uncertain_reports
                    )
                    * 100,
                },
            ]
        )

    if not data_by_benchmark:
        return

    try:
        df = pd.DataFrame(data_by_benchmark)

        st.markdown("### Expected Baseline Scores")
        st.markdown(
            "Higher score indicates better performance. Read more [here](https://www.metaculus.com/help/scores-faq/#:~:text=The%20Baseline%20score%20compares,probability%20to%20all%20outcomes.)"
        )

        # Mark highest scores with stars (higher is better for baseline score)
        max_scores = df.groupby("Category")[
            "Expected Baseline Score"
        ].transform("max")
        df["Is Best Expected"] = df["Expected Baseline Score"] == max_scores

        fig = px.bar(
            df,
            x="Benchmark",
            y="Expected Baseline Score",
            color="Category",
            barmode="group",
            title="Expected Baseline Scores by Benchmark and Category",
        )
        fig.update_layout(yaxis_title="Expected Baseline Score")

        add_star_annotations(
            fig, df, "Benchmark", "Expected Baseline Score", "Is Best Expected"
        )

        st.plotly_chart(fig)

        st.markdown("### Deviation Scores")
        st.markdown(
            "Lower score indicates predictions closer to community consensus. Shown as difference in percentage points between bot and community."
        )

        # Mark lowest deviations with stars (lower is better for deviation score)
        min_deviations = df.groupby("Category")["Deviation Score"].transform(
            "min"
        )
        df["Is Best Deviation"] = df["Deviation Score"] == min_deviations

        fig = px.bar(
            df,
            x="Benchmark",
            y="Deviation Score",
            color="Category",
            barmode="group",
            title="Deviation Scores by Benchmark and Category",
        )
        fig.update_layout(yaxis_title="Deviation Score (percentage points)")

        add_star_annotations(
            fig, df, "Benchmark", "Deviation Score", "Is Best Deviation"
        )

        st.plotly_chart(fig)

    except ImportError:
        st.error(
            "Please install plotly and pandas to view the graphs: `pip install plotly pandas`"
        )


def make_perfect_benchmark(
    model_benchmark: BenchmarkForBot,
) -> BenchmarkForBot:
    print("work3")
    perfect_benchmark = model_benchmark.model_copy()
    print("work4")
    reports_of_perfect_benchmark = [
        report.model_copy() for report in perfect_benchmark.forecast_reports
    ]
    print("work5")
    reports_of_perfect_benchmark = typeguard.check_type(
        reports_of_perfect_benchmark, list[BinaryReport]
    )
    print("work6")
    for report in reports_of_perfect_benchmark:
        print(report)
        assert report.community_prediction is not None
        report.prediction = report.community_prediction
    print("work7")
    perfect_benchmark.forecast_reports = reports_of_perfect_benchmark
    perfect_benchmark.name = "Perfect Predictor (questions of benchmark 1)"
    return perfect_benchmark


def main() -> None:
    st.title("Benchmark Viewer")
    st.write("Select JSON files containing BenchmarkForBot objects.")

    project_directory = file_manipulation.get_absolute_path("")
    json_files = get_json_files(project_directory)

    if not json_files:
        st.warning(f"No JSON files found in {project_directory}")
        return

    selected_files = st.multiselect(
        "Select benchmark files:",
        json_files,
        format_func=lambda x: os.path.basename(x),
    )

    if selected_files:
        try:
            all_benchmarks: list[BenchmarkForBot] = []
            for file in selected_files:
                print(file)
                benchmarks = BenchmarkForBot.load_json_from_file_path(file)
                all_benchmarks.extend(benchmarks)
            print("work")

            perfect_benchmark = make_perfect_benchmark(all_benchmarks[0])
            print("work 2")
            all_benchmarks.insert(0, perfect_benchmark)

            benchmark_options = [
                f"{i}: {b.name} (Score: {b.average_expected_baseline_score:.4f})"
                for i, b in enumerate(all_benchmarks)
            ]
            selected_benchmarks = st.multiselect(
                "Select benchmarks to display:",
                range(len(all_benchmarks)),
                default=range(len(all_benchmarks)),
                format_func=lambda i: benchmark_options[i],
            )

            if selected_benchmarks:
                filtered_benchmarks = [
                    all_benchmarks[i] for i in selected_benchmarks
                ]
                display_benchmark_comparison_graphs(filtered_benchmarks)
                display_benchmark_list(filtered_benchmarks)
        except Exception as e:
            st.error(f"Could not load files. Error: {str(e)}")


if __name__ == "__main__":
    main()
