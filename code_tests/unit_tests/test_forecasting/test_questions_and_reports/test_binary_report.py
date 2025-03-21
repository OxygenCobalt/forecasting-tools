import numpy as np
import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.data_models.binary_report import BinaryReport


def test_prediction_validation() -> None:
    # Valid predictions
    report = ForecastingTestManager.get_fake_forecast_report(prediction=0.5)
    assert report.prediction == pytest.approx(0.5)

    report = ForecastingTestManager.get_fake_forecast_report(prediction=0.0001)
    assert report.prediction == pytest.approx(0.0001)

    report = ForecastingTestManager.get_fake_forecast_report(prediction=0.9999)
    assert report.prediction == pytest.approx(0.9999)

    report = ForecastingTestManager.get_fake_forecast_report(prediction=0)
    assert report.prediction == pytest.approx(0)

    report = ForecastingTestManager.get_fake_forecast_report(prediction=1)
    assert report.prediction == pytest.approx(1)

    # Invalid predictions
    with pytest.raises(ValueError):
        ForecastingTestManager.get_fake_forecast_report(prediction=-0.1)

    with pytest.raises(ValueError):
        ForecastingTestManager.get_fake_forecast_report(prediction=1.1)

    with pytest.raises(ValueError):
        ForecastingTestManager.get_fake_forecast_report(prediction=2.0)


async def test_aggregate_predictions() -> None:
    question = ForecastingTestManager.get_fake_binary_question()
    predictions = [0.1, 0.2, 0.3, 0.4, 0.5]

    result = await BinaryReport.aggregate_predictions(predictions, question)
    assert result == pytest.approx(0.3)  # Median of predictions

    # Test invalid predictions
    with pytest.raises(Exception):
        await BinaryReport.aggregate_predictions([-0.1, 0.5], question)

    with pytest.raises(Exception):
        await BinaryReport.aggregate_predictions([1.1, 0.5], question)


def test_expected_baseline_score() -> None:
    # Test with valid community prediction
    report = ForecastingTestManager.get_fake_forecast_report(
        prediction=0.6, community_prediction=0.7
    )
    score = report.expected_baseline_score
    assert score is not None
    expected_score = 100.0 * (
        0.7 * (np.log(0.6) + 1.0) + (1.0 - 0.7) * (np.log(1.0 - 0.6) + 1.0)
    )
    assert score == pytest.approx(expected_score)
    assert score > 0

    # Test better prediction less than worse prediction
    better_report = ForecastingTestManager.get_fake_forecast_report(
        prediction=0.6, community_prediction=0.7
    )
    worse_report = ForecastingTestManager.get_fake_forecast_report(
        prediction=0.4, community_prediction=0.7
    )
    better_score = better_report.expected_baseline_score
    worse_score = worse_report.expected_baseline_score
    assert better_score is not None
    assert worse_score is not None
    assert better_score > worse_score

    # Test with None community prediction
    report = ForecastingTestManager.get_fake_forecast_report(
        prediction=0.6, community_prediction=None
    )
    assert report.expected_baseline_score is None


def test_deviation_points() -> None:
    # Test with valid community prediction
    report = ForecastingTestManager.get_fake_forecast_report(
        prediction=0.6, community_prediction=0.7
    )
    deviation = report.deviation_points
    assert deviation is not None
    assert deviation == pytest.approx(0.1)

    # Test with None community prediction
    report = ForecastingTestManager.get_fake_forecast_report(
        prediction=0.6, community_prediction=None
    )
    assert report.deviation_points is None


def test_calculate_average_expected_log_score() -> None:
    reports = [
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.6, community_prediction=0.7
        ),
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.3, community_prediction=0.4
        ),
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.8, community_prediction=0.7
        ),
    ]

    average_score = BinaryReport.calculate_average_expected_baseline_score(
        reports
    )
    assert isinstance(average_score, float)

    # Test with None community prediction
    reports_with_none = reports + [
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.5, community_prediction=None
        )
    ]
    with pytest.raises(AssertionError):
        BinaryReport.calculate_average_expected_baseline_score(
            reports_with_none
        )


def test_calculate_average_deviation_points() -> None:
    reports = [
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.6, community_prediction=0.7
        ),  # 0.1 deviation
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.3, community_prediction=0.4
        ),  # 0.1 deviation
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.8, community_prediction=0.7
        ),  # 0.1 deviation
    ]

    average_deviation = BinaryReport.calculate_average_deviation_points(
        reports
    )
    assert average_deviation == pytest.approx(0.1)

    # Test with None community prediction
    reports_with_none = reports + [
        ForecastingTestManager.get_fake_forecast_report(
            prediction=0.5, community_prediction=None
        )
    ]
    with pytest.raises(AssertionError):
        BinaryReport.calculate_average_deviation_points(reports_with_none)
