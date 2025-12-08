"""Tests for SeasonalFourierRegressionV2"""

import pandas as pd
import pytest

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegressionV2
from chap_pymc.transformations.model_input_creator import FourierInputCreator


def test_seasonal_fourier_regression_v2_basic(simple_monthly_data, simple_future_data):
    """Smoke test for SeasonalFourierRegressionV2 with simple data."""
    model = SeasonalFourierRegressionV2(
        params=SeasonalFourierRegressionV2.Params(
            inference_params=InferenceParams(method="advi", n_iterations=100),
            fourier_hyperparameters=FourierHyperparameters(n_harmonics=1),
        )
    )

    result = model.predict(simple_monthly_data, simple_future_data)

    assert isinstance(result, pd.DataFrame)
    assert "location" in result.columns
    assert "time_period" in result.columns
    assert result["location"].nunique() == 2

    sample_cols = [c for c in result.columns if c.startswith("sample_")]
    assert len(sample_cols) > 0


def test_seasonal_fourier_regression_v2_params():
    """Test that Params model validates correctly."""
    params = SeasonalFourierRegressionV2.Params()
    assert params.inference_params.method == "hmc"  # Default is hmc
    assert params.fourier_hyperparameters.n_harmonics == 3  # Default is 3

    custom_params = SeasonalFourierRegressionV2.Params(
        inference_params=InferenceParams(method="hmc", draws=500),
        fourier_hyperparameters=FourierHyperparameters(n_harmonics=3),
    )
    assert custom_params.inference_params.method == "hmc"
    assert custom_params.fourier_hyperparameters.n_harmonics == 3


@pytest.mark.slow
def test_seasonal_fourier_regression_v2_weekly(weekly_data_fixture):
    """Test SeasonalFourierRegressionV2 with weekly frequency data."""
    from datetime import timedelta

    from chap_pymc.transformations.seasonal_xarray import SeasonalXArray

    weekly_data = weekly_data_fixture

    # Create future weekly data
    last_time_period = weekly_data["time_period"].max()
    last_end_date = pd.to_datetime(last_time_period.split("/")[1])

    locations = weekly_data["location"].unique()
    future_rows = []
    for location in locations:
        for week_offset in range(1, 4):
            week_start = last_end_date + timedelta(days=1 + (week_offset - 1) * 7)
            week_end = week_start + timedelta(days=6)
            time_period = (
                f'{week_start.strftime("%Y-%m-%d")}/{week_end.strftime("%Y-%m-%d")}'
            )
            future_rows.append(
                {
                    "location": location,
                    "time_period": time_period,
                    "mean_temperature": 20.0,
                }
            )
    future_data = pd.DataFrame(future_rows)

    input_params = FourierInputCreator.Params(
        seasonal_params=SeasonalXArray.Params(frequency="W")
    )

    model = SeasonalFourierRegressionV2(
        params=SeasonalFourierRegressionV2.Params(
            inference_params=InferenceParams(method="advi", n_iterations=100),
            fourier_hyperparameters=FourierHyperparameters(n_harmonics=2),
            input_params=input_params,
        )
    )

    result = model.predict(weekly_data, future_data, save_plot=False)

    assert isinstance(result, pd.DataFrame)
    assert "location" in result.columns
    assert "time_period" in result.columns

    n_locations = len(locations)
    n_future_weeks = 3
    assert len(result) == n_locations * n_future_weeks

    sample_cols = [c for c in result.columns if c.startswith("sample_")]
    assert len(sample_cols) > 0
    assert (result[sample_cols] >= 0).all().all()
