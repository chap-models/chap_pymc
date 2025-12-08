"""Tests for main.py chapkit service."""

import pandas as pd
import pytest

from main import FourierModelConfig, FourierModelRunner


class TestFourierModelConfig:
    """Test config schema."""

    def test_default_config(self):
        config = FourierModelConfig()
        assert config.method == "advi"
        assert config.n_harmonics == 2
        assert config.n_iterations == 30000

    def test_custom_config(self):
        config = FourierModelConfig(
            method="hmc",
            n_harmonics=3,
            draws=1000,
        )
        assert config.method == "hmc"
        assert config.n_harmonics == 3
        assert config.draws == 1000


class TestFourierModelRunner:
    """Test model runner."""

    def test_detect_frequency_monthly(self):
        runner = FourierModelRunner()
        df = pd.DataFrame({"time_period": ["2024-01", "2024-02", "2024-03"]})
        assert runner._detect_frequency(df) == "M"

    def test_detect_frequency_weekly_date_range(self):
        runner = FourierModelRunner()
        df = pd.DataFrame(
            {"time_period": ["2024-01-01/2024-01-07", "2024-01-08/2024-01-14"]}
        )
        assert runner._detect_frequency(df) == "W"

    def test_detect_frequency_weekly_iso(self):
        runner = FourierModelRunner()
        df = pd.DataFrame({"time_period": ["2024-W01", "2024-W02", "2024-W03"]})
        assert runner._detect_frequency(df) == "W"

    def test_detect_frequency_weekly_high_period(self):
        runner = FourierModelRunner()
        df = pd.DataFrame({"time_period": ["2024-15", "2024-16", "2024-17"]})
        assert runner._detect_frequency(df) == "W"


@pytest.mark.slow
async def test_train_returns_config():
    """Test that on_train returns config dict."""
    from chapkit.data import DataFrame

    runner = FourierModelRunner()
    config = FourierModelConfig(n_harmonics=3)
    data = DataFrame.from_pandas(
        pd.DataFrame(
            {
                "location": ["A", "A"],
                "time_period": ["2024-01", "2024-02"],
                "disease_cases": [10, 20],
                "mean_temperature": [25.0, 26.0],
            }
        )
    )

    result = await runner.on_train(config, data)
    assert "config" in result
    assert result["config"]["n_harmonics"] == 3
