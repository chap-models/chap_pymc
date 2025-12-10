"""Chapkit-integrated Fourier seasonal forecasting service.

CLIM-237: Chapkit integration for chap_pymc model.
"""

from typing import Any

import pandas as pd
import structlog
from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import BaseModelRunner
from geojson_pydantic import FeatureCollection

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegressionV2
from chap_pymc.transformations.model_input_creator import FourierInputCreator

log = structlog.get_logger()


class FourierModelConfig(BaseConfig):
    """Configuration for Fourier seasonal forecasting model."""

    # Inference parameters
    method: str = "advi"  # "hmc" or "advi"
    draws: int = 500
    tune: int = 500
    chains: int = 2
    target_accept: float = 0.9
    n_iterations: int = 30000
    n_samples: int = 100

    # Fourier hyperparameters
    n_harmonics: int = 2
    prior_strength: float = 1.0

    # Input parameters
    lag: int = 1


class FourierModelRunner(BaseModelRunner[FourierModelConfig]):
    """Model runner for Fourier seasonal forecasting."""

    async def on_train(
        self,
        config: FourierModelConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train is a no-op - PyMC model trains during prediction.

        Returns config for reproducibility tracking.
        """
        log.info("fourier_model_train", note="Model trains during prediction")
        return {"config": config.model_dump()}

    async def on_predict(
        self,
        config: FourierModelConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions using the Fourier seasonal model."""
        historic_df = historic.to_pandas()
        future_df = future.to_pandas()

        frequency = self._detect_frequency(historic_df)
        log.info(
            "prediction_started",
            frequency=frequency,
            historic_rows=len(historic_df),
            future_rows=len(future_df),
        )

        # Build model parameters from config
        inference_params = InferenceParams(
            method=config.method,
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            n_iterations=config.n_iterations,
            n_samples=config.n_samples,
        )
        fourier_hyperparameters = FourierHyperparameters(
            n_harmonics=config.n_harmonics,
            prior_strength=config.prior_strength,
        )
        input_params = FourierInputCreator.Params(lag=config.lag)
        input_params.seasonal_params.frequency = frequency

        params = SeasonalFourierRegressionV2.Params(
            inference_params=inference_params,
            fourier_hyperparameters=fourier_hyperparameters,
            input_params=input_params,
        )

        # Run prediction
        regression_model = SeasonalFourierRegressionV2(params)
        predictions_df = regression_model.predict(
            historic_df, future_df, save_plot=False
        )

        log.info(
            "prediction_complete",
            rows=len(predictions_df),
            sample_columns=len(
                [c for c in predictions_df.columns if c.startswith("sample_")]
            ),
        )

        return DataFrame.from_pandas(predictions_df)

    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect data frequency from time_period format."""
        sample_period = str(df["time_period"].iloc[0])

        if "/" in sample_period:
            return "W"
        if "w" in sample_period.lower():
            return "W"

        parts = sample_period.split("-")
        if len(parts) == 2:
            period = int(parts[1])
            if period > 12:
                return "W"

        return "M"


# Service configuration
info = MLServiceInfo(
    display_name="Fourier Seasonal Forecasting Service",
    version="1.0.0",
    summary="Bayesian seasonal forecasting using Fourier harmonics",
    description=(
        "Predict disease incidence using PyMC-based Fourier seasonal model "
        "with temperature features and hierarchical priors."
    ),
    author="DHIS2 CHAP Team",
    author_assessed_status=AssessedStatus.yellow,
    contact_email="chap@dhis2.org",
)

HIERARCHY = ArtifactHierarchy(
    name="fourier_forecast",
    level_labels={0: "ml_training", 1: "ml_prediction"},
)

runner = FourierModelRunner()

app = (
    MLServiceBuilder(
        info=info,
        config_schema=FourierModelConfig,
        hierarchy=HIERARCHY,
        runner=runner,
    )
    .with_monitoring()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
