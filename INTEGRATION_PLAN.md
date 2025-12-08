# Chapkit Integration Plan

## Overview

This plan outlines the integration of chap_pymc's Fourier-based seasonal forecasting model with chapkit to create a simple, clean example of chapkit model integration.

## Current State

The chap_pymc repository contains:
- **Core model**: `SeasonalFourierRegressionV2` - Bayesian seasonal model using PyMC
- **Data transformation**: Complex pipeline converting time-series to seasonal xarray format
- **CLI**: cyclopts-based CLI in `main.py` with `train` and `predict` commands
- **Many auxiliary files**: plotting, exploratory analysis, legacy models (~6000 lines total)

## Target State

A minimal chapkit-integrated service with:
- Single `main.py` exposing train/predict endpoints via chapkit's `MLServiceBuilder`
- Core model files only (Fourier parametrization + data transformation)
- Chapkit-compatible config schema
- REST API for train/predict operations

## Files to KEEP (Essential)

### Core Model Logic
| File | Purpose |
|------|---------|
| `chap_pymc/curve_parametrizations/fourier_parametrization.py` | Core Fourier model definition |
| `chap_pymc/curve_parametrizations/__init__.py` | Package init |
| `chap_pymc/models/seasonal_fourier_regression.py` | Main model orchestration |
| `chap_pymc/transformations/model_input_creator.py` | Data transformation pipeline |
| `chap_pymc/transformations/seasonal_transform.py` | Time-series to seasonal format |
| `chap_pymc/transformations/seasonal_xarray.py` | xarray dataset creation |
| `chap_pymc/inference_params.py` | HMC/ADVI inference parameters |
| `chap_pymc/util.py` | Utility functions |
| `chap_pymc/__init__.py` | Package init |

### Configuration
| File | Purpose |
|------|---------|
| `configurations/` | Keep one example config (optional) |

### Project Files
| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies (will need chapkit added) |
| `CLAUDE.md` | Development instructions |
| `tests/` | Essential tests only |

## Files to REMOVE (Auxiliary/Legacy)

### Plotting & Visualization
- `chap_pymc/plotting.py` (634 lines)
- `chap_pymc/dataset_plots.py`
- `chap_pymc/season_plot.py`
- `chap_pymc/season_correlation.py`
- `chap_pymc/correlation_plots.py`
- `chap_pymc/yearly_regression.py`
- `chap_pymc/curve_parametrizations/fourier_parametrization_plots.py`
- `chap_pymc/example_dataset_plots/`

### Legacy/Alternative Models
- `chap_pymc/models/seasonal_regression.py` (superseded by V2)
- `chap_pymc/models/kmer_model.py` (unrelated)
- `chap_pymc/models/model_with_dimensions.py` (experimental)
- `chap_pymc/models/loc_scale_finder.py` (exploratory)
- `chap_pymc/extension.py`
- `chap_pymc/mcmc_params.py`

### Old CLI & Examples
- `main.py` (replaced by chapkit integration)
- `examples/` directory
- `apps/` directory (if exists)

### Config Files
- `chap_pymc/configs/chap_config.py` (will be replaced by chapkit BaseConfig)

### Generated/Temporary Files
- `runs/`, `results/`, `mlruns/`, `old_output/`
- `*.nc` files, `*.png` files, `*.csv` test outputs
- `target/`
- `.idea/`
- `notebooks/`

## New Files to CREATE

### 1. `main.py` - Chapkit Service Entry Point

```python
"""Chapkit-integrated Fourier seasonal forecasting service."""

from typing import Any

import structlog
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import BaseModelRunner

from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegressionV2
from chap_pymc.inference_params import InferenceParams
from chap_pymc.curve_parametrizations.fourier_parametrization import FourierHyperparameters
from chap_pymc.transformations.model_input_creator import FourierInputCreator
from chap_pymc.transformations.seasonal_xarray import SeasonalXArray

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
    mixture_weight_prior: float = 0.5

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
        """Train is a no-op - this model trains during prediction."""
        # PyMC models don't separate train/predict in the traditional sense
        # The model is built and sampled during prediction
        log.info("fourier_model_train_called", note="Model trains during prediction")
        return {"config": config.model_dump()}

    async def on_predict(
        self,
        config: FourierModelConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions using the Fourier model."""
        import pandas as pd

        historic_df = historic.to_pandas()
        future_df = future.to_pandas()

        # Detect frequency from data
        frequency = self._detect_frequency(historic_df)
        log.info("prediction_started", frequency=frequency,
                 historic_rows=len(historic_df), future_rows=len(future_df))

        # Build model parameters
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
            mixture_weight_prior=config.mixture_weight_prior,
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
        predictions_df = regression_model.predict(historic_df, future_df, save_plot=False)

        log.info("prediction_complete", sample_columns=len([c for c in predictions_df.columns if c.startswith('sample_')]))

        return DataFrame.from_pandas(predictions_df)

    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect data frequency from time_period format."""
        sample_period = str(df['time_period'].iloc[0])

        if '/' in sample_period:
            return 'W'
        if 'w' in sample_period.lower():
            return 'W'

        parts = sample_period.split('-')
        if len(parts) == 2:
            period = int(parts[1])
            if period > 12:
                return 'W'

        return 'M'


# Service configuration
info = MLServiceInfo(
    display_name="Fourier Seasonal Forecasting Service",
    version="1.0.0",
    summary="Bayesian seasonal forecasting using Fourier harmonics",
    description="Train and predict disease incidence using PyMC-based Fourier seasonal model with temperature features",
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
```

### 2. Update `pyproject.toml`

Add chapkit dependency:
```toml
dependencies = [
    # ... existing deps ...
    "chapkit",  # Add this
]
```

Remove cyclopts (no longer needed):
```toml
# Remove: "cyclopts>=2.0.0",
```

Update entry point:
```toml
[project.scripts]
chap-pymc = "main:app"
```

## Implementation Steps

### Phase 1: Create Branch & Backup
1. [x] Create `chapkit-integration` branch
2. [ ] Tag current state for reference

### Phase 2: Add Chapkit Integration
3. [ ] Add chapkit to dependencies
4. [ ] Create new `main.py` with chapkit service
5. [ ] Create `FourierModelConfig` extending `BaseConfig`
6. [ ] Implement `FourierModelRunner` with `on_train` and `on_predict`

### Phase 3: Clean Up Code
7. [ ] Remove plotting modules
8. [ ] Remove legacy/alternative models
9. [ ] Remove old CLI (main.py -> rename to main_old.py first)
10. [ ] Remove auxiliary files (examples, notebooks, etc.)
11. [ ] Clean up configs module

### Phase 4: Update Dependencies
12. [ ] Remove cyclopts from pyproject.toml
13. [ ] Remove visualization dependencies (matplotlib, seaborn, plotly, altair)
14. [ ] Add chapkit, structlog

### Phase 5: Update Tests
15. [ ] Remove tests for deleted modules
16. [ ] Add integration test for chapkit service
17. [ ] Ensure remaining tests pass

### Phase 6: Documentation
18. [ ] Update README with chapkit usage
19. [ ] Update CLAUDE.md if needed

## Design Decisions

### Why BaseModelRunner over FunctionalModelRunner?
- Cleaner organization for complex model with multiple parameters
- Lifecycle hooks may be useful for future enhancements
- Type-safe config parameter

### Why train returns minimal data?
- PyMC models combine training and prediction in one step
- MCMC sampling happens during prediction
- The "model" artifact stores config for reproducibility
- Future: Could pre-compile PyTensor graph during train

### Frequency Detection
- Kept inline in runner (simple utility)
- Supports monthly ('M') and weekly ('W') formats
- Auto-detected from time_period column format

## Expected Final Structure

```
chap_pymc/
├── main.py                          # NEW: Chapkit service
├── pyproject.toml                   # Updated dependencies
├── CLAUDE.md                        # Development guide
├── README.md                        # Updated usage docs
├── chap_pymc/
│   ├── __init__.py
│   ├── inference_params.py          # KEEP
│   ├── util.py                      # KEEP
│   ├── curve_parametrizations/
│   │   ├── __init__.py
│   │   └── fourier_parametrization.py  # KEEP
│   ├── models/
│   │   ├── __init__.py
│   │   └── seasonal_fourier_regression.py  # KEEP
│   └── transformations/
│       ├── __init__.py
│       ├── model_input_creator.py   # KEEP
│       ├── seasonal_transform.py    # KEEP
│       └── seasonal_xarray.py       # KEEP
├── tests/
│   ├── conftest.py
│   ├── test_integration.py          # NEW: Chapkit integration test
│   ├── curve_parametrizations/
│   │   └── test_fourier_parametrization.py
│   ├── models/
│   │   └── test_seasonal_fourier_regression.py
│   └── transformations/
│       ├── test_model_input_creator_v2.py
│       └── test_seasonal_xarray.py
└── configurations/
    └── example_config.yml           # Example config
```

## Estimated Line Count Reduction

| Category | Before | After | Reduction |
|----------|--------|-------|-----------|
| Production code | ~6000 | ~1500 | 75% |
| Test code | ~1500 | ~800 | 47% |
| Total | ~7500 | ~2300 | 69% |

## API Usage After Integration

### Start Service
```bash
uv run fastapi dev main.py
```

### Create Config
```bash
curl -X POST http://localhost:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"method": "advi", "n_harmonics": 2, "n_iterations": 30000}'
```

### Train (stores config)
```bash
curl -X POST http://localhost:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "<config_ulid>",
    "data": {"columns": [...], "data": [...]}
  }'
```

### Predict
```bash
curl -X POST http://localhost:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_id": "<training_artifact_ulid>",
    "historic": {"columns": [...], "data": [...]},
    "future": {"columns": [...], "data": [...]}
  }'
```