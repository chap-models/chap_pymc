# chap_pymc

Fourier seasonal forecasting service for disease incidence prediction, integrated with [chapkit](https://github.com/dhis2-chap/chapkit).

## Overview

This package provides a Bayesian seasonal forecasting model using PyMC with Fourier harmonics. It's designed to predict disease cases based on seasonal patterns and temperature features.

## Installation

```bash
uv sync
```

## Running the Service

Start the FastAPI service:

```bash
uv run fastapi dev main.py
```

The service will be available at `http://localhost:8000`.

## API Usage

### Create Configuration

```bash
curl -X POST http://localhost:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{
    "method": "advi",
    "n_harmonics": 2,
    "n_iterations": 30000
  }'
```

### Train Model

```bash
curl -X POST http://localhost:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "<config_ulid>",
    "data": {
      "columns": ["location", "time_period", "disease_cases", "mean_temperature"],
      "data": [...]
    }
  }'
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_id": "<training_artifact_ulid>",
    "historic": {...},
    "future": {...}
  }'
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"advi"` | Inference method: `"hmc"` or `"advi"` |
| `draws` | `500` | Number of posterior draws (HMC) |
| `tune` | `500` | Number of tuning steps (HMC) |
| `chains` | `2` | Number of chains (HMC) |
| `n_iterations` | `30000` | ADVI iterations |
| `n_samples` | `100` | Number of prediction samples |
| `n_harmonics` | `2` | Number of Fourier harmonics |
| `prior_strength` | `1.0` | Prior scale factor |
| `lag` | `1` | Temperature lag in months |

## Development

### Run Tests

```bash
uv run pytest -v -m 'not slow'
```

### Type Checking

```bash
uv run mypy chap_pymc/
```

### Linting

```bash
uv run ruff check chap_pymc/ main.py
```

## Project Structure

```
chap_pymc/
├── main.py                          # Chapkit service entry point
├── chap_pymc/
│   ├── inference_params.py          # HMC/ADVI parameters
│   ├── curve_parametrizations/
│   │   └── fourier_parametrization.py  # Core Fourier model
│   ├── models/
│   │   └── seasonal_fourier_regression.py  # Model orchestration
│   └── transformations/
│       ├── model_input_creator.py   # Data transformation
│       ├── seasonal_transform.py    # Seasonal format conversion
│       └── seasonal_xarray.py       # xarray dataset creation
└── tests/
```

## License

MIT
