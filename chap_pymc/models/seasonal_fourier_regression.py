"""
SeasonalFourierRegression - Fourier-based seasonal disease forecasting model
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pymc as pm
import xarray
from pandas import DataFrame
from xarray import DataArray, Dataset

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
    FourierParametrization,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.transformations.model_input_creator import (
    FourierInputCreator,
    NormalizationParams,
)
from chap_pymc.transformations.seasonal_xarray import TimeCoords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_output(training_pdf: pd.DataFrame, posterior_samples: np.ndarray, n_samples: int = 1000, season_length: int = 12) -> pd.DataFrame:
    """
    Convert posterior samples to output DataFrame format.

    Args:
        training_pdf: Training data DataFrame with location and time_period columns
        posterior_samples: Array of shape (n_locations, n_periods, n_samples)
        n_samples: Number of samples to include in output
        season_length: Number of periods per year (12 for months, 52 for weeks)

    Returns:
        DataFrame with columns: location, time_period, sample_0, sample_1, ..., sample_N
    """
    n_samples = min(n_samples, posterior_samples.shape[-1])
    horizon = posterior_samples.shape[-2]
    locations = training_pdf['location'].unique()
    last_time_idx = training_pdf['time_period'].max()
    year, period = map(int, last_time_idx.split('-'))

    # Calculate future time periods
    raw_periods = np.arange(horizon) + period
    new_periods = (raw_periods % season_length) + 1
    new_years = year + raw_periods // season_length
    new_time_periods = [f'{y:d}-{p:02d}' for y, p in zip(new_years, new_periods, strict=True)]

    colnames = ['location', 'time_period'] + [f'sample_{i}' for i in range(n_samples)]
    rows = []

    for l_id, location in enumerate(locations):
        for t_id, time_period in enumerate(new_time_periods):
            samples = posterior_samples[l_id, t_id, -n_samples:]
            new_row = [location, time_period] + samples.tolist()
            rows.append(new_row)

    return pd.DataFrame(rows, columns=colnames)

class SeasonalFourierRegressionV2:
    class Params(pydantic.BaseModel):
        inference_params: InferenceParams = InferenceParams()
        fourier_hyperparameters: FourierHyperparameters = FourierHyperparameters()
        input_params: FourierInputCreator.Params = FourierInputCreator.Params()

    def __init__(self, params: Params = Params(), name: str|None=None) -> None:
        self._params = params
        self._name = name

    def predict(self, training_data: pd.DataFrame, future_data: pd.DataFrame,
                save_plot: bool = False, country: str = 'model') -> pd.DataFrame:
        ds, mapping = self.get_input_data(future_data, training_data)
        samples = self.get_raw_samples(ds)
        prediction_df = self.get_predictions_df(future_data, mapping, samples)
        return prediction_df

    def get_input_data(self, future_data: DataFrame, training_data: DataFrame) -> tuple[Dataset, tuple[dict[str, TimeCoords], NormalizationParams]]:
        ds, mappings = FourierInputCreator(params=self._params.input_params).v2(training_data, future_data)

        return ds, mappings

    def get_predictions_df(self, future_data: DataFrame, mappings: tuple[dict[str, TimeCoords], NormalizationParams], samples: DataArray) -> DataFrame:
        mapping, n_params = mappings
        samples_xr = samples*n_params.std+n_params.mean
        n_samples = self._params.inference_params.n_samples
        indices = np.random.choice(samples_xr.sizes['samples'], replace=True, size=n_samples)
        samples_np: np.ndarray[Any, Any] = np.maximum(0, np.expm1(samples_xr.isel(samples=indices)))
        colnames = ['location', 'time_period'] + [f'sample_{i}' for i in range(n_samples)]
        rows = []
        # Convert back to xarray for selection
        samples_final = xarray.DataArray(samples_np, coords=samples_xr.isel(samples=indices).coords, dims=samples_xr.dims)
        for row in future_data.itertuples():
            location = row.location
            time_period = row.time_period
            array_coords = mapping[str(time_period)]
            sample_values = samples_final.sel(location=location, **array_coords.model_dump()).values
            new_row = [location, time_period] + sample_values.tolist()
            rows.append(new_row)
        prediction_df = pd.DataFrame(rows, columns=colnames)
        return prediction_df

    def get_raw_samples(self, ds: xarray.Dataset) -> xarray.DataArray:
        season_length = ds.attrs.get('season_length', 12)  # Get season_length from Dataset attributes
        fourier_model = FourierParametrization(self._params.fourier_hyperparameters, season_length=season_length)
        # ds = ds.expand_dims(fourier_model.extra_dims)
        coords = {dim: ds[dim].values for dim in ds.dims} | fourier_model.extra_dims
        with pm.Model(coords=coords):
            prev_year_y = ds.get('prev_year_y', None)  # Get from Dataset if available
            fourier_model.get_regression_model(ds.X, ds.y, prev_year_y=prev_year_y)

            # Choose inference method based on inference_params.method
            inference_params = self._params.inference_params
            if inference_params.method == 'hmc':
                idata = pm.sample(**inference_params.model_dump(exclude={'method', 'n_iterations'}))
            else:  # 'advi'
                approx = pm.fit(n=inference_params.n_iterations, method='advi')
                idata = approx.sample(inference_params.n_samples)
            posterior_predictive = pm.sample_posterior_predictive(idata, var_names=['y_obs', 'A']).posterior_predictive
        samples: xarray.DataArray = posterior_predictive['y_obs'].stack(samples=('chain', 'draw'))
        return samples
