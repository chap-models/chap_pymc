"""Tests for FourierParametrization model."""

import numpy as np
import pymc as pm
import pytest
import xarray

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
    FourierParametrization,
)


@pytest.fixture
def coords():
    """Create test coordinates."""
    return {
        "location": np.arange(3),
        "epi_year": np.arange(4),
        "epi_offset": np.arange(12),
    }


@pytest.fixture
def y(coords):
    """Create synthetic target data.

    DataArray with dimensions (location, epi_year, epi_offset)
    locations: 3
    years: 4 (epi years)
    months: 12
    """
    L = len(coords["location"])
    Y = len(coords["epi_year"])
    M = len(coords["epi_offset"])
    months = np.arange(M)
    locs = np.arange(L)
    years = np.arange(Y)
    pattern = np.sin(2 * np.pi * months / 12)
    y_data = locs[:, None, None] + years[None, :, None] + pattern[None, None, :]
    return xarray.DataArray(y_data, dims=["location", "epi_year", "epi_offset"])


def test_fourier_hyperparameters_defaults():
    """Test default hyperparameters."""
    hp = FourierHyperparameters()
    assert hp.n_harmonics == 3  # Default is 3
    assert hp.prior_strength == 1.0


def test_fourier_hyperparameters_custom():
    """Test custom hyperparameters."""
    hp = FourierHyperparameters(n_harmonics=3, prior_strength=0.5)
    assert hp.n_harmonics == 3
    assert hp.prior_strength == 0.5


def test_fourier_parametrization_extra_dims():
    """Test that extra_dims returns harmonic dimension."""
    fp = FourierParametrization(FourierHyperparameters(n_harmonics=3))
    assert "harmonic" in fp.extra_dims
    assert len(fp.extra_dims["harmonic"]) == 4  # 0=baseline + 3 harmonics


def test_fourier_parametrization_model_builds(y, coords):
    """Test that the model can be built without errors."""
    coords_with_harmonic = coords | {"harmonic": np.arange(0, 3)}
    with pm.Model(coords=coords_with_harmonic):
        FourierParametrization(FourierHyperparameters(n_harmonics=2)).get_model(y)
        # Just verify model builds without sampling


@pytest.mark.slow
def test_fourier_parametrization_samples(y, coords):
    """Test that the model can sample (slow test)."""
    coords_with_harmonic = coords | {"harmonic": np.arange(0, 3)}
    with pm.Model(coords=coords_with_harmonic):
        FourierParametrization(FourierHyperparameters(n_harmonics=2)).get_model(y)
        idata = pm.sample(draws=50, tune=50, chains=1, progressbar=False)
    assert "mu" in idata.posterior
