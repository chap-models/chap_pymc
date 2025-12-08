"""Test fixtures for chap_pymc tests."""

import itertools
import logging
from pathlib import Path

import pandas as pd
import pydantic
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging for all tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        force=True,
    )


@pytest.fixture
def df() -> pd.DataFrame:
    """Load training data fixture."""
    p = Path(__file__).parent / "fixtures" / "data" / "training_data.csv"
    return pd.read_csv(p)


@pytest.fixture
def data_path() -> Path:
    """Return path to fixtures data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def large_df(data_path: Path) -> pd.DataFrame:
    """Load Thailand dataset."""
    p = data_path / "thailand.csv"
    return pd.read_csv(p)


@pytest.fixture
def colombia_df(data_path: Path) -> pd.DataFrame:
    """Load Colombia dataset."""
    return pd.read_csv(data_path / "colombia.csv")


class Coords(pydantic.BaseModel):
    """Coordinate model for tests."""

    locations: list[str]
    years: list[int]
    variables: list[str]


@pytest.fixture
def simple_coords() -> Coords:
    """Return simple test coordinates."""
    locations = ["loc1", "loc2"]
    years = [2021, 2022]
    variables = ["disease_cases", "mean_temperature"]
    return Coords(locations=locations, years=years, variables=variables)


@pytest.fixture
def simple_monthly_data(simple_coords: Coords) -> pd.DataFrame:
    """Create simple monthly data for tests."""
    months = list(range(1, 13))
    rows = []
    for i, (location, year, month) in enumerate(
        itertools.product(simple_coords.locations, simple_coords.years, months)
    ):
        rows.append(
            {
                "location": location,
                "time_period": f"{year}-{month:02d}",
            }
            | {var: float(i * t + 1) for t, var in enumerate(simple_coords.variables)}
        )
    return pd.DataFrame(rows)


@pytest.fixture
def simple_future_data(simple_coords: Coords) -> pd.DataFrame:
    """Create simple future data for tests."""
    months = [1, 2, 3]
    rows = []
    for i, (location, month) in enumerate(
        itertools.product(simple_coords.locations, months)
    ):
        rows.append(
            {
                "location": location,
                "time_period": f"2023-{month:02d}",
            }
            | {var: float(i * t + 100) for t, var in enumerate(simple_coords.variables)}
        )
    return pd.DataFrame(rows)


@pytest.fixture
def weekly_data() -> pd.DataFrame:
    """Alias for weekly_data_fixture for backward compatibility."""
    return weekly_data_fixture_impl()


def weekly_data_fixture_impl() -> pd.DataFrame:
    """Create weekly time series data with date range format.

    Returns DataFrame with columns: location, time_period, disease_cases, mean_temperature
    Uses format: "YYYY-MM-DD/YYYY-MM-DD" for time_period
    """
    from datetime import datetime, timedelta

    import numpy as np

    locations = ["LocationA", "LocationB"]
    start_date = datetime(2020, 1, 6)
    weeks = 52 * 3

    data = []
    for location in locations:
        for week_num in range(weeks):
            week_start = start_date + timedelta(weeks=week_num)
            week_end = week_start + timedelta(days=6)

            week_of_year = week_start.isocalendar()[1]
            disease_cases = (
                10 + 5 * np.sin(2 * np.pi * week_of_year / 52) + np.random.randn() * 0.5
            )
            mean_temperature = (
                20 + 10 * np.sin(2 * np.pi * week_of_year / 52) + np.random.randn()
            )

            time_period = (
                f'{week_start.strftime("%Y-%m-%d")}/{week_end.strftime("%Y-%m-%d")}'
            )
            data.append(
                {
                    "location": location,
                    "time_period": time_period,
                    "disease_cases": max(0, disease_cases),
                    "mean_temperature": mean_temperature,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def weekly_data_fixture() -> pd.DataFrame:
    """Fixture wrapper for weekly data implementation."""
    return weekly_data_fixture_impl()
