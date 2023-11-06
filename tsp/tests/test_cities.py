import numpy as np
import pytest
from ..city import City, get_cities_csv, get_cities_numpy


def test_city_init():
    """
    Test the initialization of a City object.

    This test verifies that the City object is correctly initialized with the provided
    coordinates and name.
    """
    city = City(x=1, y=2, name="A")
    assert city.x == 1
    assert city.y == 2
    assert city.name == "A"


def test_get_cities_csv():
    """
    Test loading cities from a CSV file.

    This test ensures that City objects are correctly created when loading cities from a CSV file.
    """
    City.cities = []
    cities = get_cities_csv("tsp/data/15-Points.csv")
    assert len(cities) == 15
    assert isinstance(cities[0], City)


def test_get_cities_numpy_file():
    """
    Test loading cities from a NumPy binary file.

    This test ensures that City objects are correctly created when loading cities from a NumPy
    binary file.
    """
    City.cities = []
    cities = get_cities_numpy(file_name="tsp/data/15-Points.npy")
    assert len(cities) == 15
    assert isinstance(cities[0], City)


def test_get_cities_numpy_array():
    """
    Test creating City objects from a NumPy array.

    This test verifies that City objects are correctly created from a NumPy array, and that the
    array has the expected shape.
    """
    array = np.ones((15, 2))
    cities = get_cities_numpy(array)
    assert array.shape[-1] == 2, "Must Be A 2 Dimensional Array"
    assert len(cities) == 15
    assert isinstance(cities[0], City)
