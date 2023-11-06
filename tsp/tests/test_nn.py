import pytest
import numpy as np
from ..nn import NearestNeighbor
from ..city import get_cities_csv


@pytest.fixture
def cities():
    """
    Fixture that returns a list of cities loaded from a CSV file.

    Returns:
        List[City]: A list of City objects representing cities.
    """
    return get_cities_csv()


@pytest.fixture
def tsp_nn_return(cities):
    """
    Fixture that initializes a NearestNeighbor for TSP with return-to-origin.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        NearestNeighbor: An instance of NearestNeighbor for the TSP with return-to-origin.
    """
    return NearestNeighbor(cities, return_to_city=True, verbose=False)


@pytest.fixture
def tsp_nn_no_return(cities):
    """
    Fixture that initializes a NearestNeighbor for TSP without return-to-origin.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        NearestNeighbor: An instance of NearestNeighbor for the TSP without return-to-origin.
    """
    return NearestNeighbor(cities, return_to_city=False, verbose=False)


def test_get_nearest_city(tsp_nn_return, cities):
    """
    Test the NearestNeighbor algorithm for finding the nearest city.

    Args:
        cities (List[City]): A list of City objects representing cities.

    This test verifies that the get_nearest_city method of the NearestNeighbor class correctly
    identifies the nearest city to a given starting city and returns the expected results.
    """
    start_city = cities[0]

    nearest_distance, nearest_city = tsp_nn_return.get_nearest_city(start_city)

    assert np.isclose(nearest_distance, 22.71)
    assert nearest_city == cities[12]


def test_tsp_nn_return(tsp_nn_return):
    """
    Test the NearestNeighbor solver with the return_to_city option.

    Parameters:
    - tsp_nn_return (NearestNeighbor): An instance of the NearestNeighbor solver.

    It checks if the tour starts and ends at the starting city.
    """
    _, tour = tsp_nn_return.fit()
    assert tour[0] == tour[-1] == tsp_nn_return.start_city


def test_tsp_nn_no_return(tsp_nn_no_return):
    """
    Test the NearestNeighbor solver when the return_to_city option is disabled

    Parameters:
    - tsp_nn_no_return (NearestNeighbor): An instance of the NearestNeighbor solver.

    It checks if the tour starts at the starting city and visits all cities exactly once.
    """
    _, tour = tsp_nn_no_return.fit()
    assert tour[0] == tsp_nn_no_return.start_city
    assert len(set(tour)) == tsp_nn_no_return.number_cities
