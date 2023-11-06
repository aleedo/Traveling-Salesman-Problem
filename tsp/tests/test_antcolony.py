import numpy as np
import pytest
from ..city import get_cities_csv
from ..antcolony import Ant, AntColonyOptimization


@pytest.fixture
def cities():
    """
    Fixture that returns a list of cities loaded from a CSV file.

    Returns:
        List[City]: A list of City objects representing cities.
    """
    return get_cities_csv()


@pytest.fixture
def tsp_aco_return(cities):
    """
    Fixture that initializes a AntColonyOptimization for TSP with return-to-origin.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        AntColonyOptimization: An instance of AntColonyOptimization for the TSP with return-to-origin.
    """
    return AntColonyOptimization(cities, verbose=False, return_to_city=True)


@pytest.fixture
def tsp_aco_no_return(cities):
    """
    Fixture that initializes a AntColonyOptimization for TSP without return-to-origin.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        AntColonyOptimization: An instance of AntColonyOptimization for the TSP without return-to-origin.
    """
    return AntColonyOptimization(cities, verbose=False, return_to_city=False)


@pytest.fixture
def sample_ant_parameters(cities):
    """
    Fixture that provides sample parameters for creating an Ant object.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        Tuple: A tuple of parameters used to initialize an Ant object.
    """
    start_city = cities[0]
    return_to_city = False
    alpha = 1.0
    beta = 1.0
    pheromone_matrix = np.ones((len(cities), len(cities)))
    distance_matrix = np.zeros((len(cities), len(cities)))
    evaporation_rate = 0.1
    return (
        start_city,
        return_to_city,
        alpha,
        beta,
        pheromone_matrix,
        distance_matrix,
        evaporation_rate,
    )


def test_ant_generate_tour(cities, sample_ant_parameters):
    """
    Test the generation of a tour by an Ant object.

    Args:
        cities (List[City]): A list of City objects representing cities.
        sample_ant_parameters (Tuple): Sample parameters for initializing an Ant.

    This test verifies that the generated tour by an Ant object includes all cities exactly once.
    """
    ant = Ant(cities, *sample_ant_parameters)
    ant.generate_tour()
    assert len(set(ant.tour)) == len(set(cities))


def test_update_pheromone_matrix(cities):
    """
    Test the update of the pheromone matrix by the AntColonyOptimization.

    Args:
        cities (List[City]): A list of City objects representing cities.

    This test ensures that the update of the pheromone matrix results in a matrix where each row
    contains only two unique values (start and end city pheromone levels).
    """
    ant_colony = AntColonyOptimization(cities, number_ants=1)
    ant = ant_colony.ants[0]
    ant.generate_tour()
    ant_colony.update_pheromone_matrix()
    assert all(len(set(row)) == 2 for row in ant_colony.pheromone_matrix)


def test_tsp_aco_return(tsp_aco_return):
    """
    Test the AntColonyOptimization solver with the return_to_city option.

    Parameters:
    - tsp_aco_return (AntColonyOptimization): An instance of the AntColonyOptimization solver.

    It checks if the tour starts and ends at the starting city.
    """
    _, tour = tsp_aco_return.fit()
    assert tour[0] == tour[-1] == tsp_aco_return.start_city


def test_tsp_aco_no_return(tsp_aco_no_return):
    """
    Test the AntColonyOptimization solver when the return_to_city option is disabled

    Parameters:
    - tsp_aco_no_return (AntColonyOptimization): An instance of the AntColonyOptimization solver.

    It checks if the tour starts at the starting city and visits all cities exactly once.
    """
    _, tour = tsp_aco_no_return.fit()
    assert tour[0] == tsp_aco_no_return.start_city
    assert len(set(tour)) == tsp_aco_no_return.number_cities
