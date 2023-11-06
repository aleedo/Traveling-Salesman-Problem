import numpy as np
import pandas as pd
from typing import List


class City:
    next_id = 0

    def __init__(self, x=None, y=None, name=None):
        """
        Initialize a City object with coordinates and a name.

        Args:
            x (float): X-coordinate of the city.
            y (float): Y-coordinate of the city.
            name (int or str): Name or identifier for the city (optional).

        If 'name' is not provided, it will be automatically assigned.
        """
        self.x = x
        self.y = y
        self.name = name if name is not None else City.next_id
        City.next_id += 1

    def __repr__(self):
        """
        Return a string representation of the City object.
        """
        return f"city_{self.name}"


def get_cities_csv(file_name="tsp/data/15-Points.csv") -> List[City]:
    """
    Read cities' coordinates from a CSV file and return a list of City objects.

    Args:
        file_name (str): Path to the CSV file containing city coordinates.

    Returns:
        List[City]: A list of City objects representing the cities.

    This function reads city data from a CSV file and creates City objects for each city.
    """
    df = pd.read_csv(file_name)
    cities = [City(city.x, city.y, city.name) for _, city in df.iterrows()]
    return cities


def get_cities_numpy(array=None, file_name="tsp/data/15-Points.npy") -> List[City]:
    """
    Create City objects from a NumPy array or load them from a binary file.

    Args:
        array (numpy.ndarray): NumPy array containing city coordinates (optional).
        file_name (str): Path to a NumPy binary file containing city coordinates (optional).

    Returns:
        List[City]: A list of City objects representing the cities.

    This function creates City objects from a NumPy array or loads them from a binary file.
    If 'array' is provided, it is used directly; otherwise, data is loaded from the specified file.
    """
    if array is None:
        array = np.load(file_name)

    cities = [City(*row) for row in array]
    return cities
