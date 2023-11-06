from ..city import City
from ..src.base import _TSP
from typing import List
import numpy as np


class NearestNeighbor(_TSP):
    def __init__(
        self,
        cities,
        **kwargs,
    ):
        """
        Initialize the Nearest Neighbor TSP solver.

        Parameters:
        - cities (List[City]): A list of city objects representing the cities to be visited.
        - **kwargs: Additional keyword arguments, including solver-specific parameters.
        """
        kwargs["solver"] = "nn"
        super(NearestNeighbor, self).__init__(cities, **kwargs)

        self.current_city = self.start_city
        self.tour = [self.start_city]
        self.cost = 0
        self.visited = [False] * len(self.cities)

    def calculate_distance(self, city1, city2):
        """
        Calculate the Euclidean distance between two cities.

        Args:
        - city1 (City): The first city.
        - city2 (City): The second city.

        Returns:
        - float: The Euclidean distance between the two cities.
        """
        return np.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)

    def get_nearest_city(self, current_city):
        """
        Find the nearest unvisited city to the current city.

        Args:
        - current_city (City): The current city.

        Returns:
        - float: The distance to the nearest city.
        - City: The nearest unvisited city.
        """
        nearest_city = None
        nearest_distance = np.inf
        for i, city in enumerate(self.cities):
            if self.visited[i] or city == current_city:
                continue
            distance = self.calculate_distance(current_city, city)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_city = city
        return nearest_distance, nearest_city

    def fit(self):
        """
        Run the algorithm to find the optimal tour.

        Returns:
        - cost (float): The total cost (distance) of the optimal tour.
        - tour (List[City]): The sequence of cities representing the optimal tour.
        """
        for i in range(self.number_cities - 1):
            self.visited[self.cities.index(self.current_city)] = True
            nearest_distance, self.current_city = self.get_nearest_city(
                self.current_city
            )
            self.cost += nearest_distance
            self.tour += [self.current_city]

        if self.return_to_city:
            self.cost += self.calculate_distance(self.current_city, self.start_city)
            self.tour += [self.start_city]

        if self.verbose:
            self.plot_tour()

        return self.cost, self.tour
