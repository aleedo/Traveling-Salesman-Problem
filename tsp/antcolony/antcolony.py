from ..city import City
from ..src.base import _TSP
from typing import List
import numpy as np


class Ant:
    def __init__(
        self,
        cities,
        start_city,
        return_to_city,
        alpha,
        beta,
        pheromone_matrix,
        distance_matrix,
        evaporation_rate,
    ):
        """
        Initialize an Ant object for the Ant Colony Optimization algorithm.

        Parameters:
        - cities (List[City]): List of cities to visit.
        - start_city (City): The starting city for the ant.
        - return_to_city (bool): Whether the ant returns to the starting city.
        - alpha (float): Parameter controlling the influence of pheromone levels in ant decisions.
        - beta (float): Parameter controlling the influence of distance in ant decisions.
        - pheromone_matrix (np.ndarray): Matrix representing pheromone levels on edges between cities.
        - distance_matrix (np.ndarray): Matrix representing distances between cities.
        - evaporation_rate (float): Rate at which pheromone evaporates.

        Initializes instance variables for the ant's attributes.
        """
        self.cities = cities
        self.start_city = start_city
        self.return_to_city = return_to_city
        self.alpha = alpha
        self.beta = beta
        self.pheromone_matrix = pheromone_matrix
        self.distance_matrix = distance_matrix
        self.evaporation_rate = evaporation_rate
        self.num_cities = distance_matrix.shape[0]
        self.eps = 1e-5
        self.best_tour = None
        self.best_cost = float("inf")

    def get_next_city(self):
        """
        Select the next city for the ant to visit based on pheromone levels and distance.

        Returns:
        - next_city (City): The next city to visit.

        This method calculates probabilities for each city based on pheromone levels
        and distances, and selects the next city to visit probabilistically.
        """
        pheromone_values = (
            self.pheromone_matrix[self.cities.index(self.current_city)] ** self.alpha
        )

        proximity = (
            1 / (self.distance_matrix[self.cities.index(self.current_city)] + self.eps)
        ) ** self.beta

        probabilities = pheromone_values * proximity
        probabilities[list(map(lambda x: self.cities.index(x), self.tour))] = 0
        probabilities /= probabilities.sum()

        next_city = np.random.choice(self.cities, p=probabilities)
        return next_city

    def generate_tour(self):
        """
        Generate a tour for the ant by selecting cities to visit.

        Updates self.tour and self.cost with the tour and its total cost.
        """
        self.cost = 0
        self.current_city = self.start_city
        self.tour = [self.current_city]

        while len(self.tour) < self.num_cities:
            next_city = self.get_next_city()
            self.tour.append(next_city)
            self.cost += self.distance_matrix[self.cities.index(self.current_city)][
                self.cities.index(next_city)
            ]
            self.current_city = next_city

        if self.return_to_city:
            self.cost += self.distance_matrix[self.cities.index(self.tour[-1])][
                self.cities.index(self.start_city)
            ]
            self.tour.append(self.start_city)

        if self.cost < self.best_cost:
            self.best_cost, self.best_tour = self.cost, self.tour


class AntColonyOptimization(_TSP):
    def __init__(self, cities: List[City], *args, **kwargs):
        """
        Initialize the Ant Colony Optimization solver for the Traveling Salesman Problem.

        Parameters:
        - cities (List[City]): List of cities to visit.
        - *args, **kwargs: Additional arguments and keyword arguments.

        Initializes the AntColonyOptimization object, including ants, distance and pheromone matrices.
        """
        kwargs["solver"] = "antcolony"
        super(AntColonyOptimization, self).__init__(cities, *args, **kwargs)

        self.distance_matrix = self.generate_distance_matrix(cities)
        self.pheromone_matrix = np.ones((self.number_cities, self.number_cities))

        self.ants = [
            Ant(
                self.cities,
                self.start_city,
                self.return_to_city,
                self.alpha,
                self.beta,
                self.pheromone_matrix,
                self.distance_matrix,
                self.evaporation_rate,
            )
            for _ in range(self.number_ants)
        ]

        self.cost = float("inf")

    def calculate_distance(self, city1, city2):
        """
        Calculate the Euclidean distance between two cities.

        Parameters:
        - city1 (City): First city.
        - city2 (City): Second city.

        Returns:
        - distance (float): Euclidean distance between the cities.
        """
        return np.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)

    def generate_distance_matrix(self, cities):
        """
        Generate a distance matrix based on the pairwise distances between cities.

        Args:
        - cities (List[City]): The list of City objects representing the cities.

        Returns:
        - np.ndarray: A 2D NumPy array representing the distance matrix.
        """
        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i, num_cities):
                city1 = cities[i]
                city2 = cities[j]
                distance_matrix[i, j] = distance_matrix[j, i] = self.calculate_distance(
                    city1, city2
                )
        return distance_matrix

    def update_pheromone_matrix(self):
        """
        Update the pheromone matrix based on the tours of the ants and evaporation.

        This method calculates the pheromone deposits from the best tours of ants and applies
        pheromone evaporation to the entire matrix.

        Note: This method is specific to the Ant Colony Optimization (ACO) algorithm.

        Returns:
        - None
        """
        for ant in self.ants:
            pheromone_deposit = 1.0 / ant.cost

            current_city_indices = [self.cities.index(city) for city in ant.best_tour]
            next_city_indices = [
                current_city_indices[(i + 1) % self.number_cities]
                for i in range(len(ant.best_tour))
            ]

            self.pheromone_matrix[
                current_city_indices, next_city_indices
            ] += pheromone_deposit
        self.pheromone_matrix *= 1.0 - self.evaporation_rate

    def fit(self):
        """
        Run the Ant Colony Optimization (ACO) algorithm to find the optimal TSP tour.

        The ACO algorithm iteratively constructs tours using multiple ants and updates
        the pheromone levels on the edges of the cities. This process is repeated for
        a specified number of iterations.

        Returns:
        - cost (float): The total cost (distance) of the optimal tour.
        - tour (List[City]): The sequence of cities representing the optimal tour.
        """
        for i in range(self.max_iter):
            for ant in self.ants:
                ant.pheromone_matrix = self.pheromone_matrix
                ant.generate_tour()

            best_ant = min(self.ants, key=lambda x: x.best_cost)

            if best_ant.best_cost < self.cost:
                self.cost, self.tour = best_ant.best_cost, best_ant.best_tour

            if self.verbose:
                self.plot_tour(iteration=i + 1)

            self.update_pheromone_matrix()

        return self.cost, self.tour
