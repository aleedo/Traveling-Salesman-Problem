from ..nn import NearestNeighbor
from ..genetics import GeneticsAlgorithm
from ..antcolony import AntColonyOptimization
from .base import _TSP


class TSP(_TSP):
    def __init__(
        self,
        cities,
        *args,
        **kwargs,
    ):
        """
        Initialize the Traveling Salesman Problem (TSP) solver.

        Parameters:
        - cities (List[City]): A list of city objects representing the cities to be visited.
        - start_city (City, optional): The starting city for the TSP tour. If not provided, a random city is chosen as the start.
        - return_to_city (bool): Whether to return to the starting city at the end of the tour. Default: False.
        - solver (str): The solver algorithm to use. Options: "nn" (Nearest Neighbor), "genetics" (Genetic Algorithm), "antcolony" (Ant Colony Optimization). Default: "nn".
        - random_state (int, optional): Seed for the random number generator, providing reproducibility. Default: None.
        - verbose (bool): Whether to enable verbose mode for displaying tour plots. Default: True.

        Genetic Algorithm Parameters (for solver="genetics"):
        - pop_size (int): The population size for the genetic algorithm. Default: 100.
        - generations_count (int): The number of generations to run the genetic algorithm. Default: 10.
        - elitism_frac (float): Fraction of the population that is preserved as elite individuals. Default: 0.01.
        - crossover_prop (float): Probability of performing crossover (recombination) in each generation. Default: 0.7.
        - mutation_prop (float): Probability of performing mutation in each generation. Default: 0.8.
        - selection_type (str): The type of selection method for parents. Options: "tournament", "random". Default: "tournament".
        - k_tournament_selection (int): The number of individuals in each tournament for selection. Default: 10.

        Ant Colony Optimization Parameters (for solver="antcolony"):
        - number_ants (int): The number of ants in the ant colony. Default: 15.
        - alpha (float): The alpha parameter controlling the influence of pheromone levels in ant decisions. Default: 2.
        - beta (float): The beta parameter controlling the influence of distance in ant decisions. Default: 7.
        - evaporation_rate (float): The rate at which pheromone evaporates on each iteration. Default: 0.1.
        - max_iter (int): The maximum number of iterations for the ant colony algorithm. Default: 10.
        """

        super(TSP, self).__init__(
            cities,
            *args,
            **kwargs,
        )

        self.solvers = {
            "nn": NearestNeighbor(
                self.cities,
                *args,
                **kwargs,
            ),
            "genetics": GeneticsAlgorithm(
                self.cities,
                *args,
                **kwargs,
            ),
            "antcolony": AntColonyOptimization(self.cities, *args, **kwargs),
        }

    def fit(self):
        """
        Run the TSP solver to find the optimal tour.

        Returns:
        - cost (float): The total cost (distance) of the optimal tour.
        - tour (List[City]): The sequence of cities representing the optimal tour.
        """
        self.cost, self.tour = self.solvers[self.solver].fit()
        return self.cost, self.tour
