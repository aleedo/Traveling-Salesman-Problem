from ..city import City
from ..src.base import _TSP
from typing import List
import numpy as np


class Chromosome:
    def __init__(self, genes: List[City]):
        """
        Initialize a chromosome representing a tour in the TSP.

        Parameters:
        - genes (List[City]): A list of City objects representing the order of cities in the tour.
        """
        self.genes = genes
        self.cost = self.calculate_cost()
        self.fitness = 1 / self.cost

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

    def calculate_cost(self):
        """
        Calculate the total cost (distance) of the tour.

        Returns:
        - float: The total cost of the tour.
        """
        return sum(
            self.calculate_distance(city, self.genes[i + 1])
            for i, city in enumerate(self.genes[:-1])
        )

    def __repr__(self):
        return f"Genes:\n{self.genes}\nCost = {self.cost}\nFitness= {self.fitness}"

    def __len__(self):
        return len(self.genes)


class GeneticsAlgorithm(_TSP):
    def __init__(self, cities, **kwargs):
        """
        Initialize the TSP solver using a Genetic Algorithm.

        Parameters:
        - cities (List[City]): A list of city objects representing the cities to be visited.
        - **kwargs: Additional keyword arguments, including solver-specific parameters.
        """
        kwargs["solver"] = "genetics"
        super(GeneticsAlgorithm, self).__init__(cities, **kwargs)

        self.elitism_size = int(self.elitism_frac * self.pop_size)

        if self.elitism_size % 2:
            self.elitism_size += 1

        self.crossover_size = self.pop_size - self.elitism_size

    def create_initial_population(self) -> List[Chromosome]:
        """
        Create the initial population of chromosomes.

        Returns:
        - List[Chromosome]: The initial population of chromosomes.
        """
        population_ = []
        for _ in range(self.pop_size):
            gene_sequence = [self.start_city]
            remaining_cities = self.cities.copy()
            remaining_cities.remove(self.start_city)

            gene_sequence += np.random.choice(
                remaining_cities, replace=False, size=self.number_cities - 1
            ).tolist()

            if self.return_to_city:
                gene_sequence += [self.start_city]

            generated_chromosome = Chromosome(gene_sequence)
            population_.append(generated_chromosome)
        return population_

    def elitism(self, __population):
        """
        Select elite chromosomes to be preserved in the next generation.

        Args:
        - __population (List[Chromosome]): The current population.

        Returns:
        - List[Chromosome]: The elite chromosomes to be preserved.
        """
        elites = sorted(__population, key=lambda x: x.fitness, reverse=True)[
            : self.elitism_size
        ]
        return elites

    def selection(self, __population):
        """
        Calls which type of selection method for crossover parents.

        Args:
        - __population (List[Chromosome]): The current population.

        Returns:
        - List[Chromosome]: Two selected parents.
        """
        selection_type = {
            "random": self.__selection_random,
            "tournament": self.__selection_tournament,
        }
        return selection_type[self.selection_type](__population)

    def __selection_random(self, __population):
        """
        Randomly select two parents from the population.

        Args:
        - __population (List[Chromosome]): The current population.

        Returns:
        - List[Chromosome]: Two selected parents.
        """
        selected_parents = np.random.choice(__population, size=2)
        return selected_parents

    def __selection_tournament(self, __population):
        """
        Perform tournament selection to select two parents from the population.

        Args:
        - __population (List[Chromosome]): The current population.

        Returns:
        - List[Chromosome]: Two selected parents.
        """
        __population_ = __population.copy()

        selected_parents = []
        for _ in range(2):
            winner = max(
                np.random.choice(__population_, size=self.k_tournament_selection),
                key=lambda x: x.fitness,
            )
            selected_parents += [winner]
            __population_.remove(winner)
        return selected_parents

    def crossover(self, __new_generation):
        """
        Perform crossover (recombination) to generate a new generation.

        Args:
        - __new_generation (List[Chromosome]): The new generation.

        Returns:
        - List[Chromosome]: The offspring generation after crossover.
        """
        crossover_generation = []

        for _ in range(self.crossover_size // 2):
            parent_1, parent_2 = self.selection(__new_generation)

            if np.random.rand() < self.crossover_prop:
                child_1, child_2 = self.partially_mapped_crossover(parent_1, parent_2)
                crossover_generation += [child_1, child_2]
            else:
                crossover_generation += [parent_1, parent_2]

        return crossover_generation

    @staticmethod
    def partially_mapped_crossover(parent_1: Chromosome, parent_2: Chromosome):
        """
        Perform partially-mapped crossover between two parents to create two children.

        Args:
        - parent_1 (Chromosome): The first parent.
        - parent_2 (Chromosome): The second parent.

        Returns:
        - Tuple[Chromosome, Chromosome]: Two children resulting from crossover.
        """
        child_1 = parent_1.genes.copy()
        child_2 = parent_2.genes.copy()

        i, j = sorted(np.random.choice(range(1, len(child_1) - 1), 2, replace=False))

        for k in range(i, j):
            gene = parent_2.genes[k]
            index = child_1.index(gene)
            child_1[index] = child_1[k]
            child_1[k] = gene

            gene = parent_1.genes[k]
            index = child_2.index(gene)
            child_2[index] = child_2[k]
            child_2[k] = gene

        assert child_1[i:j] == parent_2.genes[i:j]
        assert child_2[i:j] == parent_1.genes[i:j]

        return Chromosome(child_1), Chromosome(child_2)

    def mutate_gene(self, parent: Chromosome):
        """
        Perform mutation on a single chromosome (swap two genes).

        Args:
        - parent (Chromosome): The chromosome to be mutated.

        Returns:
        - Chromosome: The mutated chromosome.
        """
        parent_genes = parent.genes.copy()
        i, j = sorted(
            np.random.choice(
                np.arange(1, self.number_cities - 1), size=2, replace=False
            )
        )
        parent_genes[i], parent_genes[j] = parent_genes[j], parent_genes[i]
        return Chromosome(parent_genes)

    def mutation(self, __new_generation):
        """
        Apply mutation to a new generation.

        Args:
        - __new_generation (List[Chromosome]): The new generation before mutation.

        Returns:
        - List[Chromosome]: The new generation after applying mutation.
        """
        __new_generation_ = __new_generation.copy()
        mutated_generation = []

        for parent in __new_generation_:
            if np.random.rand() < self.mutation_prop:
                mutated_parent = self.mutate_gene(parent)
                mutated_generation.append(mutated_parent)
            else:
                mutated_generation.append(parent)

        return mutated_generation

    def fit(self):
        """
        Run the algorithm to find the optimal tour.

        Returns:
        - cost (float): The total cost (distance) of the optimal tour.
        - tour (List[City]): The sequence of cities representing the optimal tour.
        """
        generated_population = self.create_initial_population()

        solutions = []
        best_final = None

        for i in range(self.generations_count):
            elite_population = self.elitism(generated_population)

            new_generation = self.crossover(generated_population)
            new_generation = self.mutation(new_generation)
            new_generation.extend(elite_population)

            best_chromosome = max(new_generation, key=lambda x: x.fitness)

            if best_final is None or best_chromosome.fitness > best_final.fitness:
                best_final = best_chromosome

            if self.verbose:
                self.tour = best_final.genes
                self.cost = best_final.cost
                self.plot_tour(iteration=i + 1)

            generated_population = new_generation.copy()

        self.tour = best_final.genes
        self.cost = best_final.cost
        return self.cost, self.tour
