import pytest
from ..city import get_cities_csv
from ..genetics import Chromosome, GeneticsAlgorithm


@pytest.fixture
def cities():
    """
    Fixture that returns a list of cities loaded from a CSV file.

    Returns:
        List[City]: A list of City objects representing cities.
    """
    return get_cities_csv()


@pytest.fixture
def tsp_genetics_return(cities):
    """
    Fixture that initializes a GeneticsAlgorithm for TSP with return-to-origin.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        GeneticsAlgorithm: An instance of GeneticsAlgorithm for the TSP with return-to-origin.
    """
    return GeneticsAlgorithm(cities, verbose=False, return_to_city=True)


@pytest.fixture
def tsp_genetics_no_return(cities):
    """
    Fixture that initializes a GeneticsAlgorithm for TSP without return-to-origin.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        GeneticsAlgorithm: An instance of GeneticsAlgorithm for the TSP without return-to-origin.
    """
    return GeneticsAlgorithm(cities, verbose=False, return_to_city=False)


@pytest.fixture
def population(tsp_genetics_return):
    """
    Fixture that creates an initial population for the GeneticsAlgorithm.

    Args:
        tsp_genetics_return (GeneticsAlgorithm): A GeneticsAlgorithm instance.

    Returns:
        List[Chromosome]: A list of Chromosome objects representing the initial population.
    """
    return tsp_genetics_return.create_initial_population()


@pytest.fixture
def sample_chromosomes(cities):
    """
    Fixture that provides sample Chromosome objects.

    Args:
        cities (List[City]): A list of City objects representing cities.

    Returns:
        Tuple: A tuple of sample Chromosome objects for testing.
    """
    genes_1 = cities
    genes_2 = cities[::-1]
    chromosome_1 = Chromosome(genes_1)
    chromosome_2 = Chromosome(genes_2)
    return chromosome_1, chromosome_2


def test_create_initial_population(tsp_genetics_return, population):
    """
    Test the creation of the initial population by the GeneticsAlgorithm.

    Args:
        tsp_genetics_return (GeneticsAlgorithm): A GeneticsAlgorithm instance.
        population (List[Chromosome]): The initial population.

    This test checks that the initial population has the expected size.
    """
    assert len(population) == tsp_genetics_return.pop_size


def test_tours_no_return(tsp_genetics_no_return, population):
    """
    Test tours for the TSP without return-to-origin.

    Args:
        tsp_genetics_no_return (GeneticsAlgorithm): A GeneticsAlgorithm instance.
        population (List[Chromosome]): The initial population.

    This test verifies that all tours in the population for the TSP without return-to-origin
    include all cities exactly once.
    """
    assert all(
        len(set(chromosome.genes)) == tsp_genetics_no_return.number_cities
        for chromosome in population
    )


def test_tours_return(tsp_genetics_return, population):
    """
    Test tours for the TSP with return-to-origin.

    Args:
        tsp_genetics_return (GeneticsAlgorithm): A GeneticsAlgorithm instance.
        population (List[Chromosome]): The initial population.

    This test verifies that all tours in the population for the TSP with return-to-origin
    start and end at the same city and include all cities exactly once.
    """
    assert all(
        chromosome.genes[0] == chromosome.genes[-1]
        and len(set(chromosome.genes[:-1])) == tsp_genetics_return.number_cities
        for chromosome in population
    )


def test_elitism(tsp_genetics_return, population):
    """
    Test elitism in the GeneticsAlgorithm.

    Args:
        tsp_genetics_return (GeneticsAlgorithm): A GeneticsAlgorithm instance.
        population (List[Chromosome]): The initial population.

    This test checks that the elitism method returns elites with higher fitness.
    """
    elites = tsp_genetics_return.elitism(population)
    assert len(elites) == tsp_genetics_return.elitism_size
    assert all(
        elite.fitness >= elites[i + 1].fitness for i, elite in enumerate(elites[:-1])
    )


def test_partially_mapped_crossover(tsp_genetics_return, sample_chromosomes):
    """
    Test partially mapped crossover in the GeneticsAlgorithm.

    Args:
        tsp_genetics_return (GeneticsAlgorithm): A GeneticsAlgorithm instance.
        sample_chromosomes (Tuple): Sample Chromosome objects for testing.

    This test ensures that the partially mapped crossover method results in child Chromosome objects
    with the same number of genes, but different gene sequences. Inside the implementation of that method
    it tests that resulting genes are arranged correctly.
    """
    parent_1, parent_2 = sample_chromosomes
    child_1, child_2 = tsp_genetics_return.partially_mapped_crossover(
        parent_1, parent_2
    )

    assert len(child_1.genes) == len(parent_1.genes)
    assert len(child_2.genes) == len(parent_2.genes)
    assert set(child_1.genes) == set(parent_1.genes)
    assert set(child_2.genes) == set(parent_2.genes)

    assert child_1.genes != parent_1.genes
    assert child_2.genes != parent_2.genes


def test_mutation(tsp_genetics_no_return, sample_chromosomes):
    """
    Test mutation in the GeneticsAlgorithm.

    Args:
        tsp_genetics_no_return (GeneticsAlgorithm): A GeneticsAlgorithm instance.
        sample_chromosomes (Tuple): Sample Chromosome objects for testing.

    This test verifies that the mutation results in a Chromosome with genes that are not
    identical to the parent Chromosome.
    """
    parent_1 = sample_chromosomes[0]
    child_1 = tsp_genetics_no_return.mutate_gene(parent_1)

    assert parent_1.genes != child_1.genes
    assert (
        sum(
            child_gene == parent_gene
            for parent_gene, child_gene in zip(parent_1.genes, child_1.genes)
        )
        == tsp_genetics_no_return.number_cities - 2
    )


def test_tsp_genetics_return(tsp_genetics_return):
    """
    Test the GeneticsAlgorithm solver with the return_to_city option.

    Parameters:
    - tsp_genetics_return (GeneticsAlgorithm): An instance of the GeneticsAlgorithm solver.

    It checks if the tour starts and ends at the starting city.
    """
    _, tour = tsp_genetics_return.fit()
    assert tour[0] == tour[-1] == tsp_genetics_return.start_city


def test_tsp_genetics_no_return(tsp_genetics_no_return):
    """
    Test the GeneticsAlgorithm solver when the return_to_city option is disabled

    Parameters:
    - tsp_genetics_no_return (GeneticsAlgorithm): An instance of the GeneticsAlgorithm solver.

    It checks if the tour starts at the starting city and visits all cities exactly once.
    """
    _, tour = tsp_genetics_no_return.fit()
    assert tour[0] == tsp_genetics_no_return.start_city
    assert len(set(tour)) == tsp_genetics_no_return.number_cities
