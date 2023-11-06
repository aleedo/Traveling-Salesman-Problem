from tsp import TSP
from tsp.city import get_cities_csv


def main():
    cities = get_cities_csv()
    start_city = cities[0]

    tsp_return = TSP(
        cities,
        start_city=cities[0],
        return_to_city=False,
        verbose=True,
        solver="genetics",
        random_state=10,
    )

    cost, tour = tsp_return.fit()
    print(f"cost {cost}, tour : {tour}")


if __name__ == "__main__":
    main()
