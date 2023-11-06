import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D


class _TSP:
    def __init__(
        self,
        cities,
        start_city=None,
        return_to_city=False,
        solver="nn",
        random_state=None,
        verbose=True,
        pop_size=100,
        generations_count=10,
        elitism_frac=0.01,
        crossover_prop=0.7,
        mutation_prop=0.8,
        selection_type="tournament",
        k_tournament_selection=10,
        number_ants=15,
        alpha=2,
        beta=7,
        evaporation_rate=0.1,
        max_iter=5,
    ):
        self.cities = cities
        self.number_cities = len(cities)
        self.random_state = np.random.seed(random_state) if random_state else None
        self.start_city = start_city if start_city else np.random.choice(self.cities)
        self.solver = solver
        self.return_to_city = return_to_city
        self.verbose = verbose

        ## NearestNeighbor Implementation
        pass

        ## GeneticsAlgorithm Implementation
        self.pop_size = pop_size
        self.generations_count = generations_count
        self.elitism_frac = elitism_frac
        self.selection_type = selection_type
        self.crossover_prop = crossover_prop
        self.mutation_prop = mutation_prop
        self.k_tournament_selection = k_tournament_selection

        ## AntColonyOptimization Implementation
        self.number_ants = number_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.max_iter = max_iter

    def plot_tour(self, tour=None, iteration=None):
        """
        Plot the tour using Matplotlib.

        Args:
            tour (List[City]): Tour sequence to be plotted (optional).
            iteration (int): Iteration or generation count for labeling the plot (optional).
        """
        if tour:
            self.tour = tour

        max_iterations = {
            "nn": self.number_cities + int(self.return_to_city),
            "genetics": self.generations_count,
            "antcolony": self.max_iter,
        }[self.solver]

        iteration = iteration if iteration else max_iterations

        subtitle = {
            "genetics": f"{iteration} generations",
            "antcolony": f"{iteration} iterations",
            "nn": f"{len(self.tour) - int(self.return_to_city)} cities",
        }

        circle_color_mapping = {0: "r", len(self.tour) - 1: "g", "default": "k"}

        legend_mapping = {"r": "Start City", "g": "Arrival City", "k": "Other Cities"}

        title = f"TSP using {self.__class__.__name__}: After {subtitle[self.solver]} with cost = {self.cost:.2f}"

        x, y = zip(*map(lambda elem: (elem.x, elem.y), self.tour))

        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots(figsize=(20, 8))
            plt.ion()

        self.ax.clear()

        legend_elements = []

        for i in range(len(self.tour)):
            city = self.tour[i]

            circle_color = circle_color_mapping.get(i, circle_color_mapping["default"])

            self.ax.add_patch(plt.Circle((x[i], y[i]), 2, color=circle_color))
            self.ax.text(
                x[i],
                y[i],
                str(city.name),
                ha="center",
                va="center",
                color="w",
                weight="bold",
            )

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"City {city.name}",
                    markersize=8,
                    markerfacecolor=circle_color,
                )
            )

        for i in range(len(self.tour) - 1):
            start = (x[i], y[i])
            end = (x[i + 1], y[i + 1])

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            arrow_count = int(arrow_length / 3)

            if arrow_count > 0:
                dx /= arrow_count
                dy /= arrow_count

                for j in range(arrow_count):
                    arrow = FancyArrowPatch(
                        (start[0] + j * dx, start[1] + j * dy),
                        (start[0] + (j + 1) * dx, start[1] + (j + 1) * dy),
                        arrowstyle="->",
                        mutation_scale=20,
                        color="k",
                    )
                    self.ax.add_patch(arrow)

        self.ax.set_xlim(min(x) - 5, max(x) + 5)
        self.ax.set_ylim(min(y) - 5, max(y) + 5)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.ax.set_title(title)

        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markersize=10,
                markerfacecolor=color,
                label=label,
            )
            for color, label in legend_mapping.items()
        ]

        self.ax.legend(handles=legend_handles, loc="best")

        self.ax.grid(False)
        plt.draw()
        plt.pause(0.5)

        if iteration == max_iterations:
            plt.ioff()
            plt.show()
