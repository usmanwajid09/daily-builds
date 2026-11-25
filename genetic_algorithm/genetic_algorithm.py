"""Genetic algorithm optimiser (minimisation), implemented with numpy.

A population of candidate solutions evolves by tournament selection,
uniform crossover and Gaussian mutation, with elitism carrying the best
individuals unchanged into the next generation. Works on any callable
objective over a bounded real-valued search space.
"""

import numpy as np

__all__ = ["GeneticAlgorithm", "OptimizeResult", "sphere", "rastrigin"]


def sphere(x):
    """Sum of squares; global minimum 0 at the origin."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"sphere expects a 1-D vector, got shape {x.shape}")
    return float(np.sum(x**2))


def rastrigin(x):
    """Rastrigin function; global minimum 0 at the origin, many local minima."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"rastrigin expects a 1-D vector, got shape {x.shape}")
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


class OptimizeResult:
    """Result of a genetic-algorithm run."""

    def __init__(self, best_solution, best_fitness, history):
        self.best_solution = np.asarray(best_solution, dtype=float)
        self.best_fitness = float(best_fitness)
        self.history = list(history)  # best fitness per generation

    def __repr__(self):
        return (
            f"OptimizeResult(best_fitness={self.best_fitness:.6g}, "
            f"generations={len(self.history)})"
        )


class GeneticAlgorithm:
    """Minimise ``objective`` over a box-bounded continuous search space."""

    def __init__(
        self,
        objective,
        bounds,
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        tournament_size=3,
        elitism=1,
        random_state=None,
    ):
        if not callable(objective):
            raise TypeError("objective must be callable")
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                f"bounds must have shape (n_dim, 2), got {bounds.shape}"
            )
        if bounds.shape[0] < 1:
            raise ValueError("bounds must define at least one dimension")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("bounds must be finite")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each bound needs low < high")
        if not isinstance(population_size, (int, np.integer)) or population_size < 4:
            raise ValueError(
                f"population_size must be an integer >= 4, got {population_size!r}"
            )
        if not isinstance(generations, (int, np.integer)) or generations < 1:
            raise ValueError(
                f"generations must be an integer >= 1, got {generations!r}"
            )
        for name, rate in (
            ("mutation_rate", mutation_rate),
            ("crossover_rate", crossover_rate),
        ):
            if not isinstance(rate, (int, float, np.floating)) or not (
                0.0 <= float(rate) <= 1.0
            ):
                raise ValueError(f"{name} must be between 0 and 1, got {rate!r}")
        if (
            not isinstance(tournament_size, (int, np.integer))
            or tournament_size < 2
            or tournament_size > population_size
        ):
            raise ValueError(
                "tournament_size must be an integer in [2, population_size], "
                f"got {tournament_size!r}"
            )
        if (
            not isinstance(elitism, (int, np.integer))
            or elitism < 0
            or elitism >= population_size
        ):
            raise ValueError(
                "elitism must be an integer in [0, population_size), "
                f"got {elitism!r}"
            )

        self.objective = objective
        self.bounds = bounds
        self.n_dim = bounds.shape[0]
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.tournament_size = int(tournament_size)
        self.elitism = int(elitism)
        self.rng = np.random.default_rng(random_state)
        self.low = bounds[:, 0]
        self.high = bounds[:, 1]

    def _fitness(self, population):
        values = np.empty(population.shape[0])
        for i, individual in enumerate(population):
            value = float(self.objective(individual))
            if not np.isfinite(value):
                raise ValueError(
                    f"objective returned a non-finite value {value!r}"
                )
            values[i] = value
        return values

    def _tournament(self, population, fitness):
        idx = self.rng.integers(0, population.shape[0], size=self.tournament_size)
        return population[idx[np.argmin(fitness[idx])]]

    def _crossover(self, parent_a, parent_b):
        if self.rng.random() > self.crossover_rate:
            return parent_a.copy()
        mask = self.rng.random(self.n_dim) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child

    def _mutate(self, individual):
        sigma = 0.1 * (self.high - self.low)
        mask = self.rng.random(self.n_dim) < self.mutation_rate
        noise = self.rng.normal(0.0, sigma)
        individual = np.where(mask, individual + noise, individual)
        return np.clip(individual, self.low, self.high)

    def run(self):
        """Evolve the population and return an :class:`OptimizeResult`."""
        population = self.rng.uniform(
            self.low, self.high, size=(self.population_size, self.n_dim)
        )
        history = []
        best_solution = None
        best_fitness = np.inf
        for _ in range(self.generations):
            fitness = self._fitness(population)
            order = np.argsort(fitness)
            if fitness[order[0]] < best_fitness:
                best_fitness = float(fitness[order[0]])
                best_solution = population[order[0]].copy()
            history.append(best_fitness)
            next_population = []
            for i in range(self.elitism):
                next_population.append(population[order[i]].copy())
            while len(next_population) < self.population_size:
                parent_a = self._tournament(population, fitness)
                parent_b = self._tournament(population, fitness)
                child = self._mutate(self._crossover(parent_a, parent_b))
                next_population.append(child)
            population = np.array(next_population)
        return OptimizeResult(best_solution, best_fitness, history)
