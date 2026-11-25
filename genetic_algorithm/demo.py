"""Minimise the 5-D sphere function with a genetic algorithm.

Runs tournament selection + uniform crossover + Gaussian mutation for 100
generations and prints the best fitness found, the best solution, and how
the best fitness improved over the run.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from genetic_algorithm import GeneticAlgorithm, sphere

N_DIM = 5
BOUNDS = [(-5.0, 5.0)] * N_DIM
SEED = 42


def main():
    ga = GeneticAlgorithm(
        sphere,
        bounds=BOUNDS,
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        tournament_size=3,
        elitism=1,
        random_state=SEED,
    )
    result = ga.run()
    print(f"Objective: sphere in {N_DIM}D (global minimum 0 at the origin)")
    print(f"Population 50, generations 100, seed {SEED}\n")
    print(f"Best fitness:  {result.best_fitness:.6f}")
    print(f"Best solution: {np.round(result.best_solution, 4).tolist()}")
    print(f"Distance to optimum: {float(np.linalg.norm(result.best_solution)):.4f}")
    print("\nBest fitness by generation:")
    for gen in (0, 9, 24, 49, 74, 99):
        print(f"  gen {gen + 1:>3}: {result.history[gen]:.6f}")


if __name__ == "__main__":
    main()
