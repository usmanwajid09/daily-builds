# Genetic Algorithm

A genetic algorithm optimiser (minimisation) implemented from scratch
with numpy. It evolves a population of candidate solutions using
**tournament selection**, **uniform crossover**, **Gaussian mutation**,
and **elitism**, and works on any callable objective over a bounded
real-valued search space.

## How it works

1. **Initialise** — sample the population uniformly inside the bounds.
2. **Evaluate** — score every individual with the objective (lower is better).
3. **Select** — tournament selection: draw a few random individuals and
   keep the best, so fitter individuals are more likely to breed.
4. **Crossover** — with probability `crossover_rate`, mix two parents
   gene-by-gene (uniform crossover); otherwise copy a parent.
5. **Mutate** — with probability `mutation_rate` per gene, add Gaussian
   noise (sigma = 10% of the bound width) and clip back into bounds.
6. **Elitism** — the best `elitism` individuals pass unchanged into the
   next generation, so the best fitness never gets worse.

`run()` returns an `OptimizeResult` with `best_solution`,
`best_fitness`, and `history` (best fitness per generation). Two test
functions are included: `sphere` (sum of squares) and `rastrigin`
(many local minima), both with a global minimum of 0 at the origin.

## Usage

```python
from genetic_algorithm import GeneticAlgorithm, sphere

ga = GeneticAlgorithm(
    sphere,
    bounds=[(-5.0, 5.0)] * 5,
    population_size=50,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=3,
    elitism=1,
    random_state=42,
)
result = ga.run()
print(result.best_fitness)    # close to 0
print(result.best_solution)   # close to the origin
```

Bad inputs (non-callable objective, bounds with the wrong shape or
`low >= high`, population < 4, rates outside [0, 1], tournament bigger
than the population, elitism >= population) raise `ValueError`/
`TypeError` with a clear message.

## Run the demo

```bash
python genetic_algorithm/demo.py
```

Minimises the 5-D sphere function for 100 generations and prints the
best fitness (well under 0.01) and the best solution found.

## Run the tests

```bash
pytest tests/test_genetic_algorithm.py -q
```
