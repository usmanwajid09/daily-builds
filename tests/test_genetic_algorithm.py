import numpy as np
import pytest

from genetic_algorithm import GeneticAlgorithm, OptimizeResult, rastrigin, sphere


def _run_ga(**overrides):
    kwargs = dict(
        objective=sphere,
        bounds=[(-5.0, 5.0)] * 5,
        population_size=50,
        generations=100,
        random_state=42,
    )
    kwargs.update(overrides)
    return GeneticAlgorithm(**kwargs).run()


def test_sphere_minimum_at_origin():
    assert sphere([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert sphere([1.0, 2.0, 2.0]) == pytest.approx(9.0)


def test_rastrigin_minimum_at_origin():
    assert rastrigin([0.0, 0.0]) == pytest.approx(0.0)
    assert rastrigin([1.0, 0.0]) > 0.0


def test_sphere_rejects_non_vector():
    with pytest.raises(ValueError, match="1-D"):
        sphere([[1.0, 2.0], [3.0, 4.0]])


def test_ga_minimises_sphere_near_zero():
    result = _run_ga()
    assert result.best_fitness < 0.01  # headline number from the demo
    assert np.linalg.norm(result.best_solution) < 0.1


def test_best_solution_stays_inside_bounds():
    result = _run_ga()
    assert np.all(result.best_solution >= -5.0)
    assert np.all(result.best_solution <= 5.0)


def test_history_is_monotonic_and_matches_generations():
    result = _run_ga()
    assert len(result.history) == 100
    assert all(
        result.history[i + 1] <= result.history[i] for i in range(len(result.history) - 1)
    )
    assert result.history[-1] == pytest.approx(result.best_fitness)


def test_deterministic_with_same_seed():
    result_a = _run_ga()
    result_b = _run_ga()
    assert result_a.best_fitness == pytest.approx(result_b.best_fitness)
    assert np.allclose(result_a.best_solution, result_b.best_solution)


def test_result_repr():
    result = OptimizeResult([0.1, 0.2], 0.05, [1.0, 0.05])
    assert "0.05" in repr(result)


def test_ga_handles_1d_problem():
    result = _run_ga(bounds=[(-5.0, 5.0)], generations=50)
    assert result.best_fitness < 0.01
    assert result.best_solution.shape == (1,)


def test_ga_minimises_rastrigin():
    result = _run_ga(objective=rastrigin, bounds=[(-5.12, 5.12)] * 2, generations=150)
    assert result.best_fitness < 1.0


def test_non_callable_objective_raises():
    with pytest.raises(TypeError, match="callable"):
        GeneticAlgorithm(objective=42, bounds=[(-1.0, 1.0)])


def test_bad_bounds_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        GeneticAlgorithm(objective=sphere, bounds=[-1.0, 1.0, 2.0])


def test_inverted_bounds_raise():
    with pytest.raises(ValueError, match="low < high"):
        GeneticAlgorithm(objective=sphere, bounds=[(1.0, -1.0)])


def test_small_population_raises():
    with pytest.raises(ValueError, match="population_size"):
        GeneticAlgorithm(objective=sphere, bounds=[(-1.0, 1.0)], population_size=3)


def test_bad_rates_raise():
    with pytest.raises(ValueError, match="mutation_rate"):
        GeneticAlgorithm(objective=sphere, bounds=[(-1.0, 1.0)], mutation_rate=1.5)
    with pytest.raises(ValueError, match="crossover_rate"):
        GeneticAlgorithm(objective=sphere, bounds=[(-1.0, 1.0)], crossover_rate=-0.1)


def test_bad_tournament_size_raises():
    with pytest.raises(ValueError, match="tournament_size"):
        GeneticAlgorithm(
            objective=sphere, bounds=[(-1.0, 1.0)], population_size=10, tournament_size=11
        )


def test_bad_elitism_raises():
    with pytest.raises(ValueError, match="elitism"):
        GeneticAlgorithm(
            objective=sphere, bounds=[(-1.0, 1.0)], population_size=10, elitism=10
        )


def test_non_finite_objective_raises():
    ga = GeneticAlgorithm(
        objective=lambda x: float("nan"),
        bounds=[(-1.0, 1.0)],
        population_size=10,
        generations=1,
        random_state=0,
    )
    with pytest.raises(ValueError, match="non-finite"):
        ga.run()
