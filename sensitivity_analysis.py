"""Small one-factor-at-a-time hyperparameter sensitivity experiment.
Run: python sensitivity_analysis.py
This script tests settings on fewer generations to keep runtime manageable.
"""
import main
import statistics
import json
from pathlib import Path

def experiment(population_size, mutation_rate, tournament_size, seeds=(11, 22, 33)):
    original = (main.POPULATION_SIZE, main.MUTATION_RATE, main.TOURNAMENT_SIZE, main.GENERATIONS, main.SEED)
    values = []
    try:
        main.POPULATION_SIZE = population_size
        main.MUTATION_RATE = mutation_rate
        main.TOURNAMENT_SIZE = tournament_size
        main.GENERATIONS = 35
        for seed in seeds:
            import random, numpy as np
            random.seed(seed); np.random.seed(seed)
            best, _, _ = main.run_ga()
            values.append(main.evaluate(best)[0])
    finally:
        (main.POPULATION_SIZE, main.MUTATION_RATE, main.TOURNAMENT_SIZE,
         main.GENERATIONS, main.SEED) = original
    return {"population": population_size, "mutation_rate": mutation_rate,
            "tournament_size": tournament_size, "runs": len(values),
            "mean_final_fitness": round(statistics.mean(values), 3),
            "stdev_final_fitness": round(statistics.stdev(values), 3) if len(values)>1 else 0.0}

if __name__ == "__main__":
    settings = [
        (30, 0.12, 4), (60, 0.12, 4), (100, 0.12, 4),
        (60, 0.05, 4), (60, 0.25, 4),
        (60, 0.12, 2), (60, 0.12, 7),
    ]
    results = [experiment(*setting) for setting in settings]
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/sensitivity_results.json").write_text(json.dumps(results, indent=2))
    for row in results:
        print(row)
