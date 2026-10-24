"""
Genetic Algorithm for Multi-UAV Disaster Surveillance
Run: python main.py
"""
from dataclasses import dataclass
import random
import math
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRID_SIZE = 30
NUM_DRONES = 3
WAYPOINTS_PER_DRONE = 7
POPULATION_SIZE = 60
GENERATIONS = 100
MUTATION_RATE = 0.12
TOURNAMENT_SIZE = 4
ELITE_COUNT = 3
MAX_ROUTE_LENGTH = 42.0

# Map cells are (x, y). Hazards have different risk weights.
NO_FLY_ZONES = {(8, y) for y in range(5, 18)} | {(x, 21) for x in range(16, 25)}
HAZARDS = {(x, y): 1.0 for x in range(10, 17) for y in range(10, 16)}
HAZARDS.update({(x, y): 0.6 for x in range(20, 26) for y in range(4, 10)})
STARTS = [(2, 2), (2, 27), (27, 2)]
PRIORITY_ZONES = {
    (x, y): 2.0 if 0 <= x < 10 else 1.0
    for x in range(3, 27, 2) for y in range(3, 28, 2)
}
TARGETS = [(5, 5), (12, 7), (22, 13), (25, 25), (15, 24), (4, 20)]

@dataclass
class DronePlan:
    """A chromosome contains waypoints, speed multipliers and sensor powers."""
    waypoints: list
    speeds: list
    sensors: list

    def clone(self):
        return DronePlan(
            [list(p) for p in self.waypoints],
            list(self.speeds),
            list(self.sensors)
        )

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def line_cells(a, b, samples=None):
    """Approximate cells crossed by a segment using evenly spaced samples."""
    if samples is None:
        samples = max(2, int(distance(a, b) * 2))
    cells = set()
    for i in range(samples + 1):
        t = i / samples
        x = round(a[0] + (b[0] - a[0]) * t)
        y = round(a[1] + (b[1] - a[1]) * t)
        cells.add((x, y))
    return cells

def random_plan():
    plans = []
    for start in STARTS:
        points = []
        for _ in range(WAYPOINTS_PER_DRONE):
            points.append([random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)])
        plans.append(DronePlan(points,
            [random.uniform(0.75, 1.25) for _ in points],
            [random.uniform(0.35, 1.0) for _ in points]))
    return plans

def route_for(drone_index, plan):
    return [STARTS[drone_index]] + [tuple(p) for p in plan.waypoints]

def evaluate(individual):
    """Return scalar fitness and interpretable objective metrics."""
    covered = set()
    energy, flight_time, hazard_exposure, violations = 0.0, 0.0, 0.0, 0
    total_distance = 0.0
    for d, plan in enumerate(individual):
        route = route_for(d, plan)
        for idx, (a, b) in enumerate(zip(route[:-1], route[1:])):
            seg_len = distance(a, b)
            total_distance += seg_len
            speed = plan.speeds[idx]
            sensor = plan.sensors[idx]
            # Faster flight takes less time but increases energy; sensor power costs energy.
            energy += seg_len * (0.65 + 0.35 * speed**2) + seg_len * 0.45 * sensor
            flight_time += seg_len / max(speed, 0.1)
            cells = line_cells(a, b)
            for cell in cells:
                if cell in NO_FLY_ZONES:
                    violations += 1
                hazard_exposure += HAZARDS.get(cell, 0.0)
                # A powered sensor detects targets within radius 2.5*power+1.
                radius = 1.0 + 2.5 * sensor
                for target in TARGETS:
                    if distance(cell, target) <= radius:
                        covered.add(target)
    coverage_score = sum(2.0 if t in TARGETS[:3] else 1.0 for t in covered)
    # Penalize long paths, hazard exposure and restricted-airspace intersections.
    range_violations = sum(max(0.0, sum(distance(a,b) for a,b in zip(route_for(i,p)[:-1], route_for(i,p)[1:])) - MAX_ROUTE_LENGTH)
                           for i,p in enumerate(individual))
    fitness = (coverage_score * 18.0
               - energy * 0.30
               - flight_time * 0.12
               - hazard_exposure * 1.8
               - violations * 30.0
               - range_violations * 8.0)
    return fitness, {
        "coverage": len(covered), "coverage_score": coverage_score,
        "energy": energy, "flight_time": flight_time,
        "hazard_exposure": hazard_exposure, "no_fly_violations": violations,
        "range_violation": range_violations, "distance": total_distance
    }

def tournament(population, scores):
    candidates = random.sample(range(len(population)), TOURNAMENT_SIZE)
    winner = max(candidates, key=lambda i: scores[i][0])
    return population[winner]

def crossover(parent1, parent2):
    """Swap complete waypoint/speed/sensor segments independently per drone."""
    child = []
    for p1, p2 in zip(parent1, parent2):
        cut1 = random.randint(1, WAYPOINTS_PER_DRONE-1)
        cut2 = random.randint(cut1, WAYPOINTS_PER_DRONE)
        points = [list(x) for x in p1.waypoints[:cut1]] + [list(x) for x in p2.waypoints[cut1:]]
        speeds = p1.speeds[:cut1] + p2.speeds[cut1:]
        sensors = p1.sensors[:cut1] + p2.sensors[cut1:]
        # Occasionally use a second segment from the other parent.
        if cut2 < WAYPOINTS_PER_DRONE:
            points[cut2:] = [list(x) for x in p1.waypoints[cut2:]]
            speeds[cut2:] = p1.speeds[cut2:]
            sensors[cut2:] = p1.sensors[cut2:]
        child.append(DronePlan(points, speeds, sensors))
    return child

def mutate(individual):
    """Spatial mutation perturbs coordinates; bounded real-valued mutation for speed/sensor."""
    child = [p.clone() for p in individual]
    for plan in child:
        for i in range(WAYPOINTS_PER_DRONE):
            if random.random() < MUTATION_RATE:
                plan.waypoints[i][0] = max(0, min(GRID_SIZE-1,
                    plan.waypoints[i][0] + random.randint(-5, 5)))
                plan.waypoints[i][1] = max(0, min(GRID_SIZE-1,
                    plan.waypoints[i][1] + random.randint(-5, 5)))
            if random.random() < MUTATION_RATE:
                plan.speeds[i] = max(0.6, min(1.4, plan.speeds[i] + random.gauss(0, 0.12)))
            if random.random() < MUTATION_RATE:
                plan.sensors[i] = max(0.2, min(1.0, plan.sensors[i] + random.gauss(0, 0.12)))
    return child

def repair(individual):
    """Repair coordinates outside grid and clamp controls to their legal ranges."""
    for plan in individual:
        for point in plan.waypoints:
            point[0] = int(max(0, min(GRID_SIZE-1, point[0])))
            point[1] = int(max(0, min(GRID_SIZE-1, point[1])))
        plan.speeds = [max(0.6, min(1.4, s)) for s in plan.speeds]
        plan.sensors = [max(0.2, min(1.0, s)) for s in plan.sensors]
    return individual

def run_ga():
    population = [random_plan() for _ in range(POPULATION_SIZE)]
    best_history, avg_history = [], []
    best_ever, best_ever_score = None, -float("inf")
    for generation in range(GENERATIONS):
        scored = [evaluate(ind) for ind in population]
        order = sorted(range(len(population)), key=lambda i: scored[i][0], reverse=True)
        scores = [scored[i][0] for i in range(len(population))]
        best_idx = order[0]
        if scores[best_idx] > best_ever_score:
            best_ever_score = scores[best_idx]
            best_ever = [p.clone() for p in population[best_idx]]
        best_history.append(scores[best_idx])
        avg_history.append(float(np.mean(scores)))
        next_pop = [[p.clone() for p in population[i]] for i in order[:ELITE_COUNT]]
        while len(next_pop) < POPULATION_SIZE:
            p1, p2 = tournament(population, scored), tournament(population, scored)
            child = crossover(p1, p2)
            child = repair(mutate(child))
            next_pop.append(child)
        population = next_pop
    return best_ever, best_history, avg_history

def plot_results(best, best_history, avg_history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.plot(best_history, label="Best fitness")
    plt.plot(avg_history, label="Average fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness"); plt.title("GA Fitness Convergence")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(output_dir/"fitness_convergence.png", dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    for x, y in NO_FLY_ZONES:
        ax.scatter(x, y, marker="s", s=35, color="black")
    if HAZARDS:
        hx, hy = zip(*HAZARDS.keys())
        ax.scatter(hx, hy, marker="s", s=18, color="orange", alpha=0.45, label="Hazard")
    colors = ["tab:blue", "tab:green", "tab:red"]
    for i, plan in enumerate(best):
        route = route_for(i, plan)
        xs, ys = zip(*route)
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[i], label=f"Drone {i+1}")
        ax.scatter(*STARTS[i], marker="*", s=180, color=colors[i], edgecolor="black")
    tx, ty = zip(*TARGETS)
    ax.scatter(tx, ty, marker="X", s=90, color="purple", label="Targets")
    ax.set_xlim(-1, GRID_SIZE); ax.set_ylim(-1, GRID_SIZE)
    ax.set_xlabel("X grid coordinate"); ax.set_ylabel("Y grid coordinate")
    ax.set_title("Best Evolved UAV Routes"); ax.grid(alpha=0.2); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(output_dir/"uav_routes.png", dpi=160); plt.close(fig)

    # Coverage heatmap based on cells traversed by the best candidate.
    heat = np.zeros((GRID_SIZE, GRID_SIZE))
    for i, plan in enumerate(best):
        route = route_for(i, plan)
        for a, b in zip(route[:-1], route[1:]):
            for x, y in line_cells(a, b):
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    heat[y, x] += 1
    plt.figure(figsize=(7, 6))
    plt.imshow(heat, origin="lower", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(label="Number of drone route traversals")
    plt.xlabel("X grid coordinate"); plt.ylabel("Y grid coordinate")
    plt.title("Route Coverage Heatmap")
    plt.tight_layout(); plt.savefig(output_dir/"coverage_heatmap.png", dpi=160); plt.close()

def main():
    out = Path("outputs")
    best, best_history, avg_history = run_ga()
    fitness, metrics = evaluate(best)
    plot_results(best, best_history, avg_history, out)
    summary = {"best_fitness": fitness, "metrics": metrics,
               "parameters": {"grid_size": GRID_SIZE, "drones": NUM_DRONES,
               "population_size": POPULATION_SIZE, "generations": GENERATIONS,
               "mutation_rate": MUTATION_RATE, "tournament_size": TOURNAMENT_SIZE,
               "elite_count": ELITE_COUNT}}
    (out/"results.json").write_text(json.dumps(summary, indent=2))
    print("Best fitness:", round(fitness, 2))
    print("Metrics:", json.dumps(metrics, indent=2))
    print("Plots and results saved in outputs/")
if __name__ == "__main__":
    main()
