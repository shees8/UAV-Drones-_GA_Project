# UAV Disaster Surveillance using a Genetic Algorithm

## Overview
This educational simulation uses a Genetic Algorithm (GA) to plan three UAV routes on a 30 x 30 grid. Each drone chromosome encodes waypoint coordinates, speed multipliers, and sensor-power levels. The fitness function rewards target coverage and penalizes energy, flight time, hazard exposure, no-fly-zone intersections, and route-range violations.

## Requirements
- Python 3.10+
- Packages in `requirements.txt`

## Run
```bash
python -m pip install -r requirements.txt
python main.py
```
Outputs are written to `outputs/`:
- `fitness_convergence.png`
- `uav_routes.png`
- `coverage_heatmap.png`
- `results.json`

## Algorithm
1. Initialize a random population.
2. Evaluate every candidate.
3. Preserve the top three candidates (elitism).
4. Select parents using tournament selection.
5. Apply segment crossover and coordinate/control mutation.
6. Repair bounds and repeat for 100 generations.

## Notes and limitations
This is a simplified grid-based educational model, not flight-certified software. Segment intersection and sensor detection are approximated; the objective is a weighted single fitness score rather than a formal Pareto-front multi-objective optimizer. The plotted routes are waypoint-to-waypoint line segments, so a real system would need grid-cell pathfinding/repair to guarantee every segment avoids no-fly cells. Run multiple random seeds for statistically stronger results.
