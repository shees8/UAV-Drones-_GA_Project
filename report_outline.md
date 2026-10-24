# Technical Report: Genetic Algorithm for Multi-UAV Disaster Surveillance

## 1. Introduction
The task is to coordinate a fleet of UAVs on a discretized hazard grid. The strategy must maximize priority-zone coverage and target detection while reducing energy use, flight time, and exposure to hazardous or restricted cells. A Genetic Algorithm is suitable because candidate routes and control profiles form a large, non-convex search space.

## 2. Chromosome Representation
A candidate solution has three drone plans. Each drone plan stores seven ordered waypoints `(x, y)`, seven speed multipliers, and seven sensor-power values. Coordinates are integer-valued and bounded to `[0, 29]`; speed multipliers are bounded to `[0.6, 1.4]`; sensor power is bounded to `[0.2, 1.0]`. The start point for each drone is fixed. This hybrid representation encodes spatial route decisions and continuous control choices in one chromosome.

## 3. Fitness Function
For candidate `S`, the implementation maximizes:
`F(S) = 18 C(S) - 0.30 E(S) - 0.12 T(S) - 1.8 H(S) - 30 N(S) - 8 R(S)`
where `C` is weighted detected-target coverage, `E` is estimated energy, `T` is estimated flight time, `H` is accumulated hazard exposure, `N` is the number of sampled no-fly-zone intersections, and `R` is total distance beyond the allowed route length. The large no-fly penalty makes restricted-space violations especially undesirable. The weights are heuristic and should be tuned against the scale of the metrics; they do not guarantee a mathematically Pareto-optimal result.

## 4. Genetic Operators and Constraints
Tournament selection chooses the fittest among four randomly selected candidates. Segment crossover combines waypoint, speed, and sensor segments from two parents. Mutation randomly perturbs waypoint coordinates and adds Gaussian noise to speed and sensor values. Elitism preserves the best three individuals. A repair operation clamps coordinates and controls to legal ranges. No-fly zones and route range are handled through fitness penalties; a production implementation should additionally repair each route segment with grid-based A* or another collision-free path planner.

## 5. Experimental Setup and Visualization
The demonstration uses a 30 x 30 grid, three drones, seven waypoints per drone, population size 60, 100 generations, mutation rate 0.12, tournament size 4, and three elite candidates. The map contains restricted cells, weighted hazard regions, and six target locations. The program generates (1) best and average fitness convergence, (2) evolved UAV routes over the hazard map, and (3) a route traversal heatmap.

## 6. Hyperparameter Sensitivity (Expected Effects)
- **Population size:** a larger population explores more route combinations but increases computation per generation. A small population may converge prematurely.
- **Mutation rate:** a low rate can freeze waypoint patterns; a high rate disrupts good routes. The chosen 0.12 is a starting point, not a universally optimal value.
- **Selection pressure:** tournament size 4 provides moderate pressure. Larger tournaments favor high-fitness parents more strongly and may reduce diversity; smaller tournaments preserve diversity but can slow progress.
For a valid empirical sensitivity study, rerun the experiment while changing one parameter at a time (e.g. population 30/60/100, mutation 0.05/0.12/0.25, tournament size 2/4/7), and compare best fitness over several random seeds. The current script demonstrates the baseline configuration; these comparisons should be run and recorded before claiming measured outcomes.

## 7. Comparison with Random Search
Random search samples candidates without using fitness feedback to produce offspring. The GA reuses information from strong candidates through selection and crossover, and introduces new variants through mutation. Therefore, GA is expected to improve fitness more systematically over generations. This is a hypothesis to verify experimentally by implementing a random-search baseline with the same evaluation budget; do not claim a measured advantage unless both methods are actually run.

## 8. Conclusion
The proposed GA combines a hybrid chromosome, weighted fitness function, tailored genetic operators, and constraint penalties to produce candidate UAV deployment plans. The figures and JSON metrics support evaluation of convergence, route geometry, and coverage. For operational deployment, the model would require robust collision-free routing, realistic sensor models, multi-objective Pareto analysis, and validation against physical flight constraints.
