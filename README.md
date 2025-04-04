# ACO Simulation

## Ant Colony Optimization Simulation

This project simulates an **Ant Colony Optimization (ACO)** algorithm in a grid-based environment. The simulation visualizes the behavior of worker and explorer ants as they search for food, deposit pheromones, and return to their nest. The environment is rendered using **Pygame**.

---

## Features

### Ant Behavior
- Worker ants follow pheromone trails to collect food and return to the nest.
- Explorer ants move randomly to discover new food sources.
- Ants deposit pheromones to guide others to food sources.

### Environment
- Grid-based environment with food sources, a nest, and pheromone trails.
- Randomly generated food sources with varying sizes and values.
- Pheromone levels decay over time and are influenced by weather conditions.

### Visualization
- Real-time rendering of the grid, ants, food sources, and pheromone trails using Pygame.
- Interactive controls to pause, toggle the legend, and inspect pheromone values.

---

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/begad-tamim/aco-simulation.git
    cd aco-simulation
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the simulation:
    ```sh
    python "ACO Simulation.py"
    ```

---

## Controls

- **Space**: Pause/Resume the simulation.
- **L**: Toggle the legend.
- **Mouse Click**: Inspect the pheromone value at a specific grid cell.
- **ESC**: Quit the simulation.

---

## Command-Line Arguments

Customize the simulation using the following arguments:

| Argument                  | Default Value       | Description                                   |
|---------------------------|---------------------|-----------------------------------------------|
| `--grid_size`             | `51`               | Size of the grid.                             |
| `--num_ants`              | `10`               | Number of ants in the simulation.             |
| `--food_sources_num`      | `5`                | Number of food sources.                       |
| `--food_sources_size_range` | `(10, 30)`       | Range of food source sizes.                   |
| `--exploration_rate`      | `0.2`              | Exploration rate for explorer ants.           |
| `--explorer_ants_percentage` | `0.3`           | Percentage of explorer ants.                  |
| `--pheromone_decay`       | `0.05`             | Pheromone decay rate.                         |
| `--tick_rate`             | `10`               | Simulation tick rate (frames per second).     |
| `--weather_effect_range`  | `(0.5, 1.5)`       | Range of weather effects on pheromone levels. |

### Example:
```sh
python "ACO Simulation.py" --grid_size 100 --num_ants 20 --tick_rate 30
```
## License

This project is licensed under the MIT License.
