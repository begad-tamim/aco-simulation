"""
Ant Colony Optimization Simulation
    This script simulates an ant colony optimization algorithm using a grid-based environment.
    The simulation includes worker and explorer ants, pheromone trails, food sources, and a nest.
    The ants move around the grid, collect food, and deposit pheromones to guide other ants.
    The simulation is visualized using Pygame, with a grid representing the environment and various colors for different elements.

Doctor: Omar Shalash
Eng: Ahmed Metwalli
Course: Swarm Intelligence (RB 414)

Student Names:
    - Mohamed Youssef - 211001821
    - Begad Tamim - 211002177

"""

import numpy as np
import random
import pygame
import time
import argparse

# Initialization of constants
GRID_SIZE = 51
NUM_ANTS = 10
PHEROMONE_DECAY = 0.05
EXPLORATION_RATE = 0.2
EXPLORER_ANTS_PERCENTAGE = 0.3
FOOD_SOURCES_NUM = 5
FOOD_SOURCES_SIZE_RANGE = (10, 30)
NEST_LOCATION = (GRID_SIZE // 2, GRID_SIZE // 2)
CELL_SIZE_PX = 10
WEATHER_EFFECT_RANGE = (0.5, 1.5)
TICK_RATE = 10

COLORS = {
    "food": (34, 100, 34),  # A darker green for food sources
    "explorer": (70, 130, 180),  # Steel blue for explorer ants
    "worker": (138, 43, 226),  # Dark violet for worker ants
    "ant_with_food": (255, 140, 0),  # Dark orange for ants with food
    "nest": (255, 255, 0),  # Yellow for the nest (stands out more)
    "pheromone": (255, 30, 30),  # Red for pheromone trails
    "background": (30, 30, 30),  # Dark gray instead of black for better contrast
    "line": (50, 50, 50),  # Gray for grid lines
    "font": (255, 255, 255),  # White for text
}


class Ant:
    """Class to represent an ant in the simulation."""

    def __init__(
        self,
        ant_id: int,
        x: int = NEST_LOCATION[0],
        y: int = NEST_LOCATION[1],
        role: str = "worker",
    ) -> None:
        """
        Initialize the Ant object.
        Args:
            ant_id (int): Unique identifier for the ant.
            x (int): Starting x-coordinate of the ant.
            y (int): Starting y-coordinate of the ant.
            role (str): Role of the ant, either "worker" or "explorer".
        """
        self.id = ant_id  # Unique identifier for the ant
        self.x = x  # Starting position at the nest
        self.y = y  # Starting position at the nest
        self.has_food = False  # Flag to indicate if the ant is carrying food
        self.recent_food_location = None  # Store the location of the food source
        self.role = role  # Role can be "worker" or "explorer"
        self.path = []  # Store the path taken by the ant

    # Todo: Make movement better (m) (monte carlo selection)
    def move(self, grid, food_sources, nest_grid):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        move = (0, 0)  # Default (no movement)
        score = 0

        if self.has_food:
            best_move = None
            best_value = 0  # Initialize to a low value

            for d in directions:
                new_x, new_y = self.x + d[0], self.y + d[1]
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                    if nest_grid[new_x, new_y] > best_value:
                        best_value = nest_grid[new_x, new_y]
                        best_move = d  # Update move towards the highest-value cell

            if best_move:
                # Add some randomness to the movement
                if random.random() < 0.1:
                    move = random.choice(directions)
                else:
                    move = best_move

        else:
            if random.random() < EXPLORATION_RATE and self.role == "explorer":
                move = random.choice(directions)  # Random movement for explorers
            else:
                # Find the highest food pheromone level nearby
                best_move = None
                best_value = 0

                for d in directions:
                    new_x, new_y = self.x + d[0], self.y + d[1]
                    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                        if grid[new_x, new_y] > best_value:
                            best_value = grid[new_x, new_y]
                            best_move = d  # Move towards the strongest pheromone trail

                if best_move:
                    move = best_move
                else:
                    move = random.choice(directions)

        # Update position
        new_x, new_y = self.x + move[0], self.y + move[1]

        # Randomly change direction if the ant is stuck
        if new_x == self.x and new_y == self.y:
            move = random.choice(directions)
            new_x, new_y = self.x + move[0], self.y + move[1]

        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            self.x, self.y = new_x, new_y

        self.old_x, self.old_y = self.x, self.y

        self.path.append((self.x, self.y))

        for cluster in food_sources:
            for food in cluster:
                if (self.x, self.y) == (food[0], food[1]):
                    if not self.has_food:
                        self.has_food = True
                        self.recent_food_location = (self.x, self.y)

                        food[2] -= 1  # Decrease the food source value
                        if food[2] <= 0:
                            cluster.remove(food)

                        break

        # Check if the ant is at the nest location
        if self.has_food and (self.x, self.y) == NEST_LOCATION:
            self.has_food = False
            score += 1

            # Get the path from the recent food location to the nest
            recent_path = self.path[self.path.index(self.recent_food_location) :]

            # Add phermone to the path
            for i, (x, y) in enumerate(recent_path):
                # Decay the pheromone trail
                grid[x, y] += 1 - (i / len(self.path))

        return grid, food_sources, score


class Simulation:
    """Class to simulate the Ant Colony Optimization (ACO) algorithm."""

    def __init__(self) -> None:
        """Initialize the simulation with ants and food sources."""
        self.ants = [
            (
                Ant(ant_id=i, role="worker")
                if i < NUM_ANTS * (1 - EXPLORER_ANTS_PERCENTAGE)
                else Ant(ant_id=i, role="explorer")
            )
            for i in range(NUM_ANTS)
        ]  # Create ants with roles

        self.food_sources = self.create_random_food_sources()  # Create food sources
        self.pheromone_grid = np.zeros(
            (GRID_SIZE, GRID_SIZE)
        )  # Initialize pheromone grid
        self.food_found = 0  # Initialize food found counter
        self.nest_grid = self.generate_nest_grid(GRID_SIZE)  # Generate the nest grid

        # Pygame Setup
        pygame.init()
        pygame.display.set_caption("Ant Colony Optimization Simulation")
        icon = pygame.image.load("icon.ico")  # Load the icon image
        pygame.display.set_icon(icon)

        self.screen = pygame.display.set_mode(
            ((GRID_SIZE * CELL_SIZE_PX), (GRID_SIZE * CELL_SIZE_PX) + 150)
        )
        self.clock = pygame.time.Clock()

    def generate_nest_grid(self, size):
        """
        Generates a nest grid with a specified size.

        Parameters:
        - size (int): The size of the grid.

        Returns:
        - grid (numpy.ndarray): The generated nest grid.
        """
        target = (size // 2, size // 2)
        grid = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                grid[i, j] = size - max(abs(i - target[0]), abs(j - target[1]))

        grid = grid / np.max(grid)
        return grid

    def create_random_food_sources(self) -> list:
        """
        Creates random food sources in the simulation grid.

        Returns:
            clusters (list): List of clusters, where each cluster is a list of food sources.
                Each food source is represented as a list [x, y, value], where x and y are
                the coordinates and value is a random value assigned to the food source.
        """
        clusters = []  # List to hold clusters

        # Randomly select cluster centers
        cluster_centers = []

        while len(cluster_centers) < FOOD_SOURCES_NUM:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)

            if (x, y) == NEST_LOCATION:
                continue

            # Avoid placing food sources too close to the nest
            if abs(x - NEST_LOCATION[0]) < 2 and abs(y - NEST_LOCATION[1]) < 2:
                continue

            cluster_centers.append((x, y))

        for center_x, center_y in cluster_centers:
            cluster = []
            cluster_size = random.randint(
                FOOD_SOURCES_SIZE_RANGE[0], FOOD_SOURCES_SIZE_RANGE[1]
            )
            while len(cluster) < cluster_size:
                # Generate a point around the center with normal distribution
                x = int(np.clip(np.random.normal(center_x, 1), 0, GRID_SIZE - 1))
                y = int(np.clip(np.random.normal(center_y, 1), 0, GRID_SIZE - 1))

                # Check if the point is the nest location
                if (x, y) == NEST_LOCATION:
                    continue

                value = random.randint(1, 10)  # Random value for the point
                cluster.append([x, y, value])

            clusters.append(cluster)

        return clusters

    def check_food_sources(self) -> None:
        """
        Check the food sources and perform necessary operations.

        This method checks if any food sources are empty and removes them from the list of food sources.
        If all food sources are empty, it creates new random food sources.

        Returns:
            None
        """
        # Check if food sources are empty
        for cluster in self.food_sources:
            for food in cluster:
                if food[2] <= 0:
                    cluster.remove(food)

        # Remove empty clusters
        self.food_sources = [
            cluster for cluster in self.food_sources if len(cluster) > 0
        ]

        # If all clusters are empty, create new ones
        if len(self.food_sources) == 0:
            self.food_sources = self.create_random_food_sources()

    def update_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Update the pheromone grid by decaying the pheromones and applying weather effects.
        Args:
            grid (np.ndarray): The pheromone grid.
        Returns:
            np.ndarray: The updated pheromone grid.
        """
        grid *= 1 - PHEROMONE_DECAY  # Decay all pheromones
        # Apply weather effect to the pheromone levels
        weather_effect = random.uniform(
            WEATHER_EFFECT_RANGE[0], WEATHER_EFFECT_RANGE[1]
        )
        grid *= weather_effect

        # Make any values less than 0.01 equal to 0
        grid[grid < 0.01] = 0.0

        return grid

    def draw(
        self, iteration: int, clicked_pheromone_value: float, draw_legend: bool
    ) -> None:
        """
        Draw the simulation grid, ants, food sources, and other elements.
        This method is responsible for rendering the current state of the simulation on the screen.
        Args:
            iteration (int): The current iteration of the simulation.
            clicked_pheromone_value (float): The pheromone value at the clicked location.
            draw_legend (bool): Flag to indicate whether to draw the legend.

        Returns:
            None
        """
        self.screen.fill(COLORS["background"])  # Fill the screen with a dark gray color

        # Loop through the grid and draw the cells
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                color = COLORS["background"]  # Default color for the cell

                # Get the color for the cell based on the pheromone level
                pheromone_value = self.pheromone_grid[x, y]

                if pheromone_value > 0:
                    if pheromone_value > 1:
                        pheromone_value = 1
                    color = COLORS["pheromone"]
                    color = (
                        min(color[0], int(pheromone_value * color[0] + 30)),
                        color[1],
                        color[2],
                    )  # Pheromone intensity in red

                else:
                    color = COLORS["background"]

                # Draw the ants based on their roles and food status
                for ant in self.ants:
                    if (x, y) == (ant.x, ant.y):
                        if ant.has_food:
                            color = COLORS["ant_with_food"]
                        else:
                            color = (
                                COLORS["worker"]
                                if ant.role == "worker"
                                else COLORS["explorer"]
                            )

                # Draw the cells with the determined color
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x * CELL_SIZE_PX,
                        y * CELL_SIZE_PX,
                        CELL_SIZE_PX,
                        CELL_SIZE_PX,
                    ),
                )

                # Draw the food sources
                for cluster in self.food_sources:
                    for food in cluster:
                        if (x, y) == (food[0], food[1]):
                            # Green for food sources
                            color = COLORS["food"]
                            color = (color[0], color[1] + int(food[2] * 15), color[2])

                            pygame.draw.rect(
                                self.screen,
                                color,
                                (
                                    food[0] * CELL_SIZE_PX,
                                    food[1] * CELL_SIZE_PX,
                                    CELL_SIZE_PX,
                                    CELL_SIZE_PX,
                                ),
                            )

        # Draw the nest
        pygame.draw.rect(
            self.screen,
            COLORS["nest"],
            (
                NEST_LOCATION[0] * CELL_SIZE_PX,
                NEST_LOCATION[1] * CELL_SIZE_PX,
                CELL_SIZE_PX,
                CELL_SIZE_PX,
            ),
        )

        # Add lines to the grid
        for x in range(0, GRID_SIZE * CELL_SIZE_PX, CELL_SIZE_PX):
            pygame.draw.line(
                self.screen, COLORS["line"], (x, 0), (x, GRID_SIZE * CELL_SIZE_PX)
            )
        for y in range(0, GRID_SIZE * CELL_SIZE_PX, CELL_SIZE_PX):
            pygame.draw.line(
                self.screen, COLORS["line"], (0, y), (GRID_SIZE * CELL_SIZE_PX, y)
            )
        pygame.draw.line(
            self.screen,
            COLORS["line"],
            (GRID_SIZE * CELL_SIZE_PX - 1, 0),
            (GRID_SIZE * CELL_SIZE_PX - 1, GRID_SIZE * CELL_SIZE_PX),
        )
        pygame.draw.line(
            self.screen,
            COLORS["line"],
            (0, GRID_SIZE * CELL_SIZE_PX),
            (GRID_SIZE * CELL_SIZE_PX, GRID_SIZE * CELL_SIZE_PX),
        )

        # Draw the number of food sources found
        font = pygame.font.Font(None, 28)
        text = font.render(f"Food: {self.food_found}", True, COLORS["font"])
        self.screen.blit(text, (10, GRID_SIZE * CELL_SIZE_PX + 10))

        # Draw the time
        font = pygame.font.Font(None, 28)
        text = font.render(
            f"Time: {int(time.time() - self.start_time)}", True, COLORS["font"]
        )
        self.screen.blit(text, (10, GRID_SIZE * CELL_SIZE_PX + 35))

        # Draw the iteration
        font = pygame.font.Font(None, 28)
        text = font.render(f"Iteration: {iteration}", True, COLORS["font"])
        self.screen.blit(text, (10, GRID_SIZE * CELL_SIZE_PX + 60))

        # Draw the pheromone value at the clicked location
        font = pygame.font.Font(None, 28)
        text = font.render(
            f"Pheromone Value: {clicked_pheromone_value}", True, COLORS["font"]
        )
        self.screen.blit(text, (10, GRID_SIZE * CELL_SIZE_PX + 85))

        # Draw the legend on the right side
        if draw_legend:
            legend_x = GRID_SIZE * CELL_SIZE_PX - 210
            legend_y = 10
            legend_width = 200
            legend_height = 160
            pygame.draw.rect(
                self.screen,
                COLORS["background"],
                (legend_x, legend_y, legend_width, legend_height),
                border_radius=5,
            )
            pygame.draw.rect(
                self.screen,
                COLORS["line"],
                (legend_x, legend_y, legend_width, legend_height),
                width=2,
                border_radius=5,
            )
            legend_font = pygame.font.Font(None, 20)
            legend_text = legend_font.render("Legend", True, COLORS["font"])
            self.screen.blit(legend_text, (legend_x + 10, legend_y + 8))
            legend_items = [
                ("Food", "Green", COLORS["food"]),
                ("Nest", "Yellow", COLORS["nest"]),
                ("Ant with Food", "Orange", COLORS["ant_with_food"]),
                ("Worker Ant", "Violet", COLORS["worker"]),
                ("Explorer Ant", "Blue", COLORS["explorer"]),
                ("Pheromone Trail", "Red", COLORS["pheromone"]),
            ]
            for i, (name, color, value) in enumerate(legend_items):
                pygame.draw.rect(
                    self.screen,
                    value,
                    (legend_x + 10, legend_y + 35 + i * 20, 15, 15),
                )
                text = legend_font.render(name, True, COLORS["font"])
                self.screen.blit(text, (legend_x + 35, legend_y + 35 + i * 20))

        # Draw the title in the bottom middle
        font = pygame.font.Font(None, 36)
        text = font.render("Ant Colony Optimization Simulation", True, COLORS["font"])
        self.screen.blit(
            text, (GRID_SIZE * CELL_SIZE_PX // 2 - 216, GRID_SIZE * CELL_SIZE_PX + 115)
        )

        # Draw the controls in the bottom right
        font = pygame.font.Font(None, 20)
        text = font.render(
            "Controls: Space to pause, L for legend",
            True,
            COLORS["font"],
        )
        self.screen.blit(
            text, (GRID_SIZE * CELL_SIZE_PX - 250, GRID_SIZE * CELL_SIZE_PX + 10)
        )

        # Continue drawing the controls
        text = font.render(
            "Click to get pheromone value, ESC to quit",
            True,
            COLORS["font"],
        )
        self.screen.blit(
            text, (GRID_SIZE * CELL_SIZE_PX - 271, GRID_SIZE * CELL_SIZE_PX + 30)
        )

        pygame.display.flip()  # Update the display

    def simulate(self) -> None:
        """
        Simulates the ACO (Ant Colony Optimization) algorithm.

        This method runs the main loop of the simulation, handling events, moving ants, updating the pheromone grid,
        checking food sources, drawing the simulation, and controlling the speed of the simulation.

        Returns:
            None
        """
        iteration = 0  # Start from iteration 0
        running = True  # Set running to True
        paused = False  # Set paused to False
        draw_legend = True  # Set draw_legend to True
        self.start_time = time.time()  # Start the timer
        self.elapsed_time = 0  # Initialize elapsed time
        pheremone_value = 0.0  # Initialize pheromone value

        # Main loop
        while running:
            # Handle events to allow quitting the simulation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key == pygame.K_SPACE:
                        # Pause the simulation
                        paused = True
                        self.elapsed_time = time.time() - self.start_time
                        while paused:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    paused = False
                                    running = False
                                    break
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_ESCAPE:
                                        paused = False
                                        running = False
                                        break
                                    if event.key == pygame.K_SPACE:
                                        paused = False
                                        self.start_time = (
                                            time.time() - self.elapsed_time
                                        )
                                        break
                    if event.key == pygame.K_l:
                        # Toggle the legend
                        draw_legend = not draw_legend

            # Check for a click to get the pheromone value
            if pygame.mouse.get_pressed()[0]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x = mouse_x // CELL_SIZE_PX
                grid_y = mouse_y // CELL_SIZE_PX
                if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                    pheremone_value = self.pheromone_grid[grid_x, grid_y]
                    pheremone_value = round(
                        pheremone_value, 2
                    )  # Round to 2 decimal places

            # Move ants if not paused
            for ant in self.ants:
                self.pheromone_grid, self.food_sources, score = ant.move(
                    self.pheromone_grid, self.food_sources, self.nest_grid
                )
                self.food_found += score

            # Update pheromone grid
            self.pheromone_grid = self.update_grid(self.pheromone_grid)

            # Check food sources
            self.check_food_sources()

            # Draw the simulation
            self.draw(iteration, pheremone_value, draw_legend)

            # Add a delay to control the speed of the simulation
            self.clock.tick(TICK_RATE)
            iteration += 1

        pygame.quit()


if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Ant Colony Optimization Simulation")
    parser.add_argument(
        "--grid_size", type=int, default=GRID_SIZE, help="Size of the grid"
    )
    parser.add_argument(
        "--num_ants",
        type=int,
        default=NUM_ANTS,
        help="Number of ants in the simulation",
    )
    parser.add_argument(
        "--food_sources_num",
        type=int,
        default=FOOD_SOURCES_NUM,
        help="Number of food sources in the simulation",
    )
    parser.add_argument(
        "--food_sources_size_range",
        type=int,
        nargs=2,
        default=FOOD_SOURCES_SIZE_RANGE,
        help="Range of food source sizes",
    )
    parser.add_argument(
        "--exploration_rate",
        type=float,
        default=EXPLORATION_RATE,
        help="Exploration rate for explorer ants",
    )
    parser.add_argument(
        "--explorer_ants_percentage",
        type=float,
        default=EXPLORER_ANTS_PERCENTAGE,
        help="Percentage of explorer ants",
    )
    parser.add_argument(
        "--pheromone_decay",
        type=float,
        default=PHEROMONE_DECAY,
        help="Pheromone decay rate",
    )
    parser.add_argument(
        "--tick_rate",
        type=int,
        default=TICK_RATE,
        help="Tick rate for the simulation",
    )
    parser.add_argument(
        "--weather_effect_range",
        type=float,
        nargs=2,
        default=WEATHER_EFFECT_RANGE,
        help="Range of weather effects on pheromone levels",
    )

    args = parser.parse_args()

    # Update constants based on command line arguments
    GRID_SIZE = args.grid_size
    NUM_ANTS = args.num_ants
    FOOD_SOURCES_NUM = args.food_sources_num
    FOOD_SOURCES_SIZE_RANGE = args.food_sources_size_range
    EXPLORATION_RATE = args.exploration_rate
    EXPLORER_ANTS_PERCENTAGE = args.explorer_ants_percentage
    PHEROMONE_DECAY = args.pheromone_decay
    TICK_RATE = args.tick_rate
    WEATHER_EFFECT_RANGE = args.weather_effect_range

    sim = (
        Simulation()
    )  # Create a simulation instance (which will create the ants and food sources)
    # Start the simulation
    sim.simulate()  
