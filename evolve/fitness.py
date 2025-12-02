# evolve/fitness.py
import math
import random
from typing import List, Tuple


class FoodField:
    """
    Food field inside an ellipse (a,b):

    - At reset: spawn n_init food pellets uniformly in the ellipse.
    - Each step:
        * Any pellet within FOOD_R of the agent is eaten.
        * Eaten pellets are removed.
        * New pellets are spawned until there are n_max total.
    """

    def __init__(self, a: float, b: float,
                 n_init: int = 4, n_max: int = 4,
                 seed: int = 0,
                 food_r: float = 3.0):
        self.rng = random.Random(seed)
        self.a = float(a)
        self.b = float(b)
        self.n_max = n_max
        self.food_r = float(food_r)

        n_init = min(n_init, n_max)
        self.points: List[Tuple[float, float]] = [
            self._rand_point() for _ in range(n_init)
        ]

    def _rand_point(self) -> Tuple[float, float]:
        """
        Uniform sampling inside ellipse:
        sample from unit disk, then scale by a,b.
        """
        r = math.sqrt(self.rng.random())  # radius in [0,1], area-uniform
        theta = 2 * math.pi * self.rng.random()
        x_unit = r * math.cos(theta)
        y_unit = r * math.sin(theta)
        return (x_unit * self.a * 0.9, y_unit * self.b * 0.9)  # 0.9 margin

    def step_collect(self, x: float, y: float) -> int:
        """
        - Check which pellets are eaten at (x, y).
        - Remove them.
        - Respawn new pellets until we have n_max total.
        - Return number eaten this step.
        """
        collected = 0
        remaining: List[Tuple[float, float]] = []

        for (fx, fy) in self.points:
            if math.hypot(x - fx, y - fy) <= self.food_r:
                collected += 1
            else:
                remaining.append((fx, fy))

        while len(remaining) < self.n_max:
            remaining.append(self._rand_point())

        self.points = remaining
        return collected


def per_step_reward(
    d_wall: float,
    collided: int,
    move_len: float,
    collected: int
) -> float:
    """
    Reward shaping:

    - Big negative for collisions.
    - Mild penalty for being too close to wall.
    - Positive for movement (encourages exploration).
    - Strong positive for collecting food.
    - Hunger penalty if no food collected on this step.
    """

    # You can tune these
    λc = 15.0    # collision penalty
    λw = 0.01    # proximity penalty if within R_margin
    λm = 0.02    # movement bonus per unit distance
    λf = 4.0     # food reward per item
    λh = 0.02    # hunger penalty per step with no food
    R_margin = 40.0

    r = 0.0

    # wall / collision cost
    if collided:
        r -= λc
    if d_wall < R_margin:
        r -= λw * (R_margin - d_wall)

    # exploration (distance actually moved)
    r += λm * move_len

    # food
    r += λf * collected

    # hunger: penalize steps without food
    if collected == 0:
        r -= λh

    return r
