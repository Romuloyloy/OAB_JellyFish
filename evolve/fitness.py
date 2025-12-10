# evolve/fitness.py
import math
import random
from typing import List, Tuple

FOOD_R = 3.0
ARENA_R = 250.0


class FoodField:
    """
    Food field for the EA:

      - We keep a fixed number of pellets (n) in the arena.
      - On each step, if the agent is close enough to a pellet, it is "eaten"
        and immediately respawned somewhere else.

    IMPORTANT CHANGE:
      - On episode start:
          * The FIRST pellet is forced to be NON-CENTRAL
            (i.e., outside an inner radius of the arena).
          * All other pellets (and all respawns) are uniform in the disk.

        This prevents the initial "coward" that sits in the middle from
        being rewarded just because the first pellet happens to spawn centrally.
    """

    def __init__(self, n: int = 2, seed: int = 0):
        self.rng = random.Random(seed)
        self.points: List[Tuple[float, float]] = []

        for i in range(n):
            if i == 0:
                # First pellet: non-central
                self.points.append(self._rand_point_noncentral())
            else:
                # Other initial pellets: normal uniform in disk
                self.points.append(self._rand_point())

    # --- sampling helpers ---

    def _rand_point(self) -> Tuple[float, float]:
        """
        Uniform in disk (slightly inside the boundary).
        """
        r = ARENA_R * math.sqrt(self.rng.random()) * 0.9
        a = 2 * math.pi * self.rng.random()
        return (r * math.cos(a), r * math.sin(a))

    def _rand_point_noncentral(self) -> Tuple[float, float]:
        """
        Sample a point that is NOT near the center.

        Implementation: rejection sampling.
        Repeatedly sample from _rand_point() until the radius is
        outside some inner fraction of ARENA_R (e.g. > 0.4 * R).

        This ensures the first pellet is somewhere in the mid/outer region,
        so a policy that just spins in the center cannot get rewarded
        from the very first spawn.
        """
        min_radius = 0.4 * ARENA_R  # you can tune this (0.3, 0.5, ...)
        while True:
            x, y = self._rand_point()
            if math.hypot(x, y) > min_radius:
                return (x, y)

    # --- per-step logic ---

    def step_collect(self, x: float, y: float) -> int:
        """
        Check if agent is close enough to any pellet to collect it.

        Returns:
            collected (int): how many pellets were eaten this step.

        For each eaten pellet, we immediately spawn a new one
        using the NORMAL uniform distribution (_rand_point), not the
        noncentral version. After the first spawn, we allow central food.
        """
        collected = 0
        new_points: List[Tuple[float, float]] = []

        for (fx, fy) in self.points:
            if math.hypot(x - fx, y - fy) <= FOOD_R:
                collected += 1
                # respawn a fresh pellet (now fully uniform again)
                new_points.append(self._rand_point())
            else:
                new_points.append((fx, fy))

        self.points = new_points
        return collected


def per_step_reward(d_wall, collided, move_len, collected):
    """
    Reward shaping:

      - NO close-to-wall penalty.
      - NO movement bonus.
      - ONLY:
          * negative reward for collisions
          * positive reward for eating food

    Arguments d_wall and move_len are kept for compatibility but unused.
    Adjust λc and λf as you like.
    """
    λc = 5.0      # collision penalty
    λf = 1500.0     # reward for eating one food pellet (tune as needed)

    r = 0.0
    if collided:
        r -= λc
    if collected > 0:
        r += λf * collected

    return r
