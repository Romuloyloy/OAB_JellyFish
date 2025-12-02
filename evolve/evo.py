# evolve/evo.py
import math
import random
from typing import Tuple

import torch

from sim import Arena, ARENA_BASE_R, radial_distance_to_boundary
from policy import new_policy
from fitness import FoodField, per_step_reward


def evaluate(
    weights_sd,
    J: float = 0.4,
    wall_contrast: float = 0.6,
    steps: int = 4000,
    seed: int = 0
) -> Tuple[float, dict]:
    """
    Evaluate one genome:

    - Randomize geometry per evaluation:
        * R_scale ∈ [0.7, 1.3]
        * aspect  ∈ [0.7, 1.3]  (ellipse: a = R_eff * aspect, b = R_eff / aspect)
    - Build Arena with those params.
    - Run a single episode of `steps`.
    - Return (total_reward, stats).
    """
    rng = random.Random(seed)

    # Domain randomization: radius & ellipticity
    R_scale = rng.uniform(0.7, 1.3)
    aspect = rng.uniform(0.7, 1.3)

    # build policy and load weights
    pol = new_policy()
    if weights_sd:
        pol.load_state_dict(weights_sd)

    # environment with randomized geometry
    env = Arena(J=J, wall_contrast=wall_contrast, seed=seed,
                R_scale=R_scale, aspect=aspect)
    # food layout inside same ellipse
    food = FoodField(a=env.a, b=env.b, n_init=4, n_max=4, seed=seed)
    env.reset()

    total_r = 0.0
    collisions = 0
    collected_total = 0
    last_pos = (env.agent.x, env.agent.y)

    for _ in range(steps):
        obs = env.observe()
        L, R, P = pol.act(obs)

        env.step(L, R, P)
        if env.collided_prev:
            collisions += 1

        collected = food.step_collect(env.agent.x, env.agent.y)
        collected_total += collected

        # distance to boundary along radial direction
        d_wall = radial_distance_to_boundary(env.agent.x, env.agent.y, env.a, env.b)
        # movement length
        move_len = math.hypot(env.agent.x - last_pos[0], env.agent.y - last_pos[1])
        last_pos = (env.agent.x, env.agent.y)

        total_r += per_step_reward(d_wall, env.collided_prev, move_len, collected)

    stats = {
        "collisions": collisions,
        "food": collected_total,
        "total_reward": total_r,
        "R_scale": R_scale,
        "aspect": aspect,
    }
    return total_r, stats


def mutate(sd, sigma: float = 0.1, rng=None):
    """
    Gaussian mutation over a state_dict.
    """
    if rng is None:
        rng = random.Random()
    out = {}
    for k, v in sd.items():
        noise = torch.randn_like(v) * sigma
        out[k] = v + noise
    return out
