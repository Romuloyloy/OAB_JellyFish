# evolve/evo.py
import math
import os
import json
import time
import random
import torch
from typing import Tuple

from sim import Arena
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
    Evaluate a single policy (given by state_dict) in the Arena.

    Setup:
      - Two food pellets at a time (FoodField(n=2)).
      - Food respawns only when eaten.
      - Reward:
          +λf for eating food
          -λc for collisions
        No close-wall penalty, no movement bonus.
    """
    rng = random.Random(seed)

    # build policy and load weights
    pol = new_policy()
    if weights_sd:
        pol.load_state_dict(weights_sd)

    # environment
    env = Arena(J=J, wall_contrast=wall_contrast, seed=seed)
    food = FoodField(n=2, seed=seed)  # TWO pellets at a time now
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

        # d_wall and move_len are still computed so we can pass them into
        # per_step_reward for compatibility, even though they are currently unused.
        d_wall = 250.0 - ((env.agent.x ** 2 + env.agent.y ** 2) ** 0.5)
        move_len = math.hypot(env.agent.x - last_pos[0], env.agent.y - last_pos[1])
        last_pos = (env.agent.x, env.agent.y)

        total_r += per_step_reward(d_wall, env.collided_prev, move_len, collected)

    return total_r, {
        "collisions": collisions,
        "food": collected_total,
    }


def mutate(sd, sigma=0.1, rng=None):
    if rng is None:
        rng = random.Random()
    out = {}
    for k, v in sd.items():
        noise = torch.randn_like(v) * sigma
        out[k] = v + noise
    return out
