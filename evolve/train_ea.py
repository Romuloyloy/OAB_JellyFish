# evolve/train_ea.py
import os
import json
import time
import random

import torch

from evo import evaluate, mutate
from policy import new_policy


def main():
    """
    Run an evolutionary algorithm with domain randomization:

    - Population: 64 genomes.
    - Generations: 100.
    - Each genome evaluated once per generation in a randomized ellipse:
        * radius scale ∈ [0.7, 1.3]
        * ellipticity ∈ [0.7, 1.3]
    - Saves best.pt under runs/evo/<timestamp>/best.pt
    """
    outdir = os.path.join("runs", "evo", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)

    meta = {
        "pop": 400,
        "gens": 15,
        "steps": 10000,
        "J": 0.4,
        "wall_contrast": 0.5,
        "seed": 123,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    rng = random.Random(meta["seed"])

    base = new_policy().state_dict()
    pop = [mutate(base, sigma=0.5, rng=rng) for _ in range(meta["pop"])]

    best_sd = None
    best_score = -1e9

    for g in range(meta["gens"]):
        scores = []

        for i, sd in enumerate(pop):
            s, stats = evaluate(
                sd,
                J=meta["J"],
                wall_contrast=meta["wall_contrast"],
                steps=meta["steps"],
                seed=rng.randrange(10**9),
            )
            scores.append((s, sd))

        scores.sort(key=lambda x: x[0], reverse=True)

        if scores[0][0] > best_score:
            best_score = scores[0][0]
            best_sd = scores[0][1]
            torch.save(best_sd, os.path.join(outdir, "best.pt"))

        mean = sum(s for s, _ in scores) / len(scores)
        print(f"gen {g}: best={scores[0][0]:.2f} mean={mean:.2f} (running best {best_score:.2f})")

        # elitism + breeder set
        elite = [sd for _, sd in scores[:4]]
        breeders = [sd for _, sd in scores[: max(1, len(scores) // 4)]]

        new_pop = []
        new_pop.extend(elite)
        while len(new_pop) < meta["pop"]:
            parent = rng.choice(breeders)
            child = mutate(parent, sigma=0.3, rng=rng)
            new_pop.append(child)

        pop = new_pop

    print(f"done. best saved to {os.path.join(outdir, 'best.pt')}")


if __name__ == "__main__":
    main()
