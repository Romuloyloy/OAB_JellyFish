import math
import random
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# Inputs: [contrast_front, collided_prev]
IN_DIM = 2
HID = 16
OUT_DIM = 6  # [turn_L, turn_R, turn_NONE, p1, p2, p3]


if nn is not None:

    class _TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(IN_DIM, HID), nn.Tanh(),
                nn.Linear(HID, OUT_DIM)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    # Dummy placeholder so module still imports if torch is missing
    class _TinyMLP:  # type: ignore
        def __init__(self):
            raise RuntimeError("PyTorch not available. Install torch to use policy mode.")


class _PolicyWrapper:
    """
    Wrap TinyMLP so Brain can call .act(obs) directly.
    """
    def __init__(self):
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch to use policy mode.")
        self.model = _TinyMLP()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    @torch.no_grad()
    def act(self, obs: Dict[str, Any]) -> tuple[int, int, int]:
        """
        Runtime policy input:

          - contrast_front    (float)
          - collided_prev     (0/1)

        J and positional info (heading) are deliberately NOT fed to the NN.
        """
        x = torch.tensor(
            [[
                float(obs.get("contrast_front", 0.0)),
                float(obs.get("collided_prev", 0)),
            ]],
            dtype=torch.float32,
        )
        logits = self.model(x)[0]

        # First 3 logits: turn choice
        turn = int(torch.argmax(logits[:3]).item())   # 0:L, 1:R, 2:NONE
        # Last 3 logits: pace choice
        pace = int(torch.argmax(logits[3:]).item())   # 0→P=1, 1→P=2, 2→P=3

        L = 1 if turn == 0 else 0
        R = 1 if turn == 1 else 0
        P = pace + 1
        return L, R, P


class Brain:
    """
    Brain with two modes:

      - 'random' (default): Phase 0 random policy
        L,R ∈ {(1,0),(0,1),(0,0)} with probs [0.3, 0.3, 0.4], P ∈ {1,2,3} uniform.

      - 'policy': TinyMLP loaded from a torch state_dict (.pt file).

    NOTE: The NN now only sees [contrast_front, collided_prev].
          J and heading/orientation are NOT inputs to the network.
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.J: float = 0.0            # kept for logging / UI, but not used as NN input
        self.wall_contrast: float = 0.0
        self.mode: str = "random"
        self.policy: Optional[_PolicyWrapper] = None
        self.policy_path: Optional[str] = None

    # --- lifecycle ---
    def reset(self, J: float, wall_contrast: float) -> None:
        # J still exists as an environment parameter, but NN ignores it.
        self.J = float(J)
        self.wall_contrast = float(wall_contrast)

    # --- control ---
    def set_mode(self, mode: str) -> None:
        mode = mode.lower().strip()
        if mode not in ("random", "policy"):
            raise ValueError(f"Unknown mode '{mode}'")
        if mode == "policy" and self.policy is None:
            raise RuntimeError("Policy mode requested but no policy loaded. Call load_policy(path) first.")
        self.mode = mode

    def load_policy(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch to load policies.")
        sd = torch.load(path, map_location="cpu")
        pol = _PolicyWrapper()
        pol.load_state_dict(sd)
        self.policy = pol
        self.policy_path = path
        self.mode = "policy"

    def describe_network(self) -> dict:
        """
        Return a JSON-serializable description of the current policy network.
        Used by /nn for frontend visualization.
        """
        if self.policy is None or not hasattr(self.policy, "model"):
            return {
                "mode": self.mode,
                "has_policy": False,
            }

        model = self.policy.model
        # Sequential(Linear(IN_DIM, HID), Tanh, Linear(HID, OUT_DIM))
        layer1 = model.net[0]
        layer2 = model.net[2]

        w1 = layer1.weight.detach().cpu().tolist()   # [hidden][input]
        b1 = layer1.bias.detach().cpu().tolist()
        w2 = layer2.weight.detach().cpu().tolist()   # [output][hidden]
        b2 = layer2.bias.detach().cpu().tolist()

        input_labels = [
            "contrast_front",
            "collided_prev",
        ]
        output_labels = [
            "turn_L",
            "turn_R",
            "turn_NONE",
            "P=1",
            "P=2",
            "P=3",
        ]

        return {
            "mode": self.mode,
            "has_policy": True,
            "input_labels": input_labels,
            "hidden_size": len(b1),
            "output_labels": output_labels,
            "layers": {
                "input_to_hidden": {
                    "weights": w1,
                    "bias": b1,
                },
                "hidden_to_output": {
                    "weights": w2,
                    "bias": b2,
                },
            },
        }

    # --- stepping ---
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Policy mode (NN)
        if self.mode == "policy" and self.policy is not None:
            L, R, P = self.policy.act({
                "contrast_front": obs.get("contrast_front", 0.0),
                "collided_prev": obs.get("collided_prev", 0),
            })
            return {
                "L": L,
                "R": R,
                "P": P,
                "debug": {
                    "mode": "policy",
                    "J": self.J,
                    "wall_contrast": self.wall_contrast,
                    "loaded": bool(self.policy),
                    "policy_path": self.policy_path,
                    "obs_contrast": obs.get("contrast_front"),
                    "obs_collided_prev": obs.get("collided_prev"),
                },
            }

        # Fallback: random Phase 0 policy
        lr_choices = [(1, 0), (0, 1), (0, 0)]
        lr_probs = [0.3, 0.3, 0.4]
        r = self.rng.random()
        if r < lr_probs[0]:
            L, R = lr_choices[0]
        elif r < lr_probs[0] + lr_probs[1]:
            L, R = lr_choices[1]
        else:
            L, R = lr_choices[2]
        P = self.rng.choice([1, 2, 3])

        return {
            "L": L,
            "R": R,
            "P": P,
            "debug": {
                "mode": "random",
                "J": self.J,
                "wall_contrast": self.wall_contrast,
                "obs_contrast": obs.get("contrast_front"),
                "obs_collided_prev": obs.get("collided_prev"),
            },
        }
