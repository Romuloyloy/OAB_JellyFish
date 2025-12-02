import math
import random
from typing import Dict, Any, Optional

# Optional: if you put policy.py next to this file, import from there.
# Otherwise this inline TinyMLP+Policy matches evolve/policy.py exactly.
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


IN_DIM = 5   # [contrast, collided_prev, J, cosθ, sinθ]
HID = 16
OUT_DIM = 6  # [turn_L, turn_R, turn_NONE, p1, p2, p3]


def _heading_to_angle(idx: int) -> float:
    # Mapping: 8=E(0), 4=N(pi/2), 0=W(pi), 12=S(3pi/2)
    return ((8 - (idx % 16) + 16) % 16) * (2 * math.pi / 16)


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, HID), nn.Tanh(),
            nn.Linear(HID, OUT_DIM)
        )

    def forward(self, x):
        return self.net(x)


class _PolicyWrapper:
    def __init__(self):
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch to use policy mode.")
        self.model = _TinyMLP()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    @torch.no_grad()
    def act(self, obs: Dict[str, Any]) -> (int, int, int):
        # Build input vector in the same order as training
        ang = _heading_to_angle(int(obs.get("heading_index", 8)))
        x = torch.tensor([[float(obs.get("contrast_front", 0.0)),
                           float(obs.get("collided_prev", 0)),
                           float(obs.get("j", obs.get("J", 0.0))),  # allow 'j' or 'J'
                           math.cos(ang), math.sin(ang)]],
                         dtype=torch.float32)
        logits = self.model(x)[0]
        # First 3 logits: turn choice
        turn = int(torch.argmax(logits[:3]).item())   # 0:L, 1:R, 2:NONE
        # Last 3 logits: pace choice
        pace = int(torch.argmax(logits[3:]).item())   # 0->P=1, 1->P=2, 2->P=3
        L = 1 if turn == 0 else 0
        R = 1 if turn == 1 else 0
        P = pace + 1
        return L, R, P


class Brain:
    """
    Brain with two modes:
      - 'random' (default): Phase 0 random policy.
      - 'policy': load a TinyMLP from a saved state_dict (.pt) and use it.

    Random policy:
      - L,R ∈ {(1,0),(0,1),(0,0)} with probs [0.3, 0.3, 0.4]
      - P ∈ {1,2,3} uniform
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.J: float = 0.0
        self.wall_contrast: float = 0.0
        self.mode: str = "random"
        self.policy: Optional[_PolicyWrapper] = None
        self.policy_path: Optional[str] = None

    # --- lifecycle ---
    def reset(self, J: float, wall_contrast: float) -> None:
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

    # --- stepping ---
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "policy" and self.policy is not None:
            L, R, P = self.policy.act({
                # ensure keys are present exactly as training expected
                "contrast_front": obs.get("contrast_front", 0.0),
                "collided_prev": obs.get("collided_prev", 0),
                "j": obs.get("j", obs.get("J", self.J)),
                "heading_index": obs.get("heading_index", 8),
            })
            return {
                "L": L, "R": R, "P": P,
                "debug": {
                    "mode": "policy",
                    "J": self.J,
                    "wall_contrast": self.wall_contrast,
                    "loaded": bool(self.policy),
                    "policy_path": self.policy_path,
                    "obs_contrast": obs.get("contrast_front"),
                    "obs_collided_prev": obs.get("collided_prev"),
                }
            }

        # fallback: random (Phase 0)
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
            }
        }
