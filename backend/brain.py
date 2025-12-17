import math
import random
from typing import Dict, Any, Optional

# Matches evolve/policy.py AFTER you removed J/positional info:
# Inputs: [contrast_front, collided_prev] -> 2 dims
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

IN_DIM = 2   # [contrast_front, collided_prev]
HID = 16
OUT_DIM = 6  # [turn_L, turn_R, turn_NONE, P1, P2, P3]


def _heading_to_angle(idx: int) -> float:
    """
    Kept for future use (if you later reintroduce angle features).
    Current policy does NOT use angle, so this is unused in act_with_debug.
    Mapping: 8=E(0), 4=N(pi/2), 0=W(pi), 12=S(3pi/2)
    """
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
    """
    Thin wrapper around TinyMLP that can:
      - load weights
      - run a forward pass with access to hidden activations + logits

    IMPORTANT: Input is exactly 2-D:
        x = [contrast_front, collided_prev]
    """
    def __init__(self):
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch to use policy mode.")
        self.model = _TinyMLP()
        # unpack layers so we can peek inside
        self.l1 = self.model.net[0]  # Linear(IN_DIM, HID)
        self.l2 = self.model.net[2]  # Linear(HID, OUT_DIM)

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    @torch.no_grad()
    def act_with_debug(self, obs: Dict[str, Any]):
        """
        Forward pass that returns:
          (L, R, P, hidden_activations, logits)

        Input vector:
          x = [contrast_front, collided_prev]
        """
        cf = float(obs.get("contrast_front", 0.0))
        cp = float(obs.get("collided_prev", 0))

        x = torch.tensor([[cf, cp]], dtype=torch.float32)

        # manual forward so we can inspect
        h1 = self.l1(x)
        a1 = torch.tanh(h1)      # hidden activations
        out = self.l2(a1)        # logits

        hidden = a1[0].cpu().tolist()   # length HID
        logits = out[0].cpu().tolist()  # length OUT_DIM

        # Decision rule (same as before, just using 'out')
        turn = int(torch.argmax(out[0, :3]).item())   # 0:L, 1:R, 2:NONE
        pace = int(torch.argmax(out[0, 3:]).item())   # 0->1,1->2,2->3

        L = 1 if turn == 0 else 0
        R = 1 if turn == 1 else 0
        P = pace + 1

        return L, R, P, hidden, logits


class Brain:
    """
    Brain with two modes:
      - 'random' (default): Phase 0 random policy.
      - 'policy': load a TinyMLP from a saved state_dict (.pt) and use it.

    Random policy:
      - L,R ∈ {(1,0),(0,1),(0,0)} with probs [0.3, 0.3, 0.4]
      - P ∈ {1,2,3} uniform

    In 'policy' mode, we also expose hidden activations + usage and logits
    to the frontend via the 'debug' field.
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.J: float = 0.0
        self.wall_contrast: float = 0.0
        self.mode: str = "random"
        self.policy: Optional[_PolicyWrapper] = None
        self.policy_path: Optional[str] = None

        # cumulative |activation| for each hidden unit (length HID)
        self.hidden_usage = [0.0] * HID

    # --- lifecycle ---

    def reset(self, J: float, wall_contrast: float) -> None:
        # J kept only for logging / potential future use; NOT fed to NN.
        self.J = float(J)
        self.wall_contrast = float(wall_contrast)
        # reset usage at episode start
        self.hidden_usage = [0.0] * HID

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
        Assumes policy.model is TinyMLP with net = [Linear, Tanh, Linear].
        Used by the /nn endpoint and the frontend NetworkView.
        """
        if self.policy is None or not hasattr(self.policy, "model"):
            return {
                "mode": self.mode,
                "has_policy": False,
            }

        model = self.policy.model
        # We assume: Sequential(Linear(IN_DIM, HID), Tanh, Linear(HID, OUT_DIM))
        layer1 = model.net[0]
        layer2 = model.net[2]

        import torch as _t

        w1 = layer1.weight.detach().cpu().tolist()   # [HID, IN_DIM]
        b1 = layer1.bias.detach().cpu().tolist()     # [HID]
        w2 = layer2.weight.detach().cpu().tolist()   # [OUT_DIM, HID]
        b2 = layer2.bias.detach().cpu().tolist()     # [OUT_DIM]

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
                    "weights": w1,  # [hidden][input]
                    "bias": b1,
                },
                "hidden_to_output": {
                    "weights": w2,  # [output][hidden]
                    "bias": b2,
                },
            },
        }

    # --- stepping ---

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "policy" and self.policy is not None:
            # forward pass with introspection
            L, R, P, hidden, logits = self.policy.act_with_debug({
                "contrast_front": obs.get("contrast_front", 0.0),
                "collided_prev": obs.get("collided_prev", 0),
            })

            # update usage
            self.hidden_usage = [
                u + abs(a) for u, a in zip(self.hidden_usage, hidden)
            ]

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
                    "hidden": hidden,                    # current hidden activations
                    "logits": logits,                    # output logits
                    "hidden_usage": self.hidden_usage,   # cumulative |act|
                },
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
            },
        }
