# evolve/policy.py
import torch
import torch.nn as nn
from dataclasses import dataclass

# Inputs: [contrast_front, collided_prev]  -> 2 dims
# Outputs: 6 logits: [turn_L, turn_R, turn_NONE, p1, p2, p3]
IN_DIM = 2
HID = 16
OUT_DIM = 6


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, HID), nn.Tanh(),
            nn.Linear(HID, OUT_DIM)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Policy:
    model: TinyMLP

    @torch.no_grad()
    def act(self, obs) -> tuple[int, int, int]:
        """
        obs is a StepObs from sim.py.

        We ONLY use:
          - obs.contrast_front  (float)
          - obs.collided_prev   (0/1)

        Positional info (heading) and J are NOT fed to the network anymore.
        """
        x = torch.tensor(
            [[
                float(obs.contrast_front),
                float(obs.collided_prev),
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

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


def new_policy() -> Policy:
    return Policy(TinyMLP())
