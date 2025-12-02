# evolve/sim.py
import math
from dataclasses import dataclass

# Base radius used for normalization & for the "canonical" circle (frontend uses 250)
ARENA_BASE_R = 250.0
EPS = 1.0

DIRS = 16
STEP_ANGLE = 2 * math.pi / DIRS


def heading_to_angle(idx: int) -> float:
    # 8=E(0), 4=N(pi/2), 0=W(pi), 12=S(3pi/2)
    return ((8 - (idx % DIRS) + 16) % 16) * STEP_ANGLE


def ray_to_ellipse_distance(px: float, py: float, angle: float, a: float, b: float) -> float:
    """
    Distance t >= 0 along ray p + t u until it hits ellipse x^2/a^2 + y^2/b^2 = 1.
    Return 0 if no valid positive solution (shouldn't really happen for reasonable p inside ellipse).
    """
    ux, uy = math.cos(angle), math.sin(angle)

    A = (ux * ux) / (a * a) + (uy * uy) / (b * b)
    B = 2.0 * (px * ux / (a * a) + py * uy / (b * b))
    C = (px * px) / (a * a) + (py * py) / (b * b) - 1.0

    disc = B * B - 4.0 * A * C
    if disc <= 0 or A == 0.0:
        return 0.0

    sqrt_disc = math.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)

    candidates = [t for t in (t1, t2) if t > 0]
    if not candidates:
        return 0.0
    return min(candidates)


def radial_distance_to_boundary(px: float, py: float, a: float, b: float) -> float:
    """
    Distance to ellipse boundary along the radial direction from origin.

    Solve for r in:
        (r cos(theta)/a)^2 + (r sin(theta)/b)^2 = 1
      => r = 1 / sqrt(cos^2(theta)/a^2 + sin^2(theta)/b^2)

    Return d_wall = r_boundary - r_current.
    """
    r_current = math.hypot(px, py)
    if r_current == 0.0:
        # at center; distance is just radius along any direction (take theta=0)
        r_boundary = a  # along x-axis
        return r_boundary

    theta = math.atan2(py, px)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    denom = math.sqrt((cos_t * cos_t) / (a * a) + (sin_t * sin_t) / (b * b))
    if denom == 0.0:
        return 0.0

    r_boundary = 1.0 / denom
    return r_boundary - r_current


def inside_ellipse(x: float, y: float, a: float, b: float) -> bool:
    return (x * x) / (a * a) + (y * y) / (b * b) <= 1.0


@dataclass
class Agent:
    x: float = 0.0
    y: float = 0.0
    h: int = 8   # heading index; start east


@dataclass
class StepObs:
    t: int
    j: float
    contrast_front: float
    collided_prev: int
    heading_index: int


class Arena:
    """
    Arena with ellipse boundary:

      x^2 / a^2 + y^2 / b^2 <= 1

    a,b are derived from:
      - ARENA_BASE_R
      - R_scale (size)
      - aspect (ellipticity): >1 → stretched in x, <1 → stretched in y.

    For the frontend circle, we later just use a=b=ARENA_BASE_R.
    """

    def __init__(self, J: float, wall_contrast: float,
                 seed: int = 0,
                 R_scale: float = 1.0,
                 aspect: float = 1.0):
        self.J = float(J)
        self.wall_contrast = float(wall_contrast)
        self.t = 0
        self.collided_prev = 0
        self.agent = Agent()

        # Geometry
        R_eff = ARENA_BASE_R * R_scale
        # Keep area roughly stable by stretching/compressing axes inversely
        self.a = R_eff * aspect
        self.b = R_eff / aspect

    def reset(self):
        self.t = 0
        self.collided_prev = 0
        self.agent = Agent()

    def contrast_front(self) -> float:
        ang = heading_to_angle(self.agent.h)
        d = ray_to_ellipse_distance(self.agent.x, self.agent.y, ang, self.a, self.b)
        # normalize by base radius to keep scale similar to frontend
        distance_front = max(min(d / ARENA_BASE_R, 1.0), 1e-6)
        return max(0.0, min(self.wall_contrast / max(distance_front, 1e-3), 1.0))

    def observe(self) -> StepObs:
        return StepObs(
            t=self.t,
            j=self.J,
            contrast_front=self.contrast_front(),
            collided_prev=self.collided_prev,
            heading_index=self.agent.h
        )

    def step(self, L: int, Rv: int, P: int):
        # turn
        h = self.agent.h
        if L == 1 and Rv == 0:
            h = (h + 1) & 15
        elif L == 0 and Rv == 1:
            h = (h + 15) & 15
        self.agent.h = h

        # move
        ang = heading_to_angle(h)
        self.agent.x += math.cos(ang) * P
        self.agent.y += math.sin(ang) * P

        # collision correction with ellipse boundary
        if inside_ellipse(self.agent.x, self.agent.y, self.a, self.b):
            collided = 0
        else:
            collided = 1
            # project back along radial line from origin
            r_current = math.hypot(self.agent.x, self.agent.y)
            if r_current > 0.0:
                # radial distance to boundary + tiny epsilon inward
                d_wall = radial_distance_to_boundary(self.agent.x, self.agent.y, self.a, self.b)
                r_boundary = r_current + d_wall
                # clamp slightly inside boundary
                s = max((r_boundary - EPS) / r_current, 0.0)
                self.agent.x *= s
                self.agent.y *= s

        # tick
        self.t += 1
        self.collided_prev = collided
        return collided
