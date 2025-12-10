// src/arena.ts

export const R = 250;

export type Vec2 = { x: number; y: number };

export type Agent = {
  pos: Vec2;
  heading: number; // 0..15
};

export type ArenaMode =
  | "uniform"
  | "right-dark"
  | "left-dark"
  | "single-sector-dark"
  | "except-one-sector-dark"
  | "checker";

const DIRS = 16;

function wrapHeading(h: number): number {
  const m = h % DIRS;
  return m < 0 ? m + DIRS : m;
}

// 8 = E(0), 4 = N(pi/2), 0 = W(pi), 12 = S(3pi/2)
export function headingToAngle(idx: number): number {
  return ((8 - wrapHeading(idx) + DIRS) % DIRS) * (2 * Math.PI / DIRS);
}

/**
 * Find the heading index (0..15) whose angle is closest to a given angle.
 * Used for sector-based contrast.
 */
function angleToNearestHeadingIndex(angle: number): number {
  // normalize to [0, 2π)
  let a = angle % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;

  let bestIdx = 0;
  let bestDist = Infinity;
  for (let idx = 0; idx < DIRS; idx++) {
    let hAng = headingToAngle(idx);
    // wrap-around difference
    let d = Math.abs(a - hAng);
    if (d > Math.PI) d = 2 * Math.PI - d;
    if (d < bestDist) {
      bestDist = d;
      bestIdx = idx;
    }
  }
  return bestIdx;
}

/**
 * Distance along the heading ray from pos to the circle boundary.
 */
function rayToCircleDistance(px: number, py: number, angle: number): number {
  const ux = Math.cos(angle);
  const uy = Math.sin(angle);
  const b = px * ux + py * uy;
  const c = px * px + py * py - R * R;
  const disc = b * b - c;
  if (disc <= 0) return 0;
  const t1 = -b + Math.sqrt(disc);
  return Math.max(0, t1);
}

/**
 * Contrast in front of the agent, with optional arena asymmetry:
 *  - "uniform": same everywhere
 *  - "right-dark": right half (x>0) darker
 *  - "left-dark": left half (x<0) darker
 *  - "single-sector-dark": only one of 16 radial sectors darker
 *  - "except-one-sector-dark": all but one sector darker
 *  - "checker": alternating dark/light sectors around the circle
 *
 * For sector-based modes we classify the *boundary hit point* into a
 * nearest heading index 0..15 (matching your W/WN1/.../E etc mapping).
 */
export function computeContrastFront(
  pos: Vec2,
  heading: number,
  wallContrast: number,
  mode: ArenaMode = "uniform",
  sectorIndex: number = 0
): number {
  const angle = headingToAngle(heading);
  const px = pos.x;
  const py = pos.y;

  // Distance to boundary + boundary point
  const d = rayToCircleDistance(px, py, angle);
  const dirX = Math.cos(angle);
  const dirY = Math.sin(angle);
  const bx = px + dirX * d;
  const by = py + dirY * d;

  // Which heading/sector does this boundary point correspond to?
  const boundaryHeadingIdx = angleToNearestHeadingIndex(Math.atan2(by, bx));

  // local contrast factor depending on mode
  let factor = 1.0;

  if (mode === "right-dark") {
    factor = bx > 0 ? 1.5 : 0.5;
  } else if (mode === "left-dark") {
    factor = bx < 0 ? 1.5 : 0.5;
  } else if (mode === "single-sector-dark") {
    factor = boundaryHeadingIdx === wrapHeading(sectorIndex) ? 1.5 : 0.5;
  } else if (mode === "except-one-sector-dark") {
    factor = boundaryHeadingIdx === wrapHeading(sectorIndex) ? 0.5 : 1.5;
  } else if (mode === "checker") {
    // even sectors dark, odd sectors light (checkerboard around the ring)
    factor = boundaryHeadingIdx % 2 === 0 ? 1.5 : 0.5;
  }

  const local = wallContrast * factor;

  const distanceFrontNorm = Math.max(Math.min(d / R, 1.0), 1e-6);
  const contrast = local / Math.max(distanceFrontNorm, 1e-3);
  return Math.max(0, Math.min(contrast, 1)); // clamp to [0,1]
}

/**
 * Turn-then-move step with collision correction.
 */
export function stepAgent(
  agent: Agent,
  L: 0 | 1,
  Rg: 0 | 1,
  P: 1 | 2 | 3
): { agent: Agent; collided: 0 | 1 } {
  let h = wrapHeading(agent.heading);

  // turning: +1 = clockwise (eastward), -1 = counter-clockwise
  if (L === 1 && Rg === 0) {
    h = wrapHeading(h + 1);
  } else if (L === 0 && Rg === 1) {
    h = wrapHeading(h - 1);
  }

  const ang = headingToAngle(h);
  let nx = agent.pos.x + Math.cos(ang) * P;
  let ny = agent.pos.y + Math.sin(ang) * P;

  let collided: 0 | 1 = 0;
  const r = Math.hypot(nx, ny);
  if (r > R) {
    const s = (R - 1) / r;
    nx *= s;
    ny *= s;
    collided = 1;
  }

  return {
    agent: {
      pos: { x: nx, y: ny },
      heading: h,
    },
    collided,
  };
}
