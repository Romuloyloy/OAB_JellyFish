// Physics, geometry, contrast

export const R = 250;
const EPS = 1;
const DIRS = 16;

export type Vec2 = { x: number, y: number };
export type Agent = { pos: Vec2; heading: number };

function len(v: Vec2) { return Math.hypot(v.x, v.y); }
function scale(v: Vec2, s: number): Vec2 { return { x: v.x * s, y: v.y * s }; }
function add(a: Vec2, b: Vec2): Vec2 { return { x: a.x + b.x, y: a.y + b.y }; }
function dot(a: Vec2, b: Vec2) { return a.x*b.x + a.y*b.y; }

export function headingToAngle(idx: number): number {
  const step = (2*Math.PI)/DIRS;
  return (8 - (idx % DIRS) + DIRS) % DIRS * step;
}

function angleToUnit(theta: number): Vec2 {
  return { x: Math.cos(theta), y: Math.sin(theta) };
}

function clampInside(pos: Vec2): Vec2 {
  const L = len(pos);
  if (L <= R) return pos;
  const s = (R - EPS) / L;
  return scale(pos, s);
}

function rayToCircleDistance(pos: Vec2, angle: number): number {
  const u = angleToUnit(angle);
  const b = dot(pos, u);
  const c = dot(pos, pos) - R * R;
  const disc = b*b - c;
  if (disc <= 0) return 0;
  const t1 = -b + Math.sqrt(disc);
  return Math.max(0, t1);
}

export function computeContrastFront(pos: Vec2, headingIdx: number, wall_contrast: number): number {
  const angle = headingToAngle(headingIdx);
  const d = rayToCircleDistance(pos, angle);
  const distance_front = Math.max(Math.min(d / R, 1), 1e-6); // (0,1]
  const contrast_front = Math.max(0, Math.min(wall_contrast / Math.max(distance_front, 1e-3), 1));
  return contrast_front;
}

export function stepAgent(agent: Agent, L: 0|1, Rv: 0|1, P: 1|2|3): { agent: Agent, collided: 0|1 } {
  let h = agent.heading;
  if (L === 1 && Rv === 0) h = (h + 1) & 15;
  else if (L === 0 && Rv === 1) h = (h + 15) & 15;

  const ang = headingToAngle(h);
  const dir = angleToUnit(ang);
  const newPos = add(agent.pos, scale(dir, P));

  const collided = (len(newPos) > R) ? 1 : 0;
  const fixed = collided ? clampInside(newPos) : newPos;
  return { agent: { pos: fixed, heading: h }, collided };
}
