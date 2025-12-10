// src/App.tsx

import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import {
  R,
  computeContrastFront,
  stepAgent,
  type Agent,
  type ArenaMode,
  headingToAngle,
} from "./arena";
import { BrainWS } from "./ws";
import type { StepMsg } from "./types";
import { NetworkView } from "./NetworkView";

const WIDTH = 700;
const HEIGHT = 700;

// For labeling the 16 sectors consistent with your convention
const SECTOR_LABELS = [
  "0 – W",
  "1 – WN1",
  "2 – WN2",
  "3 – WN3",
  "4 – N",
  "5 – NE1",
  "6 – NE2",
  "7 – NE3",
  "8 – E",
  "9 – ES1",
  "10 – ES2",
  "11 – ES3",
  "12 – S",
  "13 – SW1",
  "14 – SW2",
  "15 – SW3",
];

function drawArena(
  ctx: CanvasRenderingContext2D,
  mode: ArenaMode,
  sectorIndex: number
) {
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  ctx.clearRect(0, 0, WIDTH, HEIGHT);

  // base circle
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#6ee7b7";
  ctx.stroke();

  // ticks
  ctx.save();
  ctx.translate(cx, cy);
  for (let i = 0; i < 16; i++) {
    const ang = (i / 16) * Math.PI * 2;
    const r1 = R - 8;
    const r2 = R;
    ctx.beginPath();
    ctx.moveTo(r1 * Math.cos(ang), r1 * Math.sin(ang));
    ctx.lineTo(r2 * Math.cos(ang), r2 * Math.sin(ang));
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#a7f3d0";
    ctx.stroke();
  }
  ctx.restore();

  // Helper to draw a sector arc centered at a heading index
  const drawSectorArc = (idx: number) => {
    const center = headingToAngle(idx); // 0..2π
    const halfSpan = Math.PI / 16; // sector width around center
    const start = center - halfSpan;
    const end = center + halfSpan;
    ctx.beginPath();
    ctx.arc(cx, cy, R, start, end);
    ctx.stroke();
  };

  // darker sectors: thicker, brighter strokes
  ctx.lineWidth = 5;
  ctx.strokeStyle = "#22c55e";

  if (mode === "right-dark") {
    // right half (x>0): angles from -π/2 to +π/2
    ctx.beginPath();
    ctx.arc(cx, cy, R, -Math.PI / 2, Math.PI / 2);
    ctx.stroke();
  } else if (mode === "left-dark") {
    // left half (x<0): angles from +π/2 to 3π/2
    ctx.beginPath();
    ctx.arc(cx, cy, R, Math.PI / 2, (3 * Math.PI) / 2);
    ctx.stroke();
  } else if (mode === "single-sector-dark") {
    drawSectorArc(sectorIndex);
  } else if (mode === "except-one-sector-dark") {
    for (let idx = 0; idx < 16; idx++) {
      if (idx === sectorIndex) continue; // this one is "lighter"
      drawSectorArc(idx);
    }
  } else if (mode === "checker") {
    // even sectors are "dark"
    for (let idx = 0; idx < 16; idx++) {
      if (idx % 2 === 0) {
        drawSectorArc(idx);
      }
    }
  }
}

function drawAgent(ctx: CanvasRenderingContext2D, agent: Agent) {
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  const ang = headingToAngle(agent.heading);
  const posx = agent.pos.x + cx;
  const posy = agent.pos.y + cy;

  // Body: circular jelly
  const bodyRadius = 10;
  ctx.beginPath();
  ctx.arc(posx, posy, bodyRadius, 0, Math.PI * 2);
  ctx.fillStyle = "#60a5fa";
  ctx.fill();

  // Facing indicator: small dot in the direction of heading
  const eyeRadius = 3;
  const eyeDistance = bodyRadius * 0.8;
  const ex = posx + Math.cos(ang) * eyeDistance;
  const ey = posy + Math.sin(ang) * eyeDistance;

  ctx.beginPath();
  ctx.arc(ex, ey, eyeRadius, 0, Math.PI * 2);
  ctx.fillStyle = "#1d4ed8";
  ctx.fill();
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 1e9));
  const [J, setJ] = useState(0.4);
  const [wallC, setWallC] = useState(0.6);
  const [connected, setConnected] = useState(false);
  const [tickHz, setTickHz] = useState(20);
  const [arenaMode, setArenaMode] = useState<ArenaMode>("uniform");
  const [sectorIndex, setSectorIndex] = useState<number>(0);

  // internal refs to avoid re-creating intervals
  const tRef = useRef(0);
  const collidedPrevRef = useRef<0 | 1>(0);
  const agentRef = useRef<Agent>({ pos: { x: 0, y: 0 }, heading: 8 }); // start east

  const [tDisplay, setTDisplay] = useState(0);
  const [collidedDisplay, setCollidedDisplay] = useState<0 | 1>(0);

  const ws = useMemo(() => new BrainWS("ws://localhost:8000/brain"), []);
  useEffect(() => {
    const off = ws.onMessage(() => {});
    const ping = setInterval(() => setConnected(ws.connected), 300);
    return () => {
      off();
      clearInterval(ping);
    };
  }, [ws]);

  const doReset = () => {
    agentRef.current = { pos: { x: 0, y: 0 }, heading: 8 };
    tRef.current = 0;
    collidedPrevRef.current = 0;
    setTDisplay(0);
    setCollidedDisplay(0);
    ws.send({ type: "reset", seed, J, wall_contrast: wallC });
  };

  useEffect(() => {
    doReset();
  }, []); // on mount

  // 60 FPS render
  useEffect(() => {
    let raf = 0;
    const ctx = canvasRef.current?.getContext("2d");
    const render = () => {
      if (!ctx) return;
      drawArena(ctx, arenaMode, sectorIndex);
      drawAgent(ctx, agentRef.current);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [arenaMode, sectorIndex]);

  // CONTROL LOOP
  useEffect(() => {
    const periodMs = Math.max(5, Math.floor(1000 / tickHz));
    let stopped = false;

    const loop = async () => {
      if (stopped) return;

      const agent = agentRef.current;
      const contrast = computeContrastFront(
        agent.pos,
        agent.heading,
        wallC,
        arenaMode,
        sectorIndex
      );

      const obs = {
        t: tRef.current + 1,
        j: J,
        contrast_front: Number(contrast.toFixed(6)),
        collided_prev: collidedPrevRef.current,
        heading_index: agent.heading,
      } as const;

      const stepMsg: StepMsg = { type: "step", obs };

      let L: 0 | 1 = 0;
      let R: 0 | 1 = 0;
      let P: 1 | 2 | 3 = 1;
      try {
        const act = await ws.stepAndWait(stepMsg, 200);
        if (act) {
          L = act.L;
          R = act.R;
          P = act.P;
        }
      } catch {
        // ignore; fallback L=0,R=0,P=1 already set
      }

      const { agent: newAgent, collided } = stepAgent(agent, L, R, P);
      agentRef.current = newAgent;
      collidedPrevRef.current = collided;

      tRef.current += 1;
      if ((tRef.current & 7) === 0) {
        setTDisplay(tRef.current);
        setCollidedDisplay(collidedPrevRef.current);
      }
    };

    const id = setInterval(loop, periodMs);
    return () => {
      stopped = true;
      clearInterval(id);
    };
  }, [tickHz, ws, J, wallC, arenaMode, sectorIndex]);

  const showSectorSelector =
    arenaMode === "single-sector-dark" || arenaMode === "except-one-sector-dark";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "700px 600px",
        gap: "16px",
        padding: 16,
      }}
    >
      {/* Left: Arena canvas */}
      <div>
        <canvas
          ref={canvasRef}
          width={WIDTH}
          height={HEIGHT}
          style={{
            border: "1px solid #0ea5e9",
            background: "#0b1220",
            borderRadius: 8,
          }}
        />
      </div>

      {/* Right: controls + NN */}
      <div
        style={{
          color: "#e5e7eb",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
        }}
      >
        <div>
          <h2 style={{ marginTop: 0 }}>Circle-Arena Jellyfish — Phase 0</h2>
          <div style={{ marginBottom: 12 }}>
            <button
              onClick={doReset}
              style={{
                padding: "8px 12px",
                borderRadius: 8,
                border: "1px solid #334155",
                background: "#111827",
                color: "#e5e7eb",
              }}
            >
              Reset
            </button>
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>Seed:&nbsp;</label>
            <input
              type="number"
              value={seed}
              onChange={(e) =>
                setSeed(parseInt(e.target.value || "0", 10) || 0)
              }
              style={{ width: 180 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>J:&nbsp;{J.toFixed(2)}</label>
            <br />
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={J}
              onChange={(e) => setJ(parseFloat(e.target.value))}
              style={{ width: 260 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>wall_contrast:&nbsp;{wallC.toFixed(2)}</label>
            <br />
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={wallC}
              onChange={(e) => setWallC(parseFloat(e.target.value))}
              style={{ width: 260 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>Tick rate (Hz):&nbsp;{tickHz}</label>
            <br />
            <input
              type="range"
              min={5}
              max={60}
              step={1}
              value={tickHz}
              onChange={(e) => setTickHz(parseInt(e.target.value, 10))}
              style={{ width: 260 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>Arena mode:&nbsp;</label>
            <select
              value={arenaMode}
              onChange={(e) =>
                setArenaMode(e.target.value as ArenaMode)
              }
              style={{
                padding: "4px 8px",
                borderRadius: 6,
                background: "#020617",
                color: "#e5e7eb",
                border: "1px solid #334155",
              }}
            >
              <option value="uniform">Uniform (default)</option>
              <option value="right-dark">Right side darker</option>
              <option value="left-dark">Left side darker</option>
              <option value="single-sector-dark">
                Single darker sector
              </option>
              <option value="except-one-sector-dark">
                All except one darker
              </option>
              <option value="checker">Checker (alternating sectors)</option>
            </select>
          </div>

          {showSectorSelector && (
            <div style={{ margin: "8px 0" }}>
              <label>Sector index:&nbsp;</label>
              <select
                value={sectorIndex}
                onChange={(e) => setSectorIndex(parseInt(e.target.value, 10))}
                style={{
                  padding: "4px 8px",
                  borderRadius: 6,
                  background: "#020617",
                  color: "#e5e7eb",
                  border: "1px solid #334155",
                }}
              >
                {SECTOR_LABELS.map((label, idx) => (
                  <option key={idx} value={idx}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
          )}

          <hr style={{ borderColor: "#1f2937" }} />

          <p>
            <b>WS:</b> {connected ? "connected ✅" : "reconnecting…"}
            <br />
            <b>t:</b> {tDisplay} &nbsp;|&nbsp; <b>collided_prev:</b>{" "}
            {collidedDisplay}
          </p>

          <p style={{ opacity: 0.8, fontSize: 14 }}>
            First turns then moves, demo 0 random / policy movement
          </p>
        </div>

        <NetworkView />
      </div>
    </div>
  );
}
