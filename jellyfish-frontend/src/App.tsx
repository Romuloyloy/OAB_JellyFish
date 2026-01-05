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

const SECTOR_LABELS = [
  "0 – W","1 – WN1","2 – WN2","3 – WN3","4 – N","5 – NE1","6 – NE2","7 – NE3",
  "8 – E","9 – ES1","10 – ES2","11 – ES3","12 – S","13 – SW1","14 – SW2","15 – SW3",
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

  // Helper: draw sector arc centered at a heading index
  const drawSectorArc = (idx: number) => {
    const center = headingToAngle(idx);
    const halfSpan = Math.PI / 16;
    const start = center - halfSpan;
    const end = center + halfSpan;
    ctx.beginPath();
    ctx.arc(cx, cy, R, start, end);
    ctx.stroke();
  };

  // darker sectors indicator
  ctx.lineWidth = 5;
  ctx.strokeStyle = "#22c55e";

  if (mode === "right-dark") {
    ctx.beginPath();
    ctx.arc(cx, cy, R, -Math.PI / 2, Math.PI / 2);
    ctx.stroke();
  } else if (mode === "left-dark") {
    ctx.beginPath();
    ctx.arc(cx, cy, R, Math.PI / 2, (3 * Math.PI) / 2);
    ctx.stroke();
  } else if (mode === "single-sector-dark") {
    drawSectorArc(sectorIndex);
  } else if (mode === "except-one-sector-dark") {
    for (let idx = 0; idx < 16; idx++) {
      if (idx === sectorIndex) continue;
      drawSectorArc(idx);
    }
  } else if (mode === "checker") {
    for (let idx = 0; idx < 16; idx++) {
      if (idx % 2 === 0) drawSectorArc(idx);
    }
  }
}

function drawAgent(ctx: CanvasRenderingContext2D, agent: Agent) {
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  const ang = headingToAngle(agent.heading);
  const posx = agent.pos.x + cx;
  const posy = agent.pos.y + cy;

  // body
  const bodyRadius = 10;
  ctx.beginPath();
  ctx.arc(posx, posy, bodyRadius, 0, Math.PI * 2);
  ctx.fillStyle = "#60a5fa";
  ctx.fill();

  // facing indicator dot
  const eyeRadius = 3;
  const eyeDistance = bodyRadius * 0.8;
  const ex = posx + Math.cos(ang) * eyeDistance;
  const ey = posy + Math.sin(ang) * eyeDistance;

  ctx.beginPath();
  ctx.arc(ex, ey, eyeRadius, 0, Math.PI * 2);
  ctx.fillStyle = "#1d4ed8";
  ctx.fill();
}

function tsName() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
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

  // NN visualization states
  const [inputActs, setInputActs] = useState<number[] | null>(null);
  const [hiddenActs, setHiddenActs] = useState<number[] | null>(null);
  const [hiddenUsage, setHiddenUsage] = useState<number[] | null>(null);
  const [logits, setLogits] = useState<number[] | null>(null);

  // UI
  const [contrastDisplay, setContrastDisplay] = useState<number>(0);

  // trajectory tracking
  const [tracking, setTracking] = useState(false);
  const trailCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastTrailPosRef = useRef<{ x: number; y: number } | null>(null);

  // internal refs
  const tRef = useRef(0);
  const collidedPrevRef = useRef<0 | 1>(0);
  const agentRef = useRef<Agent>({ pos: { x: 0, y: 0 }, heading: 8 }); // start east

  const [tDisplay, setTDisplay] = useState(0);
  const [collidedDisplay, setCollidedDisplay] = useState<0 | 1>(0);

  // setup offscreen trail canvas once
  useEffect(() => {
    const c = document.createElement("canvas");
    c.width = WIDTH;
    c.height = HEIGHT;
    trailCanvasRef.current = c;
    // init clean
    const tctx = c.getContext("2d");
    if (tctx) tctx.clearRect(0, 0, WIDTH, HEIGHT);
  }, []);

  const ws = useMemo(() => new BrainWS("ws://localhost:8000/brain"), []);
  useEffect(() => {
    const off = ws.onMessage(() => {});
    const ping = setInterval(() => setConnected(ws.connected), 300);
    return () => {
      off();
      clearInterval(ping);
    };
  }, [ws]);

  const clearTrail = () => {
    const tc = trailCanvasRef.current;
    const tctx = tc?.getContext("2d");
    if (tctx) tctx.clearRect(0, 0, WIDTH, HEIGHT);
    lastTrailPosRef.current = null;
  };

  const startTrajectory = () => {
    clearTrail();
    setTracking(true);
    // seed trail start at current agent position
    lastTrailPosRef.current = { ...agentRef.current.pos };
  };

  const stopAndSaveTrajectory = () => {
    setTracking(false);

    const tc = trailCanvasRef.current;
    if (!tc) return;

    // download PNG
    const url = tc.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = `trajectory_${tsName()}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    // clear traces after saving
    clearTrail();
  };

  const doReset = () => {
    agentRef.current = { pos: { x: 0, y: 0 }, heading: 8 };
    tRef.current = 0;
    collidedPrevRef.current = 0;
    setTDisplay(0);
    setCollidedDisplay(0);

    // reset NN view
    setInputActs(null);
    setHiddenActs(null);
    setHiddenUsage(null);
    setLogits(null);
    setContrastDisplay(0);

    // stop tracking + clear trail
    setTracking(false);
    clearTrail();

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

      // draw arena
      drawArena(ctx, arenaMode, sectorIndex);

      // overlay trail if any
      const tc = trailCanvasRef.current;
      if (tc) {
        ctx.drawImage(tc, 0, 0);
      }

      // draw agent on top
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

      // inputs to NN view
      setInputActs([obs.contrast_front, obs.collided_prev]);
      setContrastDisplay(obs.contrast_front);

      const stepMsg: StepMsg = { type: "step", obs };

      let L: 0 | 1 = 0;
      let Rv: 0 | 1 = 0;
      let P: 1 | 2 | 3 = 1;

      try {
        const act: any = await ws.stepAndWait(stepMsg, 200);
        if (act) {
          L = act.L;
          Rv = act.R;
          P = act.P;

          const dbg = act.debug;
          if (dbg) {
            if (Array.isArray(dbg.hidden)) setHiddenActs(dbg.hidden);
            if (Array.isArray(dbg.hidden_usage)) setHiddenUsage(dbg.hidden_usage);
            if (Array.isArray(dbg.logits)) setLogits(dbg.logits);
          }
        }
      } catch {
        // ignore; fallback action already set
      }

      const prevPos = { ...agentRef.current.pos };
      const { agent: newAgent, collided } = stepAgent(agent, L, Rv, P);
      agentRef.current = newAgent;
      collidedPrevRef.current = collided;

      // trajectory draw (only if tracking)
      if (tracking) {
        const tc = trailCanvasRef.current;
        const tctx = tc?.getContext("2d");
        if (tctx) {
          const cx = WIDTH / 2;
          const cy = HEIGHT / 2;

          // line from last -> new
          const last = lastTrailPosRef.current ?? prevPos;
          const a = { x: last.x + cx, y: last.y + cy };
          const b = { x: newAgent.pos.x + cx, y: newAgent.pos.y + cy };

          tctx.beginPath();
          tctx.moveTo(a.x, a.y);
          tctx.lineTo(b.x, b.y);
          tctx.lineWidth = 2;
          tctx.strokeStyle = "rgba(251, 191, 36, 0.65)"; // amber-ish
          tctx.stroke();

          lastTrailPosRef.current = { ...newAgent.pos };
        }
      }

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
  }, [tickHz, ws, J, wallC, arenaMode, sectorIndex, tracking]);

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
      <div style={{ color: "#e5e7eb", display: "flex", flexDirection: "column", gap: 16 }}>
        <div>
          <h2 style={{ marginTop: 0 }}>Circle-Arena Jellyfish — Phase 0</h2>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
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

            {!tracking ? (
              <button
                onClick={startTrajectory}
                style={{
                  padding: "8px 12px",
                  borderRadius: 8,
                  border: "1px solid #334155",
                  background: "#0b1220",
                  color: "#e5e7eb",
                }}
              >
                Start Trajectory
              </button>
            ) : (
              <button
                onClick={stopAndSaveTrajectory}
                style={{
                  padding: "8px 12px",
                  borderRadius: 8,
                  border: "1px solid #334155",
                  background: "#1f2937",
                  color: "#e5e7eb",
                }}
              >
                Stop + Save Trajectory
              </button>
            )}

            <button
              onClick={clearTrail}
              style={{
                padding: "8px 12px",
                borderRadius: 8,
                border: "1px solid #334155",
                background: "#020617",
                color: "#e5e7eb",
              }}
            >
              Clear Traces
            </button>
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>Seed:&nbsp;</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value || "0", 10) || 0)}
              style={{ width: 180 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>J:&nbsp;{J.toFixed(2)}</label><br />
            <input
              type="range" min={0} max={1} step={0.01}
              value={J}
              onChange={(e) => setJ(parseFloat(e.target.value))}
              style={{ width: 260 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>wall_contrast:&nbsp;{wallC.toFixed(2)}</label><br />
            <input
              type="range" min={0} max={1} step={0.01}
              value={wallC}
              onChange={(e) => setWallC(parseFloat(e.target.value))}
              style={{ width: 260 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>Tick rate (Hz):&nbsp;{tickHz}</label><br />
            <input
              type="range" min={5} max={60} step={1}
              value={tickHz}
              onChange={(e) => setTickHz(parseInt(e.target.value, 10))}
              style={{ width: 260 }}
            />
          </div>

          <div style={{ margin: "8px 0" }}>
            <label>Arena mode:&nbsp;</label>
            <select
              value={arenaMode}
              onChange={(e) => setArenaMode(e.target.value as ArenaMode)}
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
              <option value="single-sector-dark">Single darker sector</option>
              <option value="except-one-sector-dark">All except one darker</option>
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
                  <option key={idx} value={idx}>{label}</option>
                ))}
              </select>
            </div>
          )}

          <hr style={{ borderColor: "#1f2937" }} />

          <p>
            <b>WS:</b> {connected ? "connected ✅" : "reconnecting…"}<br />
            <b>t:</b> {tDisplay} &nbsp;|&nbsp; <b>collided_prev:</b> {collidedDisplay}<br />
            <b>contrast_front:</b> {contrastDisplay.toFixed(3)}<br />
            <b>trajectory:</b> {tracking ? "recording ✍️" : "off"}
          </p>
        </div>

        <NetworkView
          inputActs={inputActs}
          hiddenActs={hiddenActs}
          hiddenUsage={hiddenUsage}
          logits={logits}
          contrast={contrastDisplay}
        />
      </div>
    </div>
  );
}
