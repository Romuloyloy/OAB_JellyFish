import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { R, computeContrastFront, stepAgent } from "./arena";
import { BrainWS } from "./ws";
import type { StepMsg } from "./types";
import type { Agent } from "./arena";

const WIDTH = 700;
const HEIGHT = 700;

function drawArena(ctx: CanvasRenderingContext2D) {
  const cx = WIDTH/2, cy = HEIGHT/2;
  ctx.clearRect(0,0,WIDTH,HEIGHT);

  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI*2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#6ee7b7";
  ctx.stroke();

  ctx.save();
  ctx.translate(cx, cy);
  for (let i=0;i<16;i++){
    const ang = (i/16) * Math.PI*2;
    const r1 = R - 8, r2 = R;
    ctx.beginPath();
    ctx.moveTo(r1*Math.cos(ang), r1*Math.sin(ang));
    ctx.lineTo(r2*Math.cos(ang), r2*Math.sin(ang));
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#a7f3d0";
    ctx.stroke();
  }
  ctx.restore();
}

function drawAgent(ctx: CanvasRenderingContext2D, agent: Agent) {
  const cx = WIDTH/2, cy = HEIGHT/2;
  const ang = (8 - agent.heading) * (Math.PI/8);
  const posx = agent.pos.x + cx, posy = agent.pos.y + cy;

  const size = 10;
  const left = ang + Math.PI*0.8;
  const right = ang - Math.PI*0.8;

  const x1 = posx + Math.cos(ang)*size*1.6;
  const y1 = posy + Math.sin(ang)*size*1.6;
  const x2 = posx + Math.cos(left)*size;
  const y2 = posy + Math.sin(left)*size;
  const x3 = posx + Math.cos(right)*size;
  const y3 = posy + Math.sin(right)*size;

  ctx.beginPath();
  ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x3,y3); ctx.closePath();
  ctx.fillStyle = "#60a5fa"; ctx.fill();

  ctx.beginPath();
  ctx.arc(x1,y1,2,0,Math.PI*2);
  ctx.fillStyle="#1d4ed8"; ctx.fill();
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random()*1e9));
  const [J, setJ] = useState(0.4);
  const [wallC, setWallC] = useState(0.6);
  const [connected, setConnected] = useState(false);
  const [tickHz, setTickHz] = useState(20);

  // internal refs to avoid re-creating intervals
  const tRef = useRef(0);
  const collidedPrevRef = useRef<0|1>(0);
  const agentRef = useRef<Agent>({ pos: {x:0,y:0}, heading: 8 }); // start east

  const [tDisplay, setTDisplay] = useState(0);
  const [collidedDisplay, setCollidedDisplay] = useState<0|1>(0);

  const ws = useMemo(() => new BrainWS("ws://localhost:8000/brain"), []);
  useEffect(() => {
    const off = ws.onMessage(() => {});
    const ping = setInterval(()=> setConnected(ws.connected), 300);
    return () => { off(); clearInterval(ping); };
  }, [ws]);

  const doReset = () => {
    agentRef.current = { pos: {x:0,y:0}, heading: 8 };
    tRef.current = 0;
    collidedPrevRef.current = 0;
    setTDisplay(0);
    setCollidedDisplay(0);
    ws.send({ type: "reset", seed, J, wall_contrast: wallC });
  };

  useEffect(() => { doReset(); }, []); // on mount

  // 60 FPS render
  useEffect(() => {
    let raf = 0;
    const ctx = canvasRef.current?.getContext("2d");
    const render = () => {
      if (!ctx) return;
      drawArena(ctx);
      drawAgent(ctx, agentRef.current);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  // CONTROL LOOP (stable interval, no state in deps)
  useEffect(() => {
    const periodMs = Math.max(5, Math.floor(1000/tickHz));
    let stopped = false;

    const loop = async () => {
      if (stopped) return;

      // build obs
      const agent = agentRef.current;
      const contrast = computeContrastFront(agent.pos, agent.heading, wallC);

      const obs = {
        t: tRef.current + 1,
        j: J,
        contrast_front: Number(contrast.toFixed(6)),
        collided_prev: collidedPrevRef.current,
        heading_index: agent.heading
      } as const;

      const stepMsg: StepMsg = { type: "step", obs };

      let L: 0|1 = 0, R: 0|1 = 0, P: 1|2|3 = 1;
      try {
        const act = await ws.stepAndWait(stepMsg, 200);
        if (act) { L = act.L; R = act.R; P = act.P; }
      } catch {
        // ignore; fallback L=0,R=0,P=1 already set
      }

      const { agent: newAgent, collided } = stepAgent(agent, L, R, P);
      agentRef.current = newAgent;
      collidedPrevRef.current = collided;

      // update visible counters (cheap setStates)
      tRef.current += 1;
      if ((tRef.current & 7) === 0) { // throttle UI updates
        setTDisplay(tRef.current);
        setCollidedDisplay(collidedPrevRef.current);
      }
    };

    const id = setInterval(loop, periodMs);
    return () => { stopped = true; clearInterval(id); };
  }, [tickHz, ws, J, wallC]);

  return (
    <div style={{display:"grid", gridTemplateColumns:"700px 1fr", gap: "16px", padding: 16}}>
      <div>
        <canvas ref={canvasRef} width={WIDTH} height={HEIGHT}
          style={{border:"1px solid #0ea5e9", background:"#0b1220", borderRadius: 8}}/>
      </div>

      <div style={{color:"#e5e7eb"}}>
        <h2 style={{marginTop:0}}>Circle-Arena Jellyfish — Phase 0</h2>
        <div style={{marginBottom:12}}>
          <button onClick={doReset} style={{padding:"8px 12px", borderRadius:8, border:"1px solid #334155", background:"#111827", color:"#e5e7eb"}}>
            Reset
          </button>
        </div>

        <div style={{margin:"8px 0"}}>
          <label>Seed:&nbsp;</label>
          <input type="number" value={seed} onChange={e=>setSeed(parseInt(e.target.value||"0"))} style={{width:180}}/>
        </div>

        <div style={{margin:"8px 0"}}>
          <label>J:&nbsp;{J.toFixed(2)}</label><br/>
          <input type="range" min={0} max={1} step={0.01} value={J} onChange={e=>setJ(parseFloat(e.target.value))} style={{width:260}}/>
        </div>

        <div style={{margin:"8px 0"}}>
          <label>wall_contrast:&nbsp;{wallC.toFixed(2)}</label><br/>
          <input type="range" min={0} max={1} step={0.01} value={wallC} onChange={e=>setWallC(parseFloat(e.target.value))} style={{width:260}}/>
        </div>

        <div style={{margin:"8px 0"}}>
          <label>Tick rate (Hz):&nbsp;{tickHz}</label><br/>
          <input type="range" min={5} max={60} step={1} value={tickHz} onChange={e=>setTickHz(parseInt(e.target.value))} style={{width:260}}/>
        </div>

        <hr style={{borderColor:"#1f2937"}}/>

        <p>
          <b>WS:</b> {connected ? "connected ✅" : "reconnecting…"}<br/>
          <b>t:</b> {tDisplay} &nbsp;|&nbsp; <b>collided_prev:</b> {collidedDisplay}
        </p>

        <p style={{opacity:0.8, fontSize:14}}>
          First turns then moves, demo 0 random movement
        </p>
      </div>
    </div>
  );
}
