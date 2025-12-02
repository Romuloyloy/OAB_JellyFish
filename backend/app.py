import csv
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from brain import Brain

app = FastAPI()

# dev-friendly CORS (frontend at localhost:5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

EP_DIR = os.path.join("runs", "episodes")
os.makedirs(EP_DIR, exist_ok=True)

# === path to your trained model (.pt) ===
# adjust this if you run a new evolutionary run
POLICY_PATH = os.path.join("evo", "elly.pt")

HTML = """<html><body>
<h3>Circle-Arena Jellyfish backend</h3>
<p>WS at <code>/brain</code></p>
</body></html>"""

@app.get("/")
async def root():
    return HTMLResponse(HTML)


@app.websocket("/brain")
async def brain_ws(ws: WebSocket):
    await ws.accept()
    brain: Optional[Brain] = None
    seed: Optional[int] = None
    writer = None
    csv_file = None

    try:
        while True:
            msg = await ws.receive_json()
            typ = msg.get("type")

            if typ == "reset":
                # close previous file if any
                if csv_file:
                    csv_file.close()
                    csv_file = None
                    writer = None

                seed = int(msg.get("seed", 0))
                J = float(msg.get("J", 0.0))
                wall_contrast = float(msg.get("wall_contrast", 0.0))

                brain = Brain(seed=seed)
                brain.reset(J, wall_contrast)

                # --- load evolved policy automatically if available ---
                if os.path.exists(POLICY_PATH):
                    try:
                        brain.load_policy(POLICY_PATH)
                        print(f"[Brain] Loaded policy from {POLICY_PATH}")
                    except Exception as e:
                        print(f"[Brain] Failed to load policy ({e}); reverting to random.")
                        brain.set_mode("random")
                else:
                    print(f"[Brain] Policy file not found: {POLICY_PATH} (using random mode)")

                # open new CSV
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(EP_DIR, f"{ts}.csv")
                csv_file = open(path, "w", newline="")
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "t", "J", "wall_contrast",
                        "contrast_front", "collided_prev",
                        "L", "R", "P"
                    ]
                )
                writer.writeheader()

                await ws.send_json({"type": "ok"})

            elif typ == "step":
                if not brain:
                    await ws.send_json({
                        "type": "action",
                        "act": {"L": 0, "R": 0, "P": 1, "debug": {"note": "no reset yet"}}
                    })
                    continue

                obs = msg.get("obs", {})
                act = brain.act(obs)

                # log CSV
                if writer:
                    writer.writerow({
                        "t": obs.get("t"),
                        "J": brain.J,
                        "wall_contrast": brain.wall_contrast,
                        "contrast_front": obs.get("contrast_front"),
                        "collided_prev": obs.get("collided_prev"),
                        "L": act["L"], "R": act["R"], "P": act["P"]
                    })
                    csv_file.flush()

                await ws.send_json({"type": "action", "act": act})

            elif typ == "done":
                if csv_file:
                    csv_file.close()
                    csv_file = None
                    writer = None
                await ws.send_json({"type": "ok"})

            else:
                await ws.send_json({"type": "ok"})

    except WebSocketDisconnect:
        pass
    finally:
        if csv_file:
            csv_file.close()
