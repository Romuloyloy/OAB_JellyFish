import type { Incoming, Outgoing, ActionPayload } from "./types";

type Listener = (msg: Incoming) => void;

export class BrainWS {
  url: string;
  ws: WebSocket | null = null;
  listeners: Set<Listener> = new Set();
  reconnectDelay = 1000;
  connected = false;

  constructor(url = "ws://localhost:8000/brain") {
    this.url = url;
    this.connect();
  }

  onMessage(fn: Listener) { this.listeners.add(fn); return () => this.listeners.delete(fn); }

  private connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => { this.connected = true; };
    this.ws.onclose = () => {
      this.connected = false;
      setTimeout(() => this.connect(), this.reconnectDelay);
    };
    this.ws.onmessage = (ev) => {
      try {
        const msg: Incoming = JSON.parse(ev.data);
        this.listeners.forEach(l => l(msg));
      } catch { /* ignore malformed */ }
    };
  }

  send(msg: Outgoing) {
    if (this.ws && this.connected) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  async stepAndWait(msg: Outgoing, timeoutMs = 200): Promise<ActionPayload | null> {
    return new Promise<ActionPayload | null>((resolve) => {
      let done = false;
      const off = this.onMessage((incoming) => {
        if (done) return;
        if (incoming.type === "action") {
          done = true; off(); resolve(incoming.act);
        }
      });
      this.send(msg);
      setTimeout(() => { if (!done) { done = true; off(); resolve(null); } }, timeoutMs);
    });
  }
}
