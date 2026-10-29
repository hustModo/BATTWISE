import type { WSEvent } from "@/types";

export type WSHandler = (ev: WSEvent) => void;

export class SessionSocket {
  private url: string;
  private ws: WebSocket | null = null;
  private handlers = new Set<WSHandler>();
  private closed = false;
  private reconnectTimer: number | null = null;
  private retries = 0;

  constructor(sessionId: string) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.url = `${proto}://${location.host}/ws/session/${sessionId}`;
  }

  subscribe(fn: WSHandler): () => void {
    this.handlers.add(fn);
    return () => this.handlers.delete(fn);
  }

  open() {
    if (this.ws) return;
    this.closed = false;
    this.ws = new WebSocket(this.url);
    this.ws.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data) as WSEvent;
        this.handlers.forEach((h) => h(ev));
      } catch (err) {
        console.warn("ws parse error", err, e.data);
      }
    };
    this.ws.onclose = () => {
      this.ws = null;
      // Synthesise a `disconnected` event so the store can show a
      // banner; otherwise users see "agent_started" cards frozen in
      // place without knowing the backend died. Reconnect is still
      // attempted below.
      this.handlers.forEach((h) =>
        h({ kind: "disconnected" } as unknown as WSEvent),
      );
      if (!this.closed) this.scheduleReconnect();
    };
    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return;
    const delay = Math.min(15_000, 500 * Math.pow(1.6, this.retries++));
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.open();
    }, delay);
  }

  close() {
    this.closed = true;
    this.ws?.close();
    this.ws = null;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
