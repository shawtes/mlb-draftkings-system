export type WsStatus = 'connected' | 'reconnecting' | 'disconnected';

class WebSocketConnection {
  private ws: WebSocket | null = null;
  private url: string;
  private listeners: Map<string, Function[]> = new Map();
  private _status: WsStatus = 'disconnected';
  private onStatusChange?: (status: WsStatus) => void;
  private shouldReconnect = true;

  constructor(url: string, onStatusChange?: (status: WsStatus) => void) {
    this.url = url;
    this.onStatusChange = onStatusChange;
    this.connect();
  }

  get status(): WsStatus {
    return this._status;
  }

  private setStatus(status: WsStatus) {
    this._status = status;
    this.onStatusChange?.(status);
  }

  private connect() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.setStatus('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit(data.type, data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        if (this.shouldReconnect) {
          this.setStatus('reconnecting');
          setTimeout(() => {
            if (this.shouldReconnect) this.connect();
          }, 3000);
        } else {
          this.setStatus('disconnected');
        }
      };

      this.ws.onerror = () => {
        // onclose will fire after onerror, status handled there
      };
    } catch {
      this.setStatus('disconnected');
    }
  }

  public disconnect() {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close();
    }
  }

  public on(eventType: string, callback: Function) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType)?.push(callback);
  }

  public off(eventType: string, callback: Function) {
    const callbacks = this.listeners.get(eventType);
    if (callbacks) {
      this.listeners.set(eventType, callbacks.filter(cb => cb !== callback));
    }
  }

  private emit(eventType: string, data: any) {
    const callbacks = this.listeners.get(eventType);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in WebSocket callback:', error);
        }
      });
    }
  }

  public send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}

export default WebSocketConnection;
