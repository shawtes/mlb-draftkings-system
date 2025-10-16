# Backend Port Configuration Fix ✅

## Problem
Backend was trying to run WebSocket on port **8080**, which was already in use:
```
Error: listen EADDRINUSE: address already in use :::8080
```

## Solution
Changed WebSocket server to **share port 5000** with HTTP server instead of using a separate port.

## Changes Made

### Before:
```javascript
// WebSocket on separate port (CAUSED CONFLICT)
const wss = new WebSocket.Server({ port: 8080 });

const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### After:
```javascript
// HTTP server starts first
const server = app.listen(PORT, () => {
  console.log(`🚀 DFS Optimizer Server running on port ${PORT}`);
  console.log(`📊 WebSocket server running on port ${PORT}/ws`);
});

// WebSocket shares the same port
const wss = new WebSocket.Server({ server });
```

## Result

✅ **Single Port**: Both HTTP and WebSocket now use port **5000**  
✅ **No Conflicts**: No more EADDRINUSE errors  
✅ **Simpler Configuration**: One port for everything  
✅ **Frontend Compatible**: Vite proxy routes `/ws` correctly  

## How It Works Now

```
Client → http://localhost:3000 (Frontend - Vite)
           ↓ (Proxied to backend)
Server → http://localhost:5000 (Backend HTTP + WebSocket)
```

### Endpoints:
- **HTTP API**: `http://localhost:5000/api/*`
- **WebSocket**: `ws://localhost:5000` (same port!)
- **Frontend**: `http://localhost:3000` (Vite dev server)

## Testing

Try starting the server again:
```bash
cd server
node index.js
```

You should see:
```
🚀 DFS Optimizer Server running on port 5000
📊 WebSocket server running on port 5000/ws
🌐 Access the app at http://localhost:5000
```

No more port conflicts! 🎉

---

**Fixed**: October 15, 2025  
**Issue**: Port 8080 conflict  
**Solution**: WebSocket shares port 5000 with HTTP server

