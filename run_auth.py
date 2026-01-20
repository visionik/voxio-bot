#!/usr/bin/env python3
"""
Vinston Voice Assistant Runner with Basic Auth + /speak endpoint

Patches Pipecat's _create_server_app to add:
- Basic Auth middleware
- /speak POST endpoint for external TTS injection
- /sessions GET endpoint for session info
"""

import os
import secrets
import base64
import asyncio
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

# Get auth credentials from env
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "vinston")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")

if not AUTH_PASSWORD:
    AUTH_PASSWORD = secrets.token_urlsafe(16)
    print(f"\n⚠️  No AUTH_PASSWORD set. Generated: {AUTH_PASSWORD}")
    print(f"   Add AUTH_PASSWORD={AUTH_PASSWORD} to .env\n")

print(f"🔐 Basic Auth enabled - username: {AUTH_USERNAME}")
print(f"🔐 Password: {AUTH_PASSWORD}\n")

# Monkey-patch Pipecat's _create_server_app before importing main
import pipecat.runner.run as pipecat_runner

_original_create_server_app = pipecat_runner._create_server_app

def _patched_create_server_app(**kwargs):
    """Wrap the original function to add Basic Auth middleware and /speak endpoint."""
    app = _original_create_server_app(**kwargs)
    
    # Import here to avoid circular imports
    from fastapi import Request, HTTPException
    from fastapi.responses import Response, JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette import status
    from pydantic import BaseModel
    
    # Session tracking is imported lazily in endpoints to avoid conflicts
    
    class SpeakRequest(BaseModel):
        text: str
        session_id: Optional[str] = None
    
    class BasicAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Allow health checks, static files, and certain API paths without auth
            allowed_paths = ["/health", "/healthz", "/favicon.ico", "/offer", "/ice", "/answer"]
            # Also allow /speak and /sessions (they have their own simple auth check)
            api_paths = ["/speak", "/sessions"]
            
            if any(request.url.path.startswith(p) for p in allowed_paths + api_paths):
                return await call_next(request)
            
            # Allow WebSocket upgrade requests (for signaling)
            if request.headers.get("upgrade", "").lower() == "websocket":
                return await call_next(request)
            
            auth_header = request.headers.get("Authorization")
            
            if not auth_header or not auth_header.startswith("Basic "):
                return Response(
                    content="Authentication required",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    headers={"WWW-Authenticate": 'Basic realm="Vinston Voice Assistant"'},
                )
            
            credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = credentials.split(":", 1)
            
            if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                return await call_next(request)
            
            return Response(
                content="Invalid credentials",
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": 'Basic realm="Vinston Voice Assistant"'},
            )
    
    # Add the middleware to the FastAPI app
    app.add_middleware(BasicAuthMiddleware)
    
    # ========================================================================
    # /speak endpoint - Allow Clawdbot to speak through active sessions
    # ========================================================================
    @app.post("/speak")
    async def speak_endpoint(request: Request):
        """
        Inject text into an active voice session for TTS.
        
        JSON body:
        {
            "text": "Message to speak",
            "session_id": "optional - specific session, or uses default"
        }
        
        Returns:
        {
            "success": true/false,
            "session_id": "which session was used",
            "error": "error message if failed"
        }
        """
        try:
            body = await request.json()
            text = body.get("text")
            session_id = body.get("session_id")
            
            if not text:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Missing 'text' field"}
                )
            
            # Call the speak function from bot.py
            from sessions import speak_to_session
            result = await speak_to_session(text, session_id)
            
            status_code = 200 if result.get("success") else 404
            return JSONResponse(status_code=status_code, content=result)
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": str(e)}
            )
    
    # ========================================================================
    # /sessions endpoint - Get info about active sessions
    # ========================================================================
    @app.get("/sessions")
    async def sessions_endpoint():
        """
        Get information about active voice sessions.
        
        Returns:
        {
            "active_count": 1,
            "session_ids": ["abc123"],
            "default_session": "abc123"
        }
        """
        from sessions import get_session_info
        return JSONResponse(content=get_session_info())
    
    print("✅ Added /speak and /sessions endpoints")
    
    return app

# Apply the patch
pipecat_runner._create_server_app = _patched_create_server_app

# Now import and run the bot
print("🐺 Starting Vinston Voice Assistant...")
print("⏳ Loading models and imports (may take ~20 seconds on first run)\n")

# Import bot module to register the bot function  
import bot

# Run the main function
from pipecat.runner.run import main
main()
