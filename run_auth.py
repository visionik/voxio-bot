#!/usr/bin/env python3
"""
Vinston Voice Assistant Runner with /speak endpoint

Patches Pipecat's _create_server_app to add:
- /speak POST endpoint for external TTS injection
- /sessions GET endpoint for session info
- /h264 custom client endpoint
"""

import os
import asyncio
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

# Monkey-patch Pipecat's _create_server_app before importing main
import pipecat.runner.run as pipecat_runner

_original_create_server_app = pipecat_runner._create_server_app

def _patched_create_server_app(**kwargs):
    """Wrap the original function to add custom endpoints."""
    app = _original_create_server_app(**kwargs)
    
    # Import here to avoid circular imports
    from fastapi import Request
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel
    from pathlib import Path
    
    class SpeakRequest(BaseModel):
        text: str
        session_id: Optional[str] = None
    
    # ========================================================================
    # /speak endpoint - Allow Clawdbot to speak through active sessions
    # ========================================================================
    @app.post("/speak")
    async def speak_endpoint(request: Request):
        """
        Inject text into an active voice session for TTS.
        
        JSON body:
        {
            "text": "Hello world",
            "session_id": "optional-specific-session"
        }
        """
        from sessions import get_active_session, speak_to_session
        
        try:
            data = await request.json()
            text = data.get("text", "")
            session_id = data.get("session_id")
            
            if not text:
                return JSONResponse(
                    content={"success": False, "error": "No text provided"},
                    status_code=400
                )
            
            # Get active session
            session = get_active_session(session_id)
            if not session:
                return JSONResponse(
                    content={"success": False, "error": "No active sessions"},
                    status_code=404
                )
            
            # Inject the text for TTS
            await speak_to_session(session, text)
            
            return JSONResponse(content={
                "success": True,
                "session_id": session["session_id"]
            })
            
        except Exception as e:
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500
            )
    
    # ========================================================================
    # /sessions endpoint - Get info about active voice sessions
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
    
    # ========================================================================
    # /h264 - Custom client with H264 codec preference (better for screen share)
    # ========================================================================
    @app.get("/h264")
    async def h264_client():
        """Serve custom H264-preferred client for better screen sharing."""
        client_path = Path(__file__).parent / "static" / "index.html"
        if client_path.exists():
            return FileResponse(client_path, media_type="text/html")
        return JSONResponse(content={"error": "Custom client not found"}, status_code=404)
    
    print("✅ Added /speak, /sessions, and /h264 endpoints")
    
    return app

# Apply the patch
pipecat_runner._create_server_app = _patched_create_server_app

# Now import and run the bot
print("🐺 Starting Vinston Voice Assistant...")
print("⏳ Loading models and imports (may take ~20 seconds on first run)\n")

# Import bot module to register the bot function  
import bot
import sys

# Add port from config if not specified
if '--port' not in sys.argv:
    port = bot.get_transport_port()
    sys.argv.extend(['--port', str(port)])
    print(f"📡 Using port from config: {port}")

# Run the main function
from pipecat.runner.run import main
main()
