#!/usr/bin/env python3
"""
Vinston Voice Assistant Runner with Basic Auth
"""

import os
import sys
import secrets
import base64
from functools import wraps

from dotenv import load_dotenv
load_dotenv(override=True)

# Get auth credentials from env
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "vinston")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")

if not AUTH_PASSWORD:
    # Generate a random password if not set
    AUTH_PASSWORD = secrets.token_urlsafe(16)
    print(f"\n⚠️  No AUTH_PASSWORD set. Generated: {AUTH_PASSWORD}")
    print(f"   Add to .env: AUTH_PASSWORD={AUTH_PASSWORD}\n")

print(f"🔐 Basic Auth enabled - username: {AUTH_USERNAME}")

# Now import and patch pipecat's FastAPI app
from pipecat.runner.run import main as pipecat_main
from pipecat.runner.server import runner  # The FastAPI app

from fastapi import Request, HTTPException, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow health checks without auth
        if request.url.path in ["/health", "/healthz"]:
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Basic "):
            return Response(
                content="Authentication required",
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Basic realm=\"Vinston Voice Assistant\""},
            )
        
        try:
            credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = credentials.split(":", 1)
            
            if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                return await call_next(request)
        except Exception:
            pass
        
        return Response(
            content="Invalid credentials",
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Basic realm=\"Vinston Voice Assistant\""},
        )

# Add the middleware to the FastAPI app
runner.app.add_middleware(BasicAuthMiddleware)

if __name__ == "__main__":
    # Pass through to pipecat's main runner
    pipecat_main()
