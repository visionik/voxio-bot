"""
Session tracking for voice assistant.
Separated from bot.py to avoid import conflicts.
"""

from typing import Optional
from loguru import logger

# Active sessions dictionary
active_sessions: dict = {}
default_session_id: Optional[str] = None


def register_session(session_id: str, task, tts, context):
    """Register a new session."""
    global default_session_id
    active_sessions[session_id] = {
        "task": task,
        "tts": tts,
        "context": context,
    }
    default_session_id = session_id
    logger.info(f"📝 Session registered: {session_id} (total: {len(active_sessions)})")


def unregister_session(session_id: str):
    """Unregister a session."""
    global default_session_id
    if session_id in active_sessions:
        del active_sessions[session_id]
        logger.info(f"🗑️ Session unregistered: {session_id} (total: {len(active_sessions)})")
        if default_session_id == session_id:
            default_session_id = next(iter(active_sessions), None)


def update_tts_for_all_sessions(new_tts):
    """Update TTS service for all active sessions (called on SIGHUP)."""
    for session_id, session in active_sessions.items():
        session["tts"] = new_tts
        logger.info(f"🔄 Updated TTS for session: {session_id}")
    logger.info(f"✅ TTS updated for {len(active_sessions)} active sessions")


async def speak_to_session(text: str, session_id: Optional[str] = None) -> dict:
    """
    Inject text into an active session for TTS and add to conversation context.
    
    This will interrupt any ambient sounds that are playing (but not normal TTS).
    """
    from pipecat.frames.frames import TTSSpeakFrame
    
    target_id = session_id or default_session_id
    
    if not target_id:
        return {"success": False, "error": "No active sessions"}
    
    if target_id not in active_sessions:
        return {"success": False, "error": f"Session {target_id} not found"}
    
    session = active_sessions[target_id]
    tts = session.get("tts")
    task = session.get("task")
    context = session.get("context")
    
    if not tts or not task:
        return {"success": False, "error": "Session missing TTS or task"}
    
    try:
        # Add the text to conversation context so Claude remembers what was said
        if context:
            context.add_message({"role": "assistant", "content": f"[Results from main system]: {text}"})
            logger.info(f"📝 Added to context: {text[:50]}...")
        
        # Queue a TTSSpeakFrame to be spoken
        frame = TTSSpeakFrame(text=text)
        await task.queue_frame(frame)
        logger.info(f"🔊 Spoke to session {target_id}: {text[:50]}...")
        return {"success": True, "session_id": target_id}
    except Exception as e:
        logger.error(f"Failed to speak to session: {e}")
        return {"success": False, "error": str(e)}


def get_session_info() -> dict:
    """Get info about active sessions."""
    return {
        "active_count": len(active_sessions),
        "session_ids": list(active_sessions.keys()),
        "default_session": default_session_id,
    }
