#!/usr/bin/env python3
"""
Pre-cache filler phrases for the voice bot.

Usage:
    # Cache for current voice (from config.toml)
    python precache_fillers.py
    
    # Cache for specific voice
    python precache_fillers.py --voice-id QzTKubutNn9TjrB7Xb2Q
    
    # Cache for multiple voices
    python precache_fillers.py --voice-id voice1 --voice-id voice2
    
    # List all filler phrases
    python precache_fillers.py --list
"""

import os
import sys
import asyncio
import argparse
import tomllib
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from filler_cache import (
    precache_fillers,
    precache_fillers_multi_voice,
    FillerConfig,
    ALL_FILLERS,
    FILLERS_ACKNOWLEDGMENT,
    FILLERS_THINKING,
    FILLERS_TRANSITION,
)


def load_config_voice() -> str | None:
    """Load voice ID from config.toml."""
    config_path = Path(__file__).parent / "config.toml"
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
        return config.get("tts", {}).get("voice")
    return None


async def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache filler phrases for voice bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python precache_fillers.py                           # Use voice from config.toml
    python precache_fillers.py --voice-id <id>           # Specific voice
    python precache_fillers.py --list                    # Show all phrases
    python precache_fillers.py --voice-id v1 --voice-id v2  # Multiple voices
        """,
    )
    parser.add_argument(
        "--voice-id", "-v",
        action="append",
        dest="voice_ids",
        help="ElevenLabs voice ID (can specify multiple)",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.getenv("ELEVENLABS_API_KEY"),
        help="ElevenLabs API key (default: ELEVENLABS_API_KEY env var)",
    )
    parser.add_argument(
        "--cache-dir", "-d",
        default="~/.cache/voxio-fillers",
        help="Cache directory (default: ~/.cache/voxio-fillers)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all filler phrases and exit",
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("=" * 50)
        print("FILLER PHRASES")
        print("=" * 50)
        print(f"\n📣 Acknowledgment ({len(FILLERS_ACKNOWLEDGMENT)} phrases):")
        print("   Played immediately after user stops speaking")
        for p in FILLERS_ACKNOWLEDGMENT:
            print(f"   • {p}")
        
        print(f"\n🤔 Thinking ({len(FILLERS_THINKING)} phrases):")
        print("   Played while waiting for LLM (optional)")
        for p in FILLERS_THINKING:
            print(f"   • {p}")
        
        print(f"\n🔄 Transition ({len(FILLERS_TRANSITION)} phrases):")
        print("   Played before main response")
        for p in FILLERS_TRANSITION:
            print(f"   • {p}")
        
        print(f"\n📊 Total: {len(ALL_FILLERS)} phrases")
        return
    
    # Validate API key
    if not args.api_key:
        print("❌ Error: ELEVENLABS_API_KEY not set")
        print("   Set it via: export ELEVENLABS_API_KEY=your_key")
        print("   Or pass:    --api-key your_key")
        sys.exit(1)
    
    # Get voice IDs
    voice_ids = args.voice_ids or []
    
    # Try to load from config if no voice specified
    if not voice_ids:
        config_voice = load_config_voice()
        if config_voice:
            voice_ids = [config_voice]
            print(f"📋 Using voice from config.toml: {config_voice}")
        else:
            print("❌ Error: No voice ID specified")
            print("   Use --voice-id <id> or set tts.voice in config.toml")
            sys.exit(1)
    
    # Create config
    config = FillerConfig(cache_dir=args.cache_dir)
    
    print(f"🗄️  Cache directory: {Path(config.cache_dir).expanduser()}")
    print(f"🎙️  Voices to cache: {len(voice_ids)}")
    print(f"💬 Phrases per voice: {len(ALL_FILLERS)}")
    print()
    
    # Pre-cache for all voices
    if len(voice_ids) == 1:
        await precache_fillers(args.api_key, voice_ids[0], config)
    else:
        await precache_fillers_multi_voice(args.api_key, voice_ids, config)
    
    print("\n✅ Pre-caching complete!")
    print(f"   Run the bot with FillerInjector to use cached responses")


if __name__ == "__main__":
    asyncio.run(main())
