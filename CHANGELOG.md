# Changelog

All notable changes to voxio-bot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-27

### Added
- Pre-cached filler response system ("got it", "interesting", etc.)
- Unified TTS cache with ElevenLabs integration
- Warping-compliant test infrastructure
- Comprehensive TURN server configuration guide
- Ambient and character sound files (15 sounds)
- Gateway mode for full Clawdbot tool access
- Local vision analysis with Moondream
- GIF display support
- Video capture and display
- TOML configuration file support
- Keep-alive mechanism for WebRTC connections

### Changed
- Ambient sounds now avoid last 5 played files (increased variety)
- Filler responses won't repeat same phrase back-to-back
- Handoff flow: speak acknowledgment first, then play ambients
- All ambient files converted to 24kHz (matches TTS output)

### Fixed
- WAV parsing uses proper readframes() instead of assuming 44-byte header
- Audio pops eliminated with sample rate alignment and fade in/out
- Callback URL now uses localhost (Cloudflare Access was blocking)

## [0.1.0] - 2026-01-17

### Added
- Initial release
- Real-time voice + video AI assistant
- WebRTC and Daily transport support
- ElevenLabs TTS integration
- Anthropic Claude LLM integration
- Image generation via nano-banana (Gemini)
- Sound effects playback
- Clawdbot handoff for complex tasks

[Unreleased]: https://github.com/visionik/voxio-bot/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/visionik/voxio-bot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/visionik/voxio-bot/releases/tag/v0.1.0
