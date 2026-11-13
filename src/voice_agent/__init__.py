"""The voice agent: a live, interruptible voice call with a pluggable brain.

Media flows through a LiveKit server. Speech recognition and synthesis are this
project's own ASR and TTS services, reached over HTTP. The brain only ever sees
text, so it can be any OpenAI-compatible chat endpoint or an agent on a NATS bus.
"""
