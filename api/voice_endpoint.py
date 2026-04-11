"""
Voice WebSocket Endpoint
-------------------------
WebSocket-based voice I/O for the agent platform.

Protocol:
  Client → Server: binary audio chunks (PCM 16kHz 16-bit mono or WebM/Opus)
  Server → Client: JSON events interleaved with binary TTS audio

  JSON Messages (server → client):
    {"type": "stt_result", "text": "user's transcribed speech"}
    {"type": "agent_token", "content": "streaming text token"}
    {"type": "agent_done", "full_response": "complete response text"}
    {"type": "tts_start"}
    {"type": "tts_end"}
    {"type": "error", "message": "..."}

  JSON Messages (client → server):
    {"type": "config", "session_id": "...", "agent_id": "...", "model": "..."}
    {"type": "stt_end"}  — signals end of speech input

Usage:
  ws://localhost:8000/ws/voice
  First message must be a config JSON.
  Then stream audio chunks, ending with {"type": "stt_end"}.
  Server responds with agent text + TTS audio.
"""

import json
import asyncio
import os
import tempfile
import re
import uuid
from typing import Optional, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from config.logger import log
from run_engine import RunEngineRequest


def setup_voice_routes(app: FastAPI, *, run_engine, profile_manager) -> None:
    """Register voice WebSocket endpoint."""

    @app.websocket("/ws/voice")
    async def voice_endpoint(ws: WebSocket):
        await ws.accept()
        log("system", "system", "Voice WebSocket connected")

        session_id: Optional[str] = None
        agent_id: str = "default"
        model: Optional[str] = None
        owner_id: Optional[str] = None

        try:
            # Phase 1: Receive config
            config_raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            config = json.loads(config_raw)
            session_id = config.get("session_id")
            agent_id = config.get("agent_id", "default")
            model = config.get("model")
            owner_id = str(config.get("owner_id") or "").strip() or None
            voice_model = config.get("voice_model", "aura-orion-en")
            sensitivity = config.get("sensitivity", 0.5)

            await ws.send_json({"type": "config_ack", "status": "ok"})
            log(
                "system",
                session_id or "voice",
                f"Voice config: agent={agent_id} model={model} v_model={voice_model} sens={sensitivity}",
                owner_id=owner_id,
            )

            # Phase 2: Dual-Mode Loop (Streaming STT & TTS)
            while True:
                user_text = ""
                
                # If Deepgram API key is present, we use Streaming STT
                dg_key = os.getenv("DEEPGRAM_API_KEY")
                if dg_key:
                    try:
                        from deepgram import DeepgramClient
                        from deepgram.core.events import EventType
                        
                        deepgram = DeepgramClient(dg_key)
                        dg_connection = deepgram.listen.live.v("1")
                        
                        transcript_queue = asyncio.Queue()

                        async def on_message(self, message, **kwargs):
                            # In v5, message is usually a ListenV1ResultsEvent
                            try:
                                if hasattr(message, 'channel') and message.channel.alternatives:
                                    sentence = message.channel.alternatives[0].transcript
                                    if sentence:
                                        # Send interim results to client
                                        await ws.send_json({
                                            "type": "stt_result", 
                                            "text": sentence, 
                                            "is_final": message.is_final
                                        })
                                        if message.is_final:
                                            await transcript_queue.put(sentence)
                            except Exception as e:
                                log("system", "voice", f"Error in on_message: {e}", level="error")

                        dg_connection.on(EventType.MESSAGE, on_message)
                        
                        # In v5, options are passed as keyword arguments to connect
                        # Map sensitivity (0–1) to Deepgram endpointing (ms)
                        # Lower sensitivity = longer endpointing (more patient)
                        # Higher sensitivity = shorter endpointing (more aggressive)
                        endpointing_ms = int(1000 - (sensitivity * 900)) # 100ms to 1000ms range
                        
                        async with dg_connection.connect(
                            model="nova-3",
                            language="en",
                            smart_format=True,
                            interim_results=True,
                            endpointing=endpointing_ms, 
                        ) as live:
                            # Bridge loop
                            while True:
                                # Wait for audio or stt_end signal
                                try:
                                    msg = await asyncio.wait_for(ws.receive(), timeout=0.05)
                                    if msg["type"] == "websocket.disconnect":
                                        raise WebSocketDisconnect()
                                    
                                    if "bytes" in msg and msg["bytes"]:
                                        await live.send_media(msg["bytes"])
                                    elif "text" in msg and msg["text"]:
                                        cmd = json.loads(msg["text"])
                                        if cmd.get("type") == "stt_end":
                                            break
                                except asyncio.TimeoutError:
                                    # Check if we got a final transcript from DG endpointing
                                    if not transcript_queue.empty():
                                        user_text = await transcript_queue.get()
                                        break
                                    continue
                            
                            if hasattr(live, 'finalize'):
                                await live.finalize()
                    except Exception as e:
                        log("system", "voice", f"Deepgram Live failed: {e}", level="warn")
                
                # Legacy Buffer Fallback (if Deepgram Live failed or no key)
                if not user_text:
                    audio_chunks = []
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect": raise WebSocketDisconnect()
                        if "bytes" in msg: audio_chunks.append(msg["bytes"])
                        elif "text" in msg:
                            if json.loads(msg["text"]).get("type") == "stt_end": break
                    
                    if not audio_chunks: continue
                    user_text = await _transcribe_audio(b"".join(audio_chunks))

                if not user_text or not user_text.strip():
                    await ws.send_json({"type": "stt_result", "text": ""})
                    continue

                # Agent & TTS Stream
                full_response = ""
                sentence_buffer = ""
                
                # Injection: Add "Voice Mode" instruction to the prompt context if in shovs Mode
                voice_optimized_prompt = (
                    "You are SHOVS. Respond in a concise, conversational tone. "
                    "BE EXTREMELY BRIEF (1-2 sentences). DO NOT use markdown, bolding, lists, or headers. "
                    "Speak like a helpful human assistant. No hashtags, no stars."
                )
                
                # FIX: Ensure we are NOT using an STT model for the LLM call
                llm_model = model
                if llm_model and "whisper" in llm_model.lower():
                    # Fallback to a proper chat model if Whisper was accidentally passed
                    if "groq" in llm_model.lower():
                        llm_model = "groq:llama-3.3-70b-versatile"
                    else:
                        llm_model = "gemini-1.5-flash"
                
                await ws.send_json({"type": "tts_start"})

                profile = profile_manager.get(agent_id) or profile_manager.get("default")
                effective_session_id = session_id or f"voice_{uuid.uuid4().hex[:8]}"
                if not session_id:
                    session_id = effective_session_id

                request = RunEngineRequest(
                    session_id=effective_session_id,
                    owner_id="",
                    agent_id=agent_id,
                    user_message=f"{voice_optimized_prompt}\n\nUser: {user_text}",
                    model=llm_model or (getattr(profile, "model", None) or "llama3.2"),
                    system_prompt=getattr(profile, "system_prompt", "") or "",
                    allowed_tools=tuple(getattr(profile, "tools", []) or []),
                    use_planner=bool(getattr(profile, "default_use_planner", True)),
                    agent_revision=getattr(profile, "revision", None),
                )

                async for event in run_engine.stream(request):
                    if event["type"] == "token":
                        token = event["content"]
                        full_response += token
                        sentence_buffer += token
                        await ws.send_json({"type": "agent_token", "content": token})
                        
                        if any(p in token for p in ".!?\n") and len(sentence_buffer.strip()) > 10:
                            clean_sentence = _clean_text_for_speech(sentence_buffer)
                            if clean_sentence.strip():
                                async for audio_chunk in _synthesize_speech(clean_sentence, voice_model=voice_model):
                                    await ws.send_bytes(audio_chunk)
                            sentence_buffer = ""

                if sentence_buffer.strip():
                    clean_sentence = _clean_text_for_speech(sentence_buffer)
                    if clean_sentence.strip():
                        async for audio_chunk in _synthesize_speech(clean_sentence, voice_model=voice_model):
                            await ws.send_bytes(audio_chunk)

                await ws.send_json({"type": "agent_done", "full_response": full_response})
                await ws.send_json({"type": "tts_end"})

        except WebSocketDisconnect:
            log("system", session_id or "voice", "Voice WebSocket disconnected")
        except asyncio.TimeoutError:
            await ws.send_json({"type": "error", "message": "Config timeout — send config JSON first"})
            await ws.close()
        except Exception as e:
            # Don't log "Cannot call receive" as a scary error if we're already shutting down
            if "receive" in str(e) and "disconnect" in str(e):
                log("system", session_id or "voice", "Voice WebSocket cleaned up")
            else:
                log("system", session_id or "voice", f"Voice error: {e}", level="error")
                try:
                    await ws.send_json({"type": "error", "message": str(e)})
                except:
                    pass


def _clean_text_for_speech(text: str) -> str:
    """
    Strips markdown and special characters that TTS shouldn't narrate.
    """
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove bold/italics
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove hashtags and other visual markers
    text = text.replace('#', '')
    
    return text.strip()


# ── STT Engine ────────────────────────────────────────────────────────────────

_whisper_model = None

async def _transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text using faster-whisper.

    Supports: PCM 16kHz 16-bit mono, WAV, WebM, MP3, FLAC, OGG
    Install: pip install faster-whisper

    Falls back to a stub message if faster-whisper is not installed,
    so the rest of the system can still be tested.
    """
    # 0. Try Deepgram Cloud STT first (ultra-fast, best for real-time)
    dg_key = os.getenv("DEEPGRAM_API_KEY")
    if dg_key:
        try:
            return await _transcribe_audio_deepgram(audio_bytes, dg_key)
        except Exception as e:
            log("system", "voice", f"Deepgram STT failed, falling back: {e}", level="warn")

    # 1. Try Groq Cloud STT (very fast)
    gr_key = os.getenv("GROQ_API_KEY")
    if gr_key:
        try:
            return await _transcribe_audio_groq(audio_bytes, gr_key)
        except Exception as e:
            log("system", "voice", f"Groq STT failed, falling back to local: {e}", level="warn")

    # 2. Fallback to local faster-whisper
    global _whisper_model

    try:
        from faster_whisper import WhisperModel
        import numpy as np
        import io

        # Lazy-load model (downloads on first use, ~1GB for 'base')
        if _whisper_model is None:
            model_size = os.getenv("WHISPER_MODEL", "base")
            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute_type = "int8" if device == "cpu" else "float16"
            _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            log("system", "system", f"Whisper model loaded: {model_size} on {device}")

        # Write to temp file (faster-whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            segments, info = _whisper_model.transcribe(
                temp_path,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return text.strip()
        finally:
            os.unlink(temp_path)

    except ImportError:
        # Stub for development: return a message indicating STT isn't installed
        log("system", "system", "faster-whisper not installed — using stub STT", level="warn")
        return "[STT unavailable — install faster-whisper or provide GROQ_API_KEY]"


async def _transcribe_audio_groq(audio_bytes: bytes, api_key: str) -> str:
    """Transcribe using Groq Cloud API."""
    from groq import AsyncGroq

    client = AsyncGroq(api_key=api_key)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        with open(temp_path, "rb") as audio_file:
            translation = await client.audio.transcriptions.create(
                file=(os.path.basename(temp_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en",
            )
            return translation.strip()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def _transcribe_audio_deepgram(audio_bytes: bytes, api_key: str) -> str:
    """Transcribe using Deepgram Cloud API."""
    from deepgram import DeepgramClient

    try:
        deepgram = DeepgramClient(api_key)
        
        # In v5, we can use listen.rest.v("1").transcribe_file
        # payload usually expects a dict with buffer
        payload = {
            "buffer": audio_bytes,
        }
        options = {
            "model": "nova-3",
            "smart_format": True,
            "language": "en",
        }
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        
        # Extract transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript.strip()
    except Exception as e:
        log("system", "voice", f"Deepgram transcription error: {e}", level="error")
        raise e


# ── TTS Engine ────────────────────────────────────────────────────────────────

async def _synthesize_speech(text: str, voice_model: str = "aura-orion-en"):
    """
    Convert text to speech audio chunks.
    Prioritizes Deepgram Aura for high-fidelity shovs voice.
    """
    # 0. Handle Deepgram Models
    if voice_model.startswith("aura-"):
        dg_key = os.getenv("DEEPGRAM_API_KEY")
        if dg_key:
            try:
                from deepgram import DeepgramClient
                deepgram = DeepgramClient(dg_key)
                
                async for chunk in deepgram.speak.v("1").audio.generate(
                    text=text,
                    model=voice_model,
                    encoding="linear16",
                    sample_rate=16000
                ):
                    if chunk:
                        yield chunk
                return
            except Exception as e:
                log("system", "voice", f"Deepgram TTS failed: {e}", level="warn")

    # 1. Handle Edge TTS (Free Cloud)
    if voice_model.startswith("edge:"):
        voice_name = voice_model.split(":", 1)[1]
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice_name)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
            return
        except Exception as e:
            log("system", "voice", f"Edge TTS failed: {e}", level="warn")

    # 2. Handle HF / Local (Stub for Hugging Face integration)
    if voice_model == "hf-parler":
        # In a real scenario, this would call a local transformers model
        # For now, fallback with log
        log("system", "voice", "HF Parler-TTS selected — using edge-tts placeholder")
        voice_model = "edge:en-US-GuyNeural"
        # continue fallback below...

    # Fallback to local/free engines
    # Try Kokoro first (fastest local)
    try:
        import kokoro
        import numpy as np
        # af_bella is feminine, maybe find a masc one for Jarvis if available
        # af_sky or similar. For now af_bella is the standard example.
        voice = kokoro.Voice("af_bella")
        audio = voice.speak(text)
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]
        return
    except ImportError:
        pass

    # Try edge-tts (free, Microsoft voices)
    try:
        import edge_tts
        log("system", "voice", f"Using edge-tts fallback", level="debug")
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural") # Guy is close to Jarvis
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
        return
    except ImportError:
        pass

    raise RuntimeError("No TTS engine available.")
