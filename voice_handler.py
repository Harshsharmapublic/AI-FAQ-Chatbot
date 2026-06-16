"""
voice_handler.py - Voice Interaction Module
============================================
Author: AI-Powered FAQ Chatbot System
Description:
    Provides Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.

    Audio Backend Priority:
        1. sounddevice  (Python 3.14 compatible, no PyAudio needed)
        2. PyAudio      (legacy fallback, may not be available on Python 3.14+)

    Uses SpeechRecognition with sounddevice for STT.
    Uses pyttsx3 for offline TTS synthesis.
    All operations are wrapped in try-except so the app never crashes if
    audio hardware or libraries are unavailable.

Usage:
    from voice_handler import VoiceHandler
    handler = VoiceHandler()
    text = handler.listen()           # Capture microphone input
    handler.speak("Hello, world!")    # Speak text aloud
"""

import io
import logging
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice Handler
# ---------------------------------------------------------------------------

class VoiceHandler:
    """
    Handles Speech-to-Text and Text-to-Speech interactions.

    STT Backend: sounddevice → numpy → SpeechRecognition (AudioData)
    TTS Backend: pyttsx3 (offline, no internet needed)

    All methods degrade gracefully if hardware/libraries are unavailable.

    Attributes:
        stt_available (bool): True if STT stack is fully functional.
        tts_available (bool): True if pyttsx3 is installed & audio output works.
    """

    # Audio recording settings
    SAMPLE_RATE = 16000     # Hz — optimal for Google Speech API
    CHANNELS = 1            # Mono
    DTYPE = "int16"         # 16-bit PCM

    def __init__(self, tts_rate: int = 170, tts_volume: float = 1.0) -> None:
        """
        Initialize voice handler.

        Args:
            tts_rate:    Words-per-minute for TTS engine (default: 170).
            tts_volume:  Volume level 0.0–1.0 for TTS (default: 1.0).
        """
        self.tts_rate = tts_rate
        self.tts_volume = tts_volume

        self.stt_available = False
        self.tts_available = False
        self._use_sounddevice = False
        self._use_pyaudio = False

        self._sr = None           # speech_recognition module reference
        self._recognizer = None
        self._sd = None           # sounddevice module reference
        self._np = None           # numpy module reference
        self._tts_engine = None
        self._tts_lock = threading.Lock()

        self._init_stt()
        self._init_tts()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_stt(self) -> None:
        """Try to initialize Speech Recognition with sounddevice or PyAudio."""
        # Step 1: Load SpeechRecognition module
        try:
            import speech_recognition as sr
            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._recognizer.pause_threshold = 1.0
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
        except ImportError:
            logger.warning("SpeechRecognition not installed — STT disabled.")
            return

        # Step 2: Try sounddevice (preferred — Python 3.14 compatible)
        try:
            import sounddevice as sd
            import numpy as np
            self._sd = sd
            self._np = np
            self._use_sounddevice = True
            self.stt_available = True
            logger.info("STT initialized with sounddevice backend.")
            return
        except ImportError:
            logger.info("sounddevice not available, trying PyAudio...")
        except Exception as exc:
            logger.warning("sounddevice init failed: %s — trying PyAudio.", exc)

        # Step 3: Fallback to PyAudio
        try:
            import pyaudio  # noqa: F401
            self._use_pyaudio = True
            self.stt_available = True
            logger.info("STT initialized with PyAudio backend.")
        except ImportError:
            logger.warning("PyAudio not installed — STT disabled. Install sounddevice for voice input.")
        except Exception as exc:
            logger.warning("PyAudio init failed: %s", exc)

    def _init_tts(self) -> None:
        """Try to initialize pyttsx3 TTS engine."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.tts_rate)
            engine.setProperty("volume", self.tts_volume)
            # Prefer a female voice if available (Windows: Microsoft Zira)
            voices = engine.getProperty("voices")
            if voices:
                for voice in voices:
                    name_lower = voice.name.lower()
                    if "female" in name_lower or "zira" in name_lower or "hazel" in name_lower:
                        engine.setProperty("voice", voice.id)
                        break
            self._tts_engine = engine
            self.tts_available = True
            logger.info("pyttsx3 TTS initialized.")
        except ImportError:
            logger.warning("pyttsx3 not installed — TTS disabled.")
        except Exception as exc:
            logger.warning("TTS initialization failed: %s", exc)

    # ------------------------------------------------------------------
    # Speech-to-Text
    # ------------------------------------------------------------------

    def listen(self, duration: int = 5, timeout: int = 5, phrase_limit: int = 12) -> Tuple[bool, str]:
        """
        Capture audio from the default microphone and transcribe it.

        Args:
            duration:      Max seconds to record (sounddevice backend).
            timeout:       Seconds to wait for speech (PyAudio backend).
            phrase_limit:  Max seconds for a phrase (PyAudio backend).

        Returns:
            (success: bool, text_or_error: str)
        """
        if not self.stt_available:
            return False, "Voice input is not available. Please type your question."

        if self._use_sounddevice:
            return self._listen_sounddevice(duration)
        elif self._use_pyaudio:
            return self._listen_pyaudio(timeout, phrase_limit)
        else:
            return False, "No audio backend available."

    def _listen_sounddevice(self, duration: int = 5) -> Tuple[bool, str]:
        """Record using sounddevice and transcribe via SpeechRecognition."""
        try:
            logger.info("Recording %ds of audio via sounddevice...", duration)
            recording = self._sd.rec(
                int(duration * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                blocking=True,
            )
            self._sd.wait()

            # Convert numpy int16 array → bytes → SpeechRecognition AudioData
            audio_bytes = recording.tobytes()
            audio_data = self._sr.AudioData(audio_bytes, self.SAMPLE_RATE, 2)  # 2 bytes per sample (int16)

            logger.info("Transcribing audio via Google Web Speech API...")
            text = self._recognizer.recognize_google(audio_data, language="en-IN")
            logger.info("Recognized: %s", text)
            return True, text.strip()

        except self._sr.UnknownValueError:
            return False, "Could not understand audio. Please speak clearly and try again."
        except self._sr.RequestError as exc:
            return False, f"Speech recognition service error: {exc}"
        except Exception as exc:
            logger.error("Unexpected STT (sounddevice) error: %s", exc)
            return False, f"Voice input error: {exc}"

    def _listen_pyaudio(self, timeout: int, phrase_limit: int) -> Tuple[bool, str]:
        """Record using PyAudio via SpeechRecognition's Microphone."""
        try:
            with self._sr.Microphone() as source:
                logger.info("Adjusting for ambient noise (PyAudio)...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Listening (PyAudio)...")
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit
                )
            text = self._recognizer.recognize_google(audio, language="en-IN")
            return True, text.strip()
        except self._sr.WaitTimeoutError:
            return False, "No speech detected. Please try again."
        except self._sr.UnknownValueError:
            return False, "Could not understand audio. Speak clearly and try again."
        except self._sr.RequestError as exc:
            return False, f"Speech recognition service error: {exc}"
        except Exception as exc:
            logger.error("Unexpected STT (PyAudio) error: %s", exc)
            return False, f"Voice input error: {exc}"

    # ------------------------------------------------------------------
    # Text-to-Speech
    # ------------------------------------------------------------------

    def speak(self, text: str, blocking: bool = False) -> bool:
        """
        Synthesize and speak the given text using pyttsx3.

        Args:
            text:      Text to speak aloud.
            blocking:  Block until speech completes (default: False).

        Returns:
            True if speech was initiated, False if TTS is unavailable.
        """
        if not self.tts_available or not text or not text.strip():
            return False

        if blocking:
            return self._speak_sync(text)
        else:
            thread = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
            thread.start()
            return True

    def _speak_sync(self, text: str) -> bool:
        """Internal synchronous TTS call."""
        with self._tts_lock:
            try:
                import pyttsx3
                import pythoncom
                
                # Initialize COM for this thread
                pythoncom.CoInitialize()
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.tts_rate)
                    engine.setProperty("volume", self.tts_volume)
                    
                    # Prefer a female voice if available
                    voices = engine.getProperty("voices")
                    if voices:
                        for voice in voices:
                            name_lower = voice.name.lower()
                            if "female" in name_lower or "zira" in name_lower or "hazel" in name_lower:
                                engine.setProperty("voice", voice.id)
                                break
                    
                    # Store current engine reference on self to support stopping
                    self._current_engine = engine
                    engine.say(text)
                    engine.runAndWait()
                    return True
                finally:
                    self._current_engine = None
                    pythoncom.CoUninitialize()
            except Exception as exc:
                logger.error("TTS speak error in background thread: %s", exc)
                return False

    def stop_speaking(self) -> None:
        """Stop any active TTS speech immediately."""
        engine = getattr(self, "_current_engine", None)
        if engine:
            try:
                engine.stop()
            except Exception as exc:
                logger.warning("Could not stop TTS: %s", exc)

    # ------------------------------------------------------------------
    # Status & Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current status of STT and TTS subsystems."""
        backend = "none"
        if self._use_sounddevice:
            backend = "sounddevice"
        elif self._use_pyaudio:
            backend = "pyaudio"

        return {
            "stt_available": self.stt_available,
            "stt_backend": backend,
            "tts_available": self.tts_available,
            "tts_rate": self.tts_rate,
            "tts_volume": self.tts_volume,
        }

    def list_tts_voices(self) -> list:
        """Return a list of available TTS voices."""
        if not self.tts_available:
            return []
        try:
            voices = self._tts_engine.getProperty("voices")
            return [{"id": v.id, "name": v.name} for v in voices]
        except Exception:
            return []

    def set_tts_rate(self, rate: int) -> None:
        """Change the TTS speaking rate at runtime."""
        self.tts_rate = rate
        if self.tts_available:
            try:
                self._tts_engine.setProperty("rate", rate)
            except Exception as exc:
                logger.warning("Could not set TTS rate: %s", exc)
