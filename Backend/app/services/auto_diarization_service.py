import logging
import queue
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1metadata import ListenV1Metadata
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE          = 16000
MIN_WORDS_TO_KEEP            = 20
EXTRA_SPEAKERS               = 10
SESSION_INIT_TIMEOUT_SECONDS = 10
PROVIDER_MAX_SPEAKERS_CAP    = 10
KEEPALIVE_IDLE_SECONDS       = 8
KEEPALIVE_POLL_SECONDS       = 1

# ─── Global sessions registry ─────────────────────────────────────────────────

_registry_lock = threading.Lock()
_sessions: dict[str, "AutoSessionState"] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Per-session state container ──────────────────────────────────────────────

class AutoSessionState:
    """
    Holds ALL mutable state for a single auto-diarization session.
    Multiple sessions can coexist safely because each has its own instance.
    """

    def __init__(
        self,
        session_id: str,
        session_name: str,
        number_of_speakers: int,
        sample_rate: int,
    ) -> None:
        self.session_id   = session_id
        self.session_name = session_name
        self.sample_rate  = sample_rate

        requested               = max(1, number_of_speakers)
        configured_max_speakers = requested + EXTRA_SPEAKERS
        effective_max_speakers  = min(configured_max_speakers, PROVIDER_MAX_SPEAKERS_CAP)

        self.requested_speakers    = requested
        self.max_speakers          = configured_max_speakers
        self.effective_max_speakers = effective_max_speakers

        # Locks
        self._live_lock = threading.Lock()

        # Threading primitives
        self.audio_queue        = queue.Queue()
        self.stream_stop_event  = threading.Event()
        self.final_result_event = threading.Event()
        self.live_thread: threading.Thread | None = None
        self.active_connection  = None

        # Runtime (transcript state) — not lock-protected separately,
        # only mutated from the single session thread
        self.runtime: dict[str, Any] = {
            "dg_request_id":           None,
            "session_start":           _utc_now(),
            "completed_turns":         [],
            "speaker_word_count":      defaultdict(int),
            "_audio_duration_seconds": 0.0,
            "_speech_model":           None,
        }

        # Live state (exposed to API callers)
        self.live_state: dict[str, Any] = {
            "running":               True,
            "last_error":            None,
            "started_at":            _utc_now(),
            "session_name":          session_name,
            "requested_speakers":    requested,
            "max_speakers":          configured_max_speakers,
            "effective_max_speakers": effective_max_speakers,
            "sample_rate":           sample_rate,
            "session_id":            None,       # filled by Deepgram metadata
            "live_updates":          [],
            "audio_chunks_received": 0,
            "final_result":          None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def append_live_update(self, update: dict[str, Any]) -> None:
        with self._live_lock:
            self.live_state["live_updates"].append(update)

    # ── Ghost speaker filter ──────────────────────────────────────────────────

    def _filter_ghost_speakers(self, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        real_speakers = {
            speaker
            for speaker, word_count in self.runtime["speaker_word_count"].items()
            if word_count >= MIN_WORDS_TO_KEEP
        }
        ghost_speakers = set(self.runtime["speaker_word_count"].keys()) - real_speakers

        if not ghost_speakers:
            return [dict(turn) for turn in turns]

        clean_turns: list[dict[str, Any]] = []
        pending_text: list[str] = []

        for turn in turns:
            if turn["speaker_label"] in ghost_speakers:
                pending_text.append(turn["text"])
                continue

            updated_turn = dict(turn)

            if pending_text and clean_turns:
                clean_turns[-1]["text"] += " " + " ".join(pending_text)
                pending_text = []
            elif pending_text:
                updated_turn["text"] = " ".join(pending_text) + " " + updated_turn["text"]
                pending_text = []

            clean_turns.append(updated_turn)

        if pending_text and clean_turns:
            clean_turns[-1]["text"] += " " + " ".join(pending_text)

        return clean_turns

    # ── Final result builder ──────────────────────────────────────────────────

    def build_final_result(self, audio_duration: float | None = None) -> dict[str, Any]:
        if audio_duration is None:
            audio_duration = self.runtime["_audio_duration_seconds"]

        completed_turns = self.runtime["completed_turns"]
        clean_turns     = self._filter_ghost_speakers(completed_turns)

        labels      = sorted({turn["speaker_label"] for turn in clean_turns})
        speaker_map = {label: f"Speaker {label}" for label in labels}

        per_speaker_raw: dict[str, list[str]] = defaultdict(list)
        speaker_segments: list[dict[str, str]] = []

        for turn in clean_turns:
            label        = turn["speaker_label"]
            speaker_name = speaker_map.get(label, f"Speaker {label}")
            per_speaker_raw[speaker_name].append(turn["text"])
            speaker_segments.append({
                "speaker":       speaker_name,
                "speaker_label": label,
                "text":          turn["text"],
            })

        per_speaker_text = {
            name: " ".join(texts)
            for name, texts in per_speaker_raw.items()
        }
        full_text = " ".join(seg["text"] for seg in speaker_segments)

        return {
            "metadata": {
                "transcript_id":          self.runtime["dg_request_id"] or self.session_id,
                "status":                 "TranscriptStatus.completed",
                "language_code":          None,
                "audio_duration_seconds": audio_duration,
                "speech_models":          [self.runtime["_speech_model"]] if self.runtime["_speech_model"] else [],
                "source":                 self.session_name,
                "speaker_map":            speaker_map,
                "saved_at":               _utc_now(),
            },
            "full_text":        full_text,
            "per_speaker_text": per_speaker_text,
            "speaker_segments": speaker_segments,
        }

    # ── Deepgram message handlers ─────────────────────────────────────────────

    def on_message_result(self, result: Any) -> None:
        channel = getattr(result, "channel", None)
        if not channel or not getattr(channel, "alternatives", None):
            return

        alt        = channel.alternatives[0]
        transcript = getattr(alt, "transcript", "") or ""
        if not transcript:
            return

        is_final = bool(getattr(result, "is_final", False))
        if not is_final:
            return

        words = getattr(alt, "words", []) or []
        if not words:
            return

        result_start    = float(getattr(result, "start", 0.0) or 0.0)
        result_duration = float(getattr(result, "duration", 0.0) or 0.0)
        result_end      = result_start + result_duration

        if result_end > self.runtime["_audio_duration_seconds"]:
            self.runtime["_audio_duration_seconds"] = result_end

        confidence = round(float(getattr(alt, "confidence", 0.0) or 0.0), 3)

        # Group consecutive words by speaker
        speaker_segments: list[dict[str, Any]] = []
        current_speaker = None
        current_words: list[Any] = []

        for word in words:
            word_speaker = str(getattr(word, "speaker", None) or "0")
            if word_speaker != current_speaker:
                if current_words:
                    speaker_segments.append({"speaker": current_speaker, "words": current_words})
                current_speaker = word_speaker
                current_words   = [word]
            else:
                current_words.append(word)

        if current_words:
            speaker_segments.append({"speaker": current_speaker, "words": current_words})

        for segment in speaker_segments:
            speaker_label  = segment["speaker"]
            segment_words  = segment["words"]
            segment_text   = " ".join(
                (getattr(w, "punctuated_word", None) or getattr(w, "word", "") or "").strip()
                for w in segment_words
            ).strip()

            if not segment_text:
                continue

            segment_start = float(getattr(segment_words[0], "start", result_start) or result_start)
            segment_end   = float(getattr(segment_words[-1], "end", result_end) or result_end)
            word_count    = len(segment_text.split())

            finalized_turn = {
                "speaker_label": speaker_label,
                "speaker":       f"Speaker {speaker_label}",
                "text":          segment_text,
                "word_count":    word_count,
                "confidence":    confidence,
                "timestamp":     _utc_now(),
                "start_ms":      int(segment_start * 1000),
                "end_ms":        int(segment_end * 1000),
            }

            self.runtime["completed_turns"].append(finalized_turn)
            self.runtime["speaker_word_count"][speaker_label] += word_count

            self.append_live_update({
                "type":          "final",
                "speaker_label": speaker_label,
                "speaker":       f"Speaker {speaker_label}",
                "text":          segment_text,
                "timestamp":     finalized_turn["timestamp"],
            })

    def on_message_metadata(self, metadata: Any) -> None:
        req_id = getattr(metadata, "request_id", None) or getattr(metadata, "id", None)
        if not req_id:
            return
        self.runtime["dg_request_id"] = str(req_id)
        with self._live_lock:
            self.live_state["session_id"] = str(req_id)

    # ── Deepgram connection runner ────────────────────────────────────────────

    def run_realtime_session(self) -> None:
        api_key = (settings.DEEPGRAM_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is missing in Backend/.env")

        dg = DeepgramClient(api_key=api_key)

        def _extract_event(args, kwargs):
            for key in ("message", "event", "error"):
                if key in kwargs:
                    return kwargs[key]
            return args[-1] if args else None

        def on_open(*args, **kwargs):
            self.append_live_update({
                "type": "system", "speaker_label": "SYSTEM", "speaker": "System",
                "text": "Deepgram automatic diarization session started.",
                "timestamp": _utc_now(),
            })

        def on_message(*args, **kwargs):
            msg = _extract_event(args, kwargs)
            if isinstance(msg, ListenV1Results):
                self.on_message_result(msg)
            elif isinstance(msg, ListenV1Metadata):
                self.on_message_metadata(msg)

        def on_error(*args, **kwargs):
            err = _extract_event(args, kwargs)
            msg = str(err) if err else "Unknown Deepgram error"
            logger.error("[%s] Deepgram error: %s", self.session_id, msg)
            with self._live_lock:
                self.live_state["last_error"] = msg
            self.append_live_update({
                "type": "system", "speaker_label": "SYSTEM", "speaker": "System",
                "text": f"Streaming error: {msg}", "timestamp": _utc_now(),
            })

        def on_close(*args, **kwargs):
            final_result = self.build_final_result()
            with self._live_lock:
                self.live_state["final_result"] = final_result
            self.final_result_event.set()

        try:
            with dg.listen.v1.connect(
                model           = "nova-3",
                language        = "multi",
                encoding        = "linear16",
                sample_rate     = self.sample_rate,
                diarize         = "true",
                punctuate       = "true",
                smart_format    = "true",
                interim_results = "true",
                endpointing     = 300,
            ) as conn:
                self.active_connection          = conn
                self.runtime["_speech_model"]   = "nova-3"

                conn.on(EventType.OPEN,    on_open)
                conn.on(EventType.MESSAGE, on_message)
                conn.on(EventType.ERROR,   on_error)
                conn.on(EventType.CLOSE,   on_close)

                listener_ready = threading.Event()

                def _listener():
                    listener_ready.set()
                    conn.start_listening()

                listener_thread = threading.Thread(target=_listener, daemon=True)
                listener_thread.start()
                listener_ready.wait(timeout=5)

                self.append_live_update({
                    "type": "system", "speaker_label": "SYSTEM", "speaker": "System",
                    "text": "Deepgram model in use: nova-3 with automatic diarization.",
                    "timestamp": _utc_now(),
                })

                # ── Keepalive thread ──────────────────────────────────
                last_audio_time: list[float] = [time.monotonic()]

                def _keepalive():
                    while not self.stream_stop_event.is_set():
                        time.sleep(KEEPALIVE_POLL_SECONDS)
                        idle = time.monotonic() - last_audio_time[0]
                        if idle < KEEPALIVE_IDLE_SECONDS:
                            continue
                        try:
                            conn.send_keep_alive()
                            logger.debug("[%s] KeepAlive sent (idle %.1fs)", self.session_id, idle)
                        except Exception as e:
                            logger.warning("[%s] KeepAlive failed: %s", self.session_id, e)
                            break

                keepalive_thread = threading.Thread(target=_keepalive, daemon=True)
                keepalive_thread.start()
                # ─────────────────────────────────────────────────────

                try:
                    while not self.stream_stop_event.is_set():
                        try:
                            chunk = self.audio_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        if chunk is None:
                            break
                        conn.send_media(chunk)
                        last_audio_time[0] = time.monotonic()
                        time.sleep(0.002)
                finally:
                    self.stream_stop_event.set()
                    keepalive_thread.join(timeout=3)
                    try:
                        conn.send_close_stream()
                    except Exception:
                        pass
                    listener_thread.join(timeout=10)
                    self.active_connection = None

        except Exception as exc:
            raise RuntimeError(
                f"Deepgram connect failed: {exc.__class__.__name__}: {exc!r}"
            )

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self.run_realtime_session()
        except Exception as exc:
            logger.exception("[%s] Auto session error: %s", self.session_id, exc)
            with self._live_lock:
                self.live_state["last_error"] = str(exc)
            if not self.final_result_event.is_set():
                with self._live_lock:
                    if self.live_state["final_result"] is None:
                        self.live_state["final_result"] = self.build_final_result(0.0)
                self.final_result_event.set()
        finally:
            with self._live_lock:
                self.live_state["running"]    = False
                self.live_state["started_at"] = None

            with _registry_lock:
                _sessions.pop(self.session_id, None)


# ─── Public API ───────────────────────────────────────────────────────────────

def start_live_session(
    *,
    session_name:       str,
    number_of_speakers: int,
    sample_rate:        int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    """
    Start a NEW isolated auto-diarization session.
    Returns a session_id the caller must pass to all subsequent calls.
    """
    session_id = str(uuid.uuid4())

    state = AutoSessionState(
        session_id         = session_id,
        session_name       = session_name,
        number_of_speakers = number_of_speakers,
        sample_rate        = sample_rate,
    )

    with _registry_lock:
        _sessions[session_id] = state

    state.live_thread = threading.Thread(target=state.run, daemon=True)
    state.live_thread.start()

    logger.info("Started auto session %s (speakers=%d)", session_id, number_of_speakers)

    return {
        "session_id":            session_id,       # ← frontend MUST store this
        "running":               True,
        "session_name":          session_name,
        "requested_speakers":    state.requested_speakers,
        "max_speakers":          state.max_speakers,
        "effective_max_speakers": state.effective_max_speakers,
        "sample_rate":           sample_rate,
    }


def _get_session(session_id: str) -> AutoSessionState:
    with _registry_lock:
        state = _sessions.get(session_id)
    if state is None:
        raise RuntimeError(f"No active auto session with id '{session_id}'")
    return state


def push_audio_chunk(session_id: str, chunk: bytes) -> None:
    if not chunk:
        return
    state = _get_session(session_id)
    with state._live_lock:
        if not state.live_state["running"]:
            raise RuntimeError("Session is not running")
        state.live_state["audio_chunks_received"] += 1
    state.audio_queue.put(chunk)


def stop_live_session(session_id: str) -> dict[str, Any]:
    state = _get_session(session_id)

    with state._live_lock:
        if not state.live_state["running"]:
            raise RuntimeError("Session is not running")

    state.stream_stop_event.set()
    state.audio_queue.put(None)
    state.final_result_event.wait(timeout=15)

    with state._live_lock:
        if state.live_state["final_result"] is None:
            state.live_state["final_result"] = state.build_final_result(0.0)
        return {
            "stopping":     True,
            "session_name": state.live_state["session_name"],
            "session_id":   state.live_state["session_id"],
            "final_result": state.live_state["final_result"],
        }


def live_status(session_id: str) -> dict[str, Any]:
    state = _get_session(session_id)
    with state._live_lock:
        ls = state.live_state
        return {
            "running":               ls["running"],
            "last_error":            ls["last_error"],
            "session_name":          ls["session_name"],
            "requested_speakers":    ls["requested_speakers"],
            "max_speakers":          ls["max_speakers"],
            "effective_max_speakers": ls["effective_max_speakers"],
            "sample_rate":           ls["sample_rate"],
            "session_id":            ls["session_id"],
            "final_result":          ls["final_result"],
            "live_updates_count":    len(ls["live_updates"]),
            "audio_chunks_received": ls["audio_chunks_received"],
        }


def live_updates(session_id: str, *, since_index: int = 0) -> dict[str, Any]:
    state = _get_session(session_id)
    with state._live_lock:
        ls = state.live_state

        # Session-init timeout check
        started_at = ls.get("started_at")
        if ls["running"] and ls["session_id"] is None and started_at:
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(started_at)
            if (
                elapsed.total_seconds() > SESSION_INIT_TIMEOUT_SECONDS
                and ls.get("audio_chunks_received", 0) == 0
                and not ls["last_error"]
            ):
                ls["last_error"] = (
                    "Automatic diarization session started but no audio chunks are arriving. "
                    "Check WebSocket connection and microphone permissions."
                )

        updates    = ls["live_updates"][since_index:]
        next_index = since_index + len(updates)

        return {
            "running":               ls["running"],
            "last_error":            ls["last_error"],
            "session_name":          ls["session_name"],
            "requested_speakers":    ls["requested_speakers"],
            "max_speakers":          ls["max_speakers"],
            "effective_max_speakers": ls["effective_max_speakers"],
            "session_id":            ls["session_id"],
            "updates":               updates,
            "next_index":            next_index,
            "audio_chunks_received": ls["audio_chunks_received"],
            "final_result":          ls["final_result"],
        }


def list_active_sessions() -> list[dict[str, Any]]:
    """Utility — returns summary of all currently running auto sessions."""
    with _registry_lock:
        snapshot = list(_sessions.items())
    result = []
    for sid, state in snapshot:
        with state._live_lock:
            result.append({
                "session_id":         sid,
                "session_name":       state.live_state["session_name"],
                "requested_speakers": state.live_state["requested_speakers"],
                "running":            state.live_state["running"],
                "started_at":         state.live_state["started_at"],
            })
    return result