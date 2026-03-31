import logging
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1metadata import ListenV1Metadata
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
MIN_WORDS_TO_KEEP = 20
EXTRA_SPEAKERS = 10
SESSION_INIT_TIMEOUT_SECONDS = 10
PROVIDER_MAX_SPEAKERS_CAP = 10
KEEPALIVE_IDLE_SECONDS = 8
KEEPALIVE_POLL_SECONDS = 1

_live_lock = threading.Lock()
_live_thread: threading.Thread | None = None
_active_connection = None
_audio_queue: queue.Queue[bytes | None] = queue.Queue()
_stream_stop_event = threading.Event()
_final_result_event = threading.Event()

_session_context: dict[str, Any] = {
    "session_name": "Automatic Diarization Session",
    "requested_speakers": 0,
    "max_speakers": 10,
    "effective_max_speakers": 10,
    "sample_rate": DEFAULT_SAMPLE_RATE,
}

_runtime: dict[str, Any] = {
    "session_id": None,
    "session_start": None,
    "completed_turns": [],
    "speaker_word_count": defaultdict(int),
    "_audio_duration_seconds": 0.0,
    "_speech_model": None,
}

_live_state: dict[str, Any] = {
    "running": False,
    "last_error": None,
    "started_at": None,
    "session_name": None,
    "requested_speakers": None,
    "max_speakers": None,
    "effective_max_speakers": None,
    "sample_rate": None,
    "session_id": None,
    "live_updates": [],
    "audio_chunks_received": 0,
    "final_result": None,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_runtime_state() -> None:
    global _audio_queue

    _runtime["session_id"] = None
    _runtime["session_start"] = _utc_now()
    _runtime["completed_turns"] = []
    _runtime["speaker_word_count"] = defaultdict(int)
    _runtime["_audio_duration_seconds"] = 0.0
    _runtime["_speech_model"] = None

    _audio_queue = queue.Queue()
    _stream_stop_event.clear()
    _final_result_event.clear()


def _append_live_update(update: dict[str, Any]) -> None:
    with _live_lock:
        _live_state["live_updates"].append(update)


def _filter_ghost_speakers(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_speakers = {
        speaker
        for speaker, word_count in _runtime["speaker_word_count"].items()
        if word_count >= MIN_WORDS_TO_KEEP
    }
    ghost_speakers = set(_runtime["speaker_word_count"].keys()) - real_speakers

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


def _build_final_result(audio_duration: float) -> dict[str, Any]:
    completed_turns: list[dict[str, Any]] = _runtime["completed_turns"]
    clean_turns = _filter_ghost_speakers(completed_turns)

    labels = sorted({turn["speaker_label"] for turn in clean_turns})
    speaker_map = {label: f"Speaker {label}" for label in labels}

    per_speaker_raw: dict[str, list[str]] = defaultdict(list)
    speaker_segments: list[dict[str, str]] = []

    for turn in clean_turns:
        label = turn["speaker_label"]
        speaker_name = speaker_map.get(label, f"Speaker {label}")
        per_speaker_raw[speaker_name].append(turn["text"])
        speaker_segments.append(
            {
                "speaker": speaker_name,
                "speaker_label": label,
                "text": turn["text"],
            }
        )

    per_speaker_text = {
        speaker_name: " ".join(texts)
        for speaker_name, texts in per_speaker_raw.items()
    }

    full_text = " ".join(segment["text"] for segment in speaker_segments)

    return {
        "metadata": {
            "transcript_id": _runtime["session_id"],
            "status": "TranscriptStatus.completed",
            "language_code": None,
            "audio_duration_seconds": audio_duration,
            "speech_models": [_runtime["_speech_model"]] if _runtime["_speech_model"] else [],
            "source": _session_context["session_name"],
            "speaker_map": speaker_map,
            "saved_at": _utc_now(),
        },
        "full_text": full_text,
        "per_speaker_text": per_speaker_text,
        "speaker_segments": speaker_segments,
    }


def _on_message_result(result: Any) -> None:
    channel = getattr(result, "channel", None)
    if not channel or not getattr(channel, "alternatives", None):
        return

    alt = channel.alternatives[0]
    transcript: str = getattr(alt, "transcript", "") or ""
    if not transcript:
        return

    is_final: bool = bool(getattr(result, "is_final", False))
    if not is_final:
        return

    words = getattr(alt, "words", []) or []
    if not words:
        return

    result_start: float = float(getattr(result, "start", 0.0) or 0.0)
    result_duration: float = float(getattr(result, "duration", 0.0) or 0.0)
    result_end: float = result_start + result_duration

    if result_end > _runtime["_audio_duration_seconds"]:
        _runtime["_audio_duration_seconds"] = result_end

    confidence = round(float(getattr(alt, "confidence", 0.0) or 0.0), 3)

    speaker_segments: list[dict[str, Any]] = []
    current_speaker = None
    current_words: list[Any] = []

    for word in words:
        word_speaker = str(getattr(word, "speaker", None) or "0")
        if word_speaker != current_speaker:
            if current_words:
                speaker_segments.append({
                    "speaker": current_speaker,
                    "words": current_words,
                })
            current_speaker = word_speaker
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        speaker_segments.append({
            "speaker": current_speaker,
            "words": current_words,
        })

    for segment in speaker_segments:
        speaker_label = segment["speaker"]
        segment_words = segment["words"]

        segment_text = " ".join(
            (getattr(word, "punctuated_word", None) or getattr(word, "word", "") or "").strip()
            for word in segment_words
        ).strip()

        if not segment_text:
            continue

        segment_start = float(getattr(segment_words[0], "start", result_start) or result_start)
        segment_end = float(getattr(segment_words[-1], "end", result_end) or result_end)
        word_count = len(segment_text.split())

        finalized_turn = {
            "speaker_label": speaker_label,
            "speaker": f"Speaker {speaker_label}",
            "text": segment_text,
            "word_count": word_count,
            "confidence": confidence,
            "timestamp": _utc_now(),
            "start_ms": int(segment_start * 1000),
            "end_ms": int(segment_end * 1000),
        }

        _runtime["completed_turns"].append(finalized_turn)
        _runtime["speaker_word_count"][speaker_label] += word_count

        _append_live_update({
            "type": "final",
            "speaker_label": speaker_label,
            "speaker": f"Speaker {speaker_label}",
            "text": segment_text,
            "timestamp": finalized_turn["timestamp"],
        })


def _on_message_metadata(metadata: Any) -> None:
    request_id = getattr(metadata, "request_id", None) or getattr(metadata, "id", None)
    if not request_id:
        return

    _runtime["session_id"] = str(request_id)
    with _live_lock:
        _live_state["session_id"] = str(request_id)


def _run_realtime_session() -> None:
    global _active_connection

    _reset_runtime_state()

    api_key = (settings.DEEPGRAM_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is missing in Backend/.env")

    dg = DeepgramClient(api_key=api_key)

    def _event_from_args_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if "message" in kwargs:
            return kwargs.get("message")
        if "event" in kwargs:
            return kwargs.get("event")
        if "error" in kwargs:
            return kwargs.get("error")
        if args:
            return args[-1]
        return None

    def on_open(*args, **kwargs) -> None:
        _append_live_update({
            "type": "system",
            "speaker_label": "SYSTEM",
            "speaker": "System",
            "text": "Deepgram automatic diarization session started.",
            "timestamp": _utc_now(),
        })

    def on_message(*args, **kwargs) -> None:
        message = _event_from_args_kwargs(args, kwargs)
        if isinstance(message, ListenV1Results):
            _on_message_result(message)
            return
        if isinstance(message, ListenV1Metadata):
            _on_message_metadata(message)

    def on_error(*args, **kwargs) -> None:
        error = _event_from_args_kwargs(args, kwargs)
        msg = str(error) if error else "Unknown Deepgram error"
        with _live_lock:
            _live_state["last_error"] = msg
        _append_live_update({
            "type": "system",
            "speaker_label": "SYSTEM",
            "speaker": "System",
            "text": f"Streaming error: {msg}",
            "timestamp": _utc_now(),
        })

    def on_close(*args, **kwargs) -> None:
        audio_duration = _runtime["_audio_duration_seconds"]
        final_result = _build_final_result(audio_duration)
        with _live_lock:
            _live_state["final_result"] = final_result
        _final_result_event.set()

    try:
        with dg.listen.v1.connect(
            model="nova-3",
            language="multi",
            encoding="linear16",
            sample_rate=_session_context["sample_rate"],
            diarize="true",
            punctuate="true",
            smart_format="true",
            interim_results="true",
            endpointing=300,
        ) as conn:
            _active_connection = conn
            _runtime["_speech_model"] = "nova-3"

            conn.on(EventType.OPEN, on_open)
            conn.on(EventType.MESSAGE, on_message)
            conn.on(EventType.ERROR, on_error)
            conn.on(EventType.CLOSE, on_close)

            listener_ready = threading.Event()

            def _listener_target() -> None:
                listener_ready.set()
                conn.start_listening()

            listener_thread = threading.Thread(target=_listener_target, daemon=True)
            listener_thread.start()
            listener_ready.wait(timeout=5)

            last_audio_time = [time.monotonic()]

            def _keepalive_target() -> None:
                while not _stream_stop_event.is_set():
                    time.sleep(KEEPALIVE_POLL_SECONDS)
                    idle_seconds = time.monotonic() - last_audio_time[0]
                    if idle_seconds < KEEPALIVE_IDLE_SECONDS:
                        continue

                    try:
                        conn.send_keep_alive()
                        logger.debug(
                            "Sent Deepgram KeepAlive for automatic diarization after %.1fs idle",
                            idle_seconds,
                        )
                    except Exception as keepalive_exc:
                        logger.warning(
                            "Automatic diarization KeepAlive send failed: %s",
                            keepalive_exc,
                        )
                        break

            keepalive_thread = threading.Thread(target=_keepalive_target, daemon=True)
            keepalive_thread.start()

            _append_live_update({
                "type": "system",
                "speaker_label": "SYSTEM",
                "speaker": "System",
                "text": "Deepgram model in use: nova-3 with automatic diarization.",
                "timestamp": _utc_now(),
            })

            try:
                while not _stream_stop_event.is_set():
                    try:
                        chunk = _audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if chunk is None:
                        break

                    if chunk:
                        conn.send_media(chunk)
                        last_audio_time[0] = time.monotonic()
                        time.sleep(0.002)
            finally:
                _stream_stop_event.set()
                keepalive_thread.join(timeout=3)
                try:
                    conn.send_close_stream()
                except Exception:
                    pass
                listener_thread.join(timeout=10)
                _active_connection = None

    except Exception as exc:
        raise RuntimeError(
            f"Deepgram websocket connect failed: {exc.__class__.__name__}: {exc!r}"
        )


def _run_live_session() -> None:
    global _live_thread

    try:
        with _live_lock:
            _live_state["running"] = True
            _live_state["last_error"] = None
            _live_state["started_at"] = _utc_now()
            _live_state["final_result"] = None
            _live_state["live_updates"] = []
            _live_state["audio_chunks_received"] = 0

        _run_realtime_session()
    except Exception as exc:
        with _live_lock:
            _live_state["last_error"] = str(exc)
        if not _final_result_event.is_set():
            with _live_lock:
                if _live_state["final_result"] is None:
                    _live_state["final_result"] = _build_final_result(audio_duration=0.0)
            _final_result_event.set()
    finally:
        with _live_lock:
            _live_state["running"] = False
            _live_state["started_at"] = None
        _live_thread = None


def start_live_session(
    *,
    session_name: str,
    number_of_speakers: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    global _live_thread

    requested = max(1, number_of_speakers)
    configured_max_speakers = requested + EXTRA_SPEAKERS
    effective_max_speakers = min(configured_max_speakers, PROVIDER_MAX_SPEAKERS_CAP)

    with _live_lock:
        if _live_state["running"]:
            raise RuntimeError("Automatic diarization live session is already running")

        _session_context["session_name"] = session_name
        _session_context["requested_speakers"] = requested
        _session_context["max_speakers"] = configured_max_speakers
        _session_context["effective_max_speakers"] = effective_max_speakers
        _session_context["sample_rate"] = sample_rate

        _live_state["session_name"] = session_name
        _live_state["requested_speakers"] = requested
        _live_state["max_speakers"] = configured_max_speakers
        _live_state["effective_max_speakers"] = effective_max_speakers
        _live_state["sample_rate"] = sample_rate
        _live_state["session_id"] = None
        _live_state["running"] = True
        _live_state["last_error"] = None
        _live_state["started_at"] = _utc_now()
        _live_state["final_result"] = None
        _live_state["live_updates"] = []
        _live_state["audio_chunks_received"] = 0

        _live_thread = threading.Thread(target=_run_live_session, daemon=True)
        _live_thread.start()

    return {
        "running": True,
        "session_name": session_name,
        "requested_speakers": requested,
        "max_speakers": configured_max_speakers,
        "effective_max_speakers": effective_max_speakers,
        "sample_rate": sample_rate,
    }


def push_audio_chunk(chunk: bytes) -> None:
    if not chunk:
        return

    with _live_lock:
        if not _live_state["running"]:
            raise RuntimeError("No automatic diarization live session is currently running")
        _live_state["audio_chunks_received"] += 1

    _audio_queue.put(chunk)


def stop_live_session() -> dict[str, Any]:
    with _live_lock:
        if not _live_state["running"]:
            raise RuntimeError("No automatic diarization live session is currently running")

    _stream_stop_event.set()
    _audio_queue.put(None)

    _final_result_event.wait(timeout=15)

    with _live_lock:
        if _live_state["final_result"] is None:
            _live_state["final_result"] = _build_final_result(audio_duration=0.0)

        return {
            "stopping": True,
            "session_name": _live_state["session_name"],
            "session_id": _live_state["session_id"],
            "final_result": _live_state["final_result"],
        }


def live_status() -> dict[str, Any]:
    with _live_lock:
        return {
            "running": _live_state["running"],
            "last_error": _live_state["last_error"],
            "session_name": _live_state["session_name"],
            "requested_speakers": _live_state["requested_speakers"],
            "max_speakers": _live_state["max_speakers"],
            "effective_max_speakers": _live_state["effective_max_speakers"],
            "sample_rate": _live_state["sample_rate"],
            "session_id": _live_state["session_id"],
            "final_result": _live_state["final_result"],
            "live_updates_count": len(_live_state["live_updates"]),
            "audio_chunks_received": _live_state["audio_chunks_received"],
        }


def live_updates(*, since_index: int = 0) -> dict[str, Any]:
    with _live_lock:
        started_at = _live_state.get("started_at")
        if _live_state["running"] and _live_state["session_id"] is None and started_at:
            started_dt = datetime.fromisoformat(started_at)
            elapsed = datetime.now(timezone.utc) - started_dt
            if (
                elapsed.total_seconds() > SESSION_INIT_TIMEOUT_SECONDS
                and _live_state.get("audio_chunks_received", 0) == 0
                and not _live_state["last_error"]
            ):
                _live_state["last_error"] = (
                    "Automatic diarization session started but no audio chunks are arriving from the browser. "
                    "Check WebSocket connection and microphone permissions."
                )

        updates = _live_state["live_updates"][since_index:]
        next_index = since_index + len(updates)
        return {
            "running": _live_state["running"],
            "last_error": _live_state["last_error"],
            "session_name": _live_state["session_name"],
            "session_id": _live_state["session_id"],
            "updates": updates,
            "next_index": next_index,
            "audio_chunks_received": _live_state["audio_chunks_received"],
            "final_result": _live_state["final_result"],
        }
