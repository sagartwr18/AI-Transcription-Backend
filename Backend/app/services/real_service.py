import logging
import queue
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.listen.v1.types.listen_v1metadata import ListenV1Metadata

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE          = 16000
SESSION_INIT_TIMEOUT_SECONDS = 10

# ─── Module-level state ───────────────────────────────────────────────────────

_live_lock    = threading.Lock()
_runtime_lock = threading.Lock()

_live_thread:      threading.Thread | None = None
_active_connection = None
_audio_queue:      queue.Queue[bytes | None] = queue.Queue()
_stream_stop_event  = threading.Event()
_final_result_event = threading.Event()

# Configuration set by start_live_session()
_session_context: dict[str, Any] = {
    'session_name': 'Realtime Session',
    'speakers':     [],                  # list[str] — speaker names
    'sample_rate':  DEFAULT_SAMPLE_RATE,
}

# Runtime state mutated during the session (protected by _runtime_lock)
_runtime: dict[str, Any] = {
    'session_id':              None,
    'session_start':           None,
    'current_speaker':         None,   # str | None — name of active speaker
    'speaker_text':            {},     # {name: str}  accumulated text per speaker
    'full_text':               '',     # entire session transcript
    'completed_turns':         [],     # list of finalized turn dicts
    '_audio_duration_seconds': 0.0,
    '_speech_model':           None,
    # print-buffer (used only for live terminal echo — harmless in API mode)
    '_print_buffer':           '',
    '_print_speaker':          None,
}

# Live state exposed through live_status() / live_updates()
_live_state: dict[str, Any] = {
    'running':               False,
    'last_error':            None,
    'started_at':            None,
    'session_name':          None,
    'speakers':              [],
    'current_speaker':       None,
    'sample_rate':           None,
    'session_id':            None,
    'live_updates':          [],
    'audio_chunks_received': 0,
    'final_result':          None,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_runtime_state() -> None:
    global _audio_queue

    with _runtime_lock:
        _runtime['session_id']              = None
        _runtime['session_start']           = _utc_now()
        _runtime['current_speaker']         = None
        _runtime['speaker_text']            = {name: '' for name in _session_context['speakers']}
        _runtime['full_text']               = ''
        _runtime['completed_turns']         = []
        _runtime['_audio_duration_seconds'] = 0.0
        _runtime['_speech_model']           = None
        _runtime['_print_buffer']           = ''
        _runtime['_print_speaker']          = None

    _audio_queue = queue.Queue()
    _stream_stop_event.clear()
    _final_result_event.clear()


def _append_live_update(update: dict[str, Any]) -> None:
    with _live_lock:
        _live_state['live_updates'].append(update)


# ─── Speaker assignment ───────────────────────────────────────────────────────

def set_active_speaker(speaker_name: str) -> dict[str, Any]:
    """
    Called by the frontend when the user clicks a speaker name.
    All transcript text received AFTER this call is attributed to speaker_name.
    """
    with _live_lock:
        if not _live_state['running']:
            raise RuntimeError('No live session is currently running')
        speakers = _live_state['speakers']

    if speaker_name not in speakers:
        raise ValueError(f"Unknown speaker '{speaker_name}'. Valid: {speakers}")

    with _runtime_lock:
        _runtime['current_speaker'] = speaker_name

    with _live_lock:
        _live_state['current_speaker'] = speaker_name

    _append_live_update({
        'type':           'speaker_change',
        'speaker':        speaker_name,
        'speaker_label':  speaker_name,
        'text':           f'Active speaker → {speaker_name}',
        'timestamp':      _utc_now(),
    })

    logger.info("Active speaker set to: %s", speaker_name)
    return {'active_speaker': speaker_name}


# ─── Transcript handler ───────────────────────────────────────────────────────

def _on_transcript(transcript: str, result_end: float) -> None:
    """
    Called for every final Deepgram transcript chunk.
    Routes the text to the currently active speaker.
    """
    transcript = transcript.strip()
    if not transcript:
        return

    with _runtime_lock:
        speaker = _runtime['current_speaker']

        # Update audio duration
        if result_end > _runtime['_audio_duration_seconds']:
            _runtime['_audio_duration_seconds'] = result_end

        # Append to full text
        _runtime['full_text'] += ' ' + transcript

        # Append to speaker bucket
        if speaker:
            _runtime['speaker_text'][speaker] += ' ' + transcript

        # Build / append a completed turn
        turns = _runtime['completed_turns']
        if turns and turns[-1]['speaker'] == speaker:
            # Same speaker continuing — merge into the last turn
            turns[-1]['text'] += ' ' + transcript
            turns[-1]['end_ms'] = int(result_end * 1000)
        else:
            # New turn
            turns.append({
                'speaker':       speaker or 'Unassigned',
                'speaker_label': speaker or 'Unassigned',
                'text':          transcript,
                'timestamp':     _utc_now(),
                'end_ms':        int(result_end * 1000),
            })

    _append_live_update({
        'type':          'final',
        'speaker':       speaker or 'Unassigned',
        'speaker_label': speaker or 'Unassigned',
        'text':          transcript,
        'timestamp':     _utc_now(),
    })


# ─── Final result builder ─────────────────────────────────────────────────────

def _build_final_result(audio_duration: float) -> dict[str, Any]:
    with _runtime_lock:
        completed_turns  = list(_runtime['completed_turns'])
        speaker_text     = {k: v.strip() for k, v in _runtime['speaker_text'].items()}
        full_text        = _runtime['full_text'].strip()
        session_id       = _runtime['session_id']
        speech_model     = _runtime['_speech_model']

    return {
        'metadata': {
            'transcript_id':          session_id,
            'status':                 'TranscriptStatus.completed',
            'audio_duration_seconds': audio_duration,
            'speech_models':          [speech_model] if speech_model else [],
            'source':                 _session_context['session_name'],
            'speakers':               _session_context['speakers'],
            'saved_at':               _utc_now(),
        },
        'full_text':        full_text,
        'per_speaker_text':     speaker_text,      # {name: full accumulated text}
        'speaker_segments': completed_turns,   # ordered turn-by-turn breakdown
    }


# ─── Deepgram message handlers ────────────────────────────────────────────────

def _on_message_result(result: Any) -> None:
    channel = getattr(result, 'channel', None)
    if not channel or not getattr(channel, 'alternatives', None):
        return

    alt        = channel.alternatives[0]
    transcript = getattr(alt, 'transcript', '') or ''
    is_final   = bool(getattr(result, 'is_final', False))

    if not is_final or not transcript:
        return

    result_start    = float(getattr(result, 'start', 0.0) or 0.0)
    result_duration = float(getattr(result, 'duration', 0.0) or 0.0)
    result_end      = result_start + result_duration

    _on_transcript(transcript, result_end)


def _on_message_metadata(metadata: Any) -> None:
    request_id = (
        getattr(metadata, 'request_id', None)
        or getattr(metadata, 'id', None)
    )
    if not request_id:
        return

    with _runtime_lock:
        _runtime['session_id'] = str(request_id)
    with _live_lock:
        _live_state['session_id'] = str(request_id)


# ─── Deepgram realtime session ────────────────────────────────────────────────

def _run_realtime_session() -> None:
    global _active_connection

    _reset_runtime_state()

    api_key = (settings.DEEPGRAM_API_KEY or '').strip()
    if not api_key:
        raise RuntimeError('DEEPGRAM_API_KEY is missing in Backend/.env')

    dg = DeepgramClient(api_key=api_key)

    def _event_from_args_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if 'message' in kwargs:
            return kwargs['message']
        if 'event' in kwargs:
            return kwargs['event']
        if 'error' in kwargs:
            return kwargs['error']
        if args:
            return args[-1]
        return None

    def on_open(*args, **kwargs) -> None:
        _append_live_update({
            'type':          'system',
            'speaker_label': 'SYSTEM',
            'speaker':       'System',
            'text':          'Deepgram WebSocket connection opened.',
            'timestamp':     _utc_now(),
        })

    def on_message(*args, **kwargs) -> None:
        message = _event_from_args_kwargs(args, kwargs)
        if isinstance(message, ListenV1Results):
            _on_message_result(message)
        elif isinstance(message, ListenV1Metadata):
            _on_message_metadata(message)

    def on_error(*args, **kwargs) -> None:
        error = _event_from_args_kwargs(args, kwargs)
        msg   = str(error) if error else 'Unknown Deepgram error'
        logger.error("Deepgram error: %s", msg)
        with _live_lock:
            _live_state['last_error'] = msg
        _append_live_update({
            'type':          'system',
            'speaker_label': 'SYSTEM',
            'speaker':       'System',
            'text':          f'Streaming error: {msg}',
            'timestamp':     _utc_now(),
        })

    def on_close(*args, **kwargs) -> None:
        with _runtime_lock:
            audio_duration = _runtime['_audio_duration_seconds']
        final_result = _build_final_result(audio_duration)
        with _live_lock:
            _live_state['final_result'] = final_result
        _final_result_event.set()

    try:
        with dg.listen.v1.connect(
            model        = 'nova-3',
            language     = 'multi',       # 'hi' for Hindi/Hinglish
            encoding     = 'linear16',
            sample_rate  = _session_context['sample_rate'],
            punctuate    = 'true',
            interim_results = 'true',     # we only act on is_final=True
            utterance_end_ms = 1000,
            vad_events   = 'true',
            endpointing  = 500,
            # ✅ diarize intentionally OMITTED — manual assignment replaces it
        ) as conn:

            _active_connection = conn
            with _runtime_lock:
                _runtime['_speech_model'] = 'nova-2'

            # ✅ Register ALL handlers BEFORE start_listening()
            conn.on(EventType.OPEN,    on_open)
            conn.on(EventType.MESSAGE, on_message)
            conn.on(EventType.ERROR,   on_error)
            conn.on(EventType.CLOSE,   on_close)

            _append_live_update({
                'type':          'system',
                'speaker_label': 'SYSTEM',
                'speaker':       'System',
                'text':          f'Session started. Speakers: {", ".join(_session_context["speakers"])}',
                'timestamp':     _utc_now(),
            })

            # ✅ Start listener thread AFTER handlers are registered
            listener_ready = threading.Event()

            def _listener_target() -> None:
                listener_ready.set()
                conn.start_listening()

            listener_thread = threading.Thread(target=_listener_target, daemon=True)
            listener_thread.start()
            listener_ready.wait(timeout=5)

            # ── Keepalive thread ──────────────────────────────────────
            # Deepgram closes the connection (error 1011) if it receives
            # no data for ~10s. We send a KeepAlive JSON message every 8s
            # whenever the audio queue has been idle (mic paused, slow
            # frontend, gap between speakers, etc.)
            import json as _json
            import time as _time

            _last_audio_time: list[float] = [_time.monotonic()]

            def _keepalive_target() -> None:
                while not _stream_stop_event.is_set():
                    _time.sleep(1)
                    idle_seconds = _time.monotonic() - _last_audio_time[0]
                    if idle_seconds >= 8:
                        try:
                            conn.send_keep_alive()
                            logger.debug("Sent Deepgram KeepAlive (idle %.1fs)", idle_seconds)
                        except Exception as ka_exc:
                            logger.warning("KeepAlive send failed: %s", ka_exc)
                            break

            keepalive_thread = threading.Thread(target=_keepalive_target, daemon=True)
            keepalive_thread.start()
            # ─────────────────────────────────────────────────────────

            try:
                while not _stream_stop_event.is_set():
                    try:
                        chunk = _audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if chunk is None:
                        break

                    conn.send_media(chunk)
                    _last_audio_time[0] = _time.monotonic()  # reset idle timer

            finally:
                _stream_stop_event.set()  # stops keepalive thread
                keepalive_thread.join(timeout=3)
                try:
                    conn.send_close_stream()
                except Exception:
                    pass
                listener_thread.join(timeout=10)
                _active_connection = None

    except Exception as exc:
        raise RuntimeError(
            f'Deepgram websocket connect failed: {exc.__class__.__name__}: {exc!r}'
        )


# ─── Session lifecycle ────────────────────────────────────────────────────────

def _run_live_session() -> None:
    global _live_thread
    try:
        _run_realtime_session()
    except Exception as exc:
        logger.exception("Live session error: %s", exc)
        with _live_lock:
            _live_state['last_error'] = str(exc)
        if not _final_result_event.is_set():
            with _live_lock:
                if _live_state['final_result'] is None:
                    _live_state['final_result'] = _build_final_result(audio_duration=0.0)
            _final_result_event.set()
    finally:
        with _live_lock:
            _live_state['running']     = False
            _live_state['started_at']  = None
        _live_thread = None


def start_live_session(
    *,
    session_name: str,
    speakers:     list[str],            # ← list of names instead of number_of_speakers
    sample_rate:  int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    """
    Start a realtime transcription session.

    speakers: e.g. ["Sagar", "Pratham", "Riya"]
    The frontend should call set_active_speaker(name) to route audio to a speaker.
    """
    global _live_thread

    if not speakers:
        raise ValueError('speakers list must not be empty')

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_speakers = [s for s in speakers if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

    with _live_lock:
        if _live_state['running']:
            raise RuntimeError('Live session is already running')

        _session_context['session_name'] = session_name
        _session_context['speakers']     = unique_speakers
        _session_context['sample_rate']  = sample_rate

        _live_state['session_name']          = session_name
        _live_state['speakers']              = unique_speakers
        _live_state['current_speaker']       = None
        _live_state['sample_rate']           = sample_rate
        _live_state['session_id']            = None
        _live_state['running']               = True
        _live_state['last_error']            = None
        _live_state['started_at']            = _utc_now()
        _live_state['final_result']          = None
        _live_state['live_updates']          = []
        _live_state['audio_chunks_received'] = 0

        _live_thread = threading.Thread(target=_run_live_session, daemon=True)
        _live_thread.start()

    return {
        'running':      True,
        'session_name': session_name,
        'speakers':     unique_speakers,
        'sample_rate':  sample_rate,
    }


def push_audio_chunk(chunk: bytes) -> None:
    if not chunk:
        return

    with _live_lock:
        if not _live_state['running']:
            raise RuntimeError('No live session is currently running')
        _live_state['audio_chunks_received'] += 1

    _audio_queue.put(chunk)


def stop_live_session() -> dict[str, Any]:
    with _live_lock:
        if not _live_state['running']:
            raise RuntimeError('No live session is currently running')

    _stream_stop_event.set()
    _audio_queue.put(None)

    _final_result_event.wait(timeout=15)

    with _live_lock:
        if _live_state['final_result'] is None:
            _live_state['final_result'] = _build_final_result(audio_duration=0.0)

        return {
            'stopping':     True,
            'session_name': _live_state['session_name'],
            'session_id':   _live_state['session_id'],
            'final_result': _live_state['final_result'],
        }


def live_status() -> dict[str, Any]:
    with _live_lock:
        return {
            'running':               _live_state['running'],
            'last_error':            _live_state['last_error'],
            'session_name':          _live_state['session_name'],
            'speakers':              _live_state['speakers'],
            'current_speaker':       _live_state['current_speaker'],
            'sample_rate':           _live_state['sample_rate'],
            'session_id':            _live_state['session_id'],
            'final_result':          _live_state['final_result'],
            'live_updates_count':    len(_live_state['live_updates']),
            'audio_chunks_received': _live_state['audio_chunks_received'],
        }


def live_updates(*, since_index: int = 0) -> dict[str, Any]:
    with _live_lock:
        started_at = _live_state.get('started_at')
        if _live_state['running'] and _live_state['session_id'] is None and started_at:
            started_dt = datetime.fromisoformat(started_at)
            elapsed    = datetime.now(timezone.utc) - started_dt
            if (
                elapsed.total_seconds() > SESSION_INIT_TIMEOUT_SECONDS
                and _live_state.get('audio_chunks_received', 0) == 0
                and not _live_state['last_error']
            ):
                _live_state['last_error'] = (
                    'Realtime session started but no audio chunks are arriving from the browser. '
                    'Check WebSocket connection and microphone permissions.'
                )

        updates    = _live_state['live_updates'][since_index:]  
        next_index = since_index + len(updates)

        return {
            'running':               _live_state['running'],
            'last_error':            _live_state['last_error'],
            'session_name':          _live_state['session_name'],
            'speakers':              _live_state['speakers'],
            'current_speaker':       _live_state['current_speaker'],
            'session_id':            _live_state['session_id'],
            'updates':               updates,
            'next_index':            next_index,
            'audio_chunks_received': _live_state['audio_chunks_received'],
            'final_result':          _live_state['final_result'],
        }
