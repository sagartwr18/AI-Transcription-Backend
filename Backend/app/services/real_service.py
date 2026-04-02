import logging
import queue
import threading
import uuid
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

# ─── Global sessions registry ─────────────────────────────────────────────────
# Maps session_id (str) → SessionState object
# All per-session state is isolated inside SessionState; nothing is module-global.

_registry_lock = threading.Lock()
_sessions: dict[str, "SessionState"] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Per-session state container ──────────────────────────────────────────────

class SessionState:
    """
    Holds ALL mutable state for a single live transcription session.
    Multiple sessions can coexist safely because each has its own instance.
    """

    def __init__(
        self,
        session_id: str,
        session_name: str,
        speakers: list[str],
        sample_rate: int,
    ) -> None:
        self.session_id   = session_id
        self.session_name = session_name
        self.speakers     = speakers
        self.sample_rate  = sample_rate

        # Locks
        self._live_lock    = threading.Lock()
        self._runtime_lock = threading.Lock()

        # Threading primitives
        self.audio_queue       = queue.Queue()
        self.stream_stop_event = threading.Event()
        self.final_result_event = threading.Event()
        self.live_thread: threading.Thread | None = None
        self.active_connection = None

        # Runtime (transcript state)
        self.runtime: dict[str, Any] = {
            'dg_request_id':           None,
            'session_start':           _utc_now(),
            'current_speaker':         None,
            'speaker_text':            {name: '' for name in speakers},
            'full_text':               '',
            'completed_turns':         [],
            '_audio_duration_seconds': 0.0,
            '_speech_model':           None,
        }

        # Live state (exposed to API callers)
        self.live_state: dict[str, Any] = {
            'running':               True,
            'last_error':            None,
            'started_at':            _utc_now(),
            'session_name':          session_name,
            'speakers':              speakers,
            'current_speaker':       None,
            'sample_rate':           sample_rate,
            'session_id':            None,       # filled by Deepgram metadata
            'live_updates':          [],
            'audio_chunks_received': 0,
            'final_result':          None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def append_live_update(self, update: dict[str, Any]) -> None:
        with self._live_lock:
            self.live_state['live_updates'].append(update)

    # ── Speaker assignment ────────────────────────────────────────────────────

    def set_active_speaker(self, speaker_name: str) -> dict[str, Any]:
        with self._live_lock:
            if not self.live_state['running']:
                raise RuntimeError('Session is not running')
            speakers = self.live_state['speakers']

        if speaker_name not in speakers:
            raise ValueError(f"Unknown speaker '{speaker_name}'. Valid: {speakers}")

        with self._runtime_lock:
            self.runtime['current_speaker'] = speaker_name
        with self._live_lock:
            self.live_state['current_speaker'] = speaker_name

        self.append_live_update({
            'type':          'speaker_change',
            'speaker':       speaker_name,
            'speaker_label': speaker_name,
            'text':          f'Active speaker → {speaker_name}',
            'timestamp':     _utc_now(),
        })

        logger.info("[%s] Active speaker set to: %s", self.session_id, speaker_name)
        return {'active_speaker': speaker_name}

    # ── Transcript handling ───────────────────────────────────────────────────

    def on_transcript(self, transcript: str, result_end: float) -> None:
        transcript = transcript.strip()
        if not transcript:
            return

        with self._runtime_lock:
            speaker = self.runtime['current_speaker']

            if result_end > self.runtime['_audio_duration_seconds']:
                self.runtime['_audio_duration_seconds'] = result_end

            self.runtime['full_text'] += ' ' + transcript

            if speaker:
                self.runtime['speaker_text'][speaker] += ' ' + transcript

            turns = self.runtime['completed_turns']
            if turns and turns[-1]['speaker'] == speaker:
                turns[-1]['text']   += ' ' + transcript
                turns[-1]['end_ms']  = int(result_end * 1000)
            else:
                turns.append({
                    'speaker':       speaker or 'Unassigned',
                    'speaker_label': speaker or 'Unassigned',
                    'text':          transcript,
                    'timestamp':     _utc_now(),
                    'end_ms':        int(result_end * 1000),
                })

        self.append_live_update({
            'type':          'final',
            'speaker':       speaker or 'Unassigned',
            'speaker_label': speaker or 'Unassigned',
            'text':          transcript,
            'timestamp':     _utc_now(),
        })

    # ── Final result ──────────────────────────────────────────────────────────

    def build_final_result(self, audio_duration: float | None = None) -> dict[str, Any]:
        with self._runtime_lock:
            if audio_duration is None:
                audio_duration = self.runtime['_audio_duration_seconds']
            completed_turns = list(self.runtime['completed_turns'])
            speaker_text    = {k: v.strip() for k, v in self.runtime['speaker_text'].items()}
            full_text       = self.runtime['full_text'].strip()
            dg_request_id   = self.runtime['dg_request_id']
            speech_model    = self.runtime['_speech_model']

        return {
            'metadata': {
                'transcript_id':          dg_request_id or self.session_id,
                'status':                 'TranscriptStatus.completed',
                'audio_duration_seconds': audio_duration,
                'speech_models':          [speech_model] if speech_model else [],
                'source':                 self.session_name,
                'speakers':               self.speakers,
                'saved_at':               _utc_now(),
            },
            'full_text':        full_text,
            'per_speaker_text': speaker_text,
            'speaker_segments': completed_turns,
        }

    # ── Deepgram connection runner ────────────────────────────────────────────

    def run_realtime_session(self) -> None:
        api_key = (settings.DEEPGRAM_API_KEY or '').strip()
        if not api_key:
            raise RuntimeError('DEEPGRAM_API_KEY is missing in Backend/.env')

        dg = DeepgramClient(api_key=api_key)

        def _extract_event(args, kwargs):
            for key in ('message', 'event', 'error'):
                if key in kwargs:
                    return kwargs[key]
            return args[-1] if args else None

        def on_open(*args, **kwargs):
            self.append_live_update({
                'type': 'system', 'speaker_label': 'SYSTEM', 'speaker': 'System',
                'text': 'Deepgram WebSocket connection opened.',
                'timestamp': _utc_now(),
            })

        def on_message(*args, **kwargs):
            msg = _extract_event(args, kwargs)
            if isinstance(msg, ListenV1Results):
                ch = getattr(msg, 'channel', None)
                if not ch or not getattr(ch, 'alternatives', None):
                    return
                alt        = ch.alternatives[0]
                transcript = getattr(alt, 'transcript', '') or ''
                is_final   = bool(getattr(msg, 'is_final', False))
                if not is_final or not transcript:
                    return
                start    = float(getattr(msg, 'start', 0.0) or 0.0)
                duration = float(getattr(msg, 'duration', 0.0) or 0.0)
                self.on_transcript(transcript, start + duration)

            elif isinstance(msg, ListenV1Metadata):
                req_id = (
                    getattr(msg, 'request_id', None)
                    or getattr(msg, 'id', None)
                )
                if req_id:
                    with self._runtime_lock:
                        self.runtime['dg_request_id'] = str(req_id)
                    with self._live_lock:
                        self.live_state['session_id'] = str(req_id)

        def on_error(*args, **kwargs):
            err = _extract_event(args, kwargs)
            msg = str(err) if err else 'Unknown Deepgram error'
            logger.error("[%s] Deepgram error: %s", self.session_id, msg)
            with self._live_lock:
                self.live_state['last_error'] = msg
            self.append_live_update({
                'type': 'system', 'speaker_label': 'SYSTEM', 'speaker': 'System',
                'text': f'Streaming error: {msg}', 'timestamp': _utc_now(),
            })

        def on_close(*args, **kwargs):
            with self._runtime_lock:
                audio_duration = self.runtime['_audio_duration_seconds']
            final_result = self.build_final_result(audio_duration)
            with self._live_lock:
                self.live_state['final_result'] = final_result
            self.final_result_event.set()

        import json as _json
        import time as _time

        try:
            with dg.listen.v1.connect(
                model            = 'nova-3',
                language         = 'multi',
                encoding         = 'linear16',
                sample_rate      = self.sample_rate,
                punctuate        = 'true',
                interim_results  = 'true',
                utterance_end_ms = 1000,
                vad_events       = 'true',
                endpointing      = 500,
            ) as conn:
                self.active_connection = conn

                with self._runtime_lock:
                    self.runtime['_speech_model'] = 'nova-3'

                conn.on(EventType.OPEN,    on_open)
                conn.on(EventType.MESSAGE, on_message)
                conn.on(EventType.ERROR,   on_error)
                conn.on(EventType.CLOSE,   on_close)

                self.append_live_update({
                    'type': 'system', 'speaker_label': 'SYSTEM', 'speaker': 'System',
                    'text': f'Session started. Speakers: {", ".join(self.speakers)}',
                    'timestamp': _utc_now(),
                })

                listener_ready = threading.Event()

                def _listener():
                    listener_ready.set()
                    conn.start_listening()

                listener_thread = threading.Thread(target=_listener, daemon=True)
                listener_thread.start()
                listener_ready.wait(timeout=5)

                # ── Keepalive thread ──────────────────────────────────
                _last_audio: list[float] = [_time.monotonic()]

                def _keepalive():
                    while not self.stream_stop_event.is_set():
                        _time.sleep(1)
                        if _time.monotonic() - _last_audio[0] >= 8:
                            try:
                                conn.send_keep_alive()
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
                        _last_audio[0] = _time.monotonic()
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
                f'Deepgram connect failed: {exc.__class__.__name__}: {exc!r}'
            )

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self.run_realtime_session()
        except Exception as exc:
            logger.exception("[%s] Session error: %s", self.session_id, exc)
            with self._live_lock:
                self.live_state['last_error'] = str(exc)
            if not self.final_result_event.is_set():
                with self._live_lock:
                    if self.live_state['final_result'] is None:
                        self.live_state['final_result'] = self.build_final_result(0.0)
                self.final_result_event.set()
        finally:
            with self._live_lock:
                self.live_state['running']    = False
                self.live_state['started_at'] = None

            # Auto-remove from registry when done
            with _registry_lock:
                _sessions.pop(self.session_id, None)


# ─── Public API ───────────────────────────────────────────────────────────────

def start_live_session(
    *,
    session_name: str,
    speakers:     list[str],
    sample_rate:  int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    """
    Start a NEW isolated transcription session.
    Returns a session_id that the caller MUST pass to all subsequent calls.
    Multiple sessions can run concurrently on different laptops/tabs.
    """
    if not speakers:
        raise ValueError('speakers list must not be empty')

    seen: set[str] = set()
    unique_speakers = [s for s in speakers if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

    session_id = str(uuid.uuid4())

    state = SessionState(
        session_id   = session_id,
        session_name = session_name,
        speakers     = unique_speakers,
        sample_rate  = sample_rate,
    )

    with _registry_lock:
        _sessions[session_id] = state

    state.live_thread = threading.Thread(target=state.run, daemon=True)
    state.live_thread.start()

    logger.info("Started session %s (speakers=%s)", session_id, unique_speakers)

    return {
        'session_id':   session_id,      # ← frontend MUST store this
        'running':      True,
        'session_name': session_name,
        'speakers':     unique_speakers,
        'sample_rate':  sample_rate,
    }


def _get_session(session_id: str) -> SessionState:
    """Retrieve a session or raise a clear error."""
    with _registry_lock:
        state = _sessions.get(session_id)
    if state is None:
        raise RuntimeError(f"No active session with id '{session_id}'")
    return state


def set_active_speaker(session_id: str, speaker_name: str) -> dict[str, Any]:
    return _get_session(session_id).set_active_speaker(speaker_name)


def push_audio_chunk(session_id: str, chunk: bytes) -> None:
    if not chunk:
        return
    state = _get_session(session_id)
    with state._live_lock:
        if not state.live_state['running']:
            raise RuntimeError('Session is not running')
        state.live_state['audio_chunks_received'] += 1
    state.audio_queue.put(chunk)


def stop_live_session(session_id: str) -> dict[str, Any]:
    state = _get_session(session_id)

    with state._live_lock:
        if not state.live_state['running']:
            raise RuntimeError('Session is not running')

    state.stream_stop_event.set()
    state.audio_queue.put(None)
    state.final_result_event.wait(timeout=15)

    with state._live_lock:
        if state.live_state['final_result'] is None:
            state.live_state['final_result'] = state.build_final_result(0.0)
        return {
            'stopping':     True,
            'session_name': state.live_state['session_name'],
            'session_id':   state.live_state['session_id'],
            'final_result': state.live_state['final_result'],
        }


def live_status(session_id: str) -> dict[str, Any]:
    state = _get_session(session_id)
    with state._live_lock:
        ls = state.live_state
        return {
            'running':               ls['running'],
            'last_error':            ls['last_error'],
            'session_name':          ls['session_name'],
            'speakers':              ls['speakers'],
            'current_speaker':       ls['current_speaker'],
            'sample_rate':           ls['sample_rate'],
            'session_id':            ls['session_id'],
            'final_result':          ls['final_result'],
            'live_updates_count':    len(ls['live_updates']),
            'audio_chunks_received': ls['audio_chunks_received'],
        }


def live_updates(session_id: str, *, since_index: int = 0) -> dict[str, Any]:
    state = _get_session(session_id)
    with state._live_lock:
        ls = state.live_state

        # Session-init timeout check
        started_at = ls.get('started_at')
        if ls['running'] and ls['session_id'] is None and started_at:
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(started_at)
            if (
                elapsed.total_seconds() > SESSION_INIT_TIMEOUT_SECONDS
                and ls.get('audio_chunks_received', 0) == 0
                and not ls['last_error']
            ):
                ls['last_error'] = (
                    'Realtime session started but no audio chunks are arriving from the browser. '
                    'Check WebSocket connection and microphone permissions.'
                )

        updates    = ls['live_updates'][since_index:]
        next_index = since_index + len(updates)

        return {
            'running':               ls['running'],
            'last_error':            ls['last_error'],
            'session_name':          ls['session_name'],
            'speakers':              ls['speakers'],
            'current_speaker':       ls['current_speaker'],
            'session_id':            ls['session_id'],
            'updates':               updates,
            'next_index':            next_index,
            'audio_chunks_received': ls['audio_chunks_received'],
            'final_result':          ls['final_result'],
        }


def list_active_sessions() -> list[dict[str, Any]]:
    """Utility — returns summary of all currently running sessions."""
    with _registry_lock:
        snapshot = list(_sessions.items())
    result = []
    for sid, state in snapshot:
        with state._live_lock:
            result.append({
                'session_id':   sid,
                'session_name': state.live_state['session_name'],
                'speakers':     state.live_state['speakers'],
                'running':      state.live_state['running'],
                'started_at':   state.live_state['started_at'],
            })
    return result