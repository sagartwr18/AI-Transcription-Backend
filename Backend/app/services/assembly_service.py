import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import assemblyai as aai

from app.core.config import settings

_PENDING_LOCK = Lock()
_PENDING_TRANSCRIPTS: dict[str, dict[str, Any]] = {}


def _normalize_label(label: Any) -> str:
    if label is None:
        return 'Unknown'
    return str(label)


def _build_speaker_samples(speaker_segments: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for segment in speaker_segments:
        grouped[segment['speaker_label']].append(segment['text'])

    return [
        {
            'speaker_label': speaker_label,
            'samples': [
                text[:120] + ('...' if len(text) > 120 else '')
                for text in texts[:2]
            ],
        }
        for speaker_label, texts in sorted(grouped.items())
    ]


def _build_final_output(session_data: dict[str, Any], speaker_map: dict[str, str]) -> dict[str, Any]:
    speaker_segments = session_data['speaker_segments']
    labels = sorted({seg['speaker_label'] for seg in speaker_segments})

    # Build label → real name mapping
    # e.g. {"A": "Alice", "B": "Joe"} — fallback to "Speaker A" if not provided
    normalized_map = {
        label: (speaker_map.get(label) or f'Speaker {label}')
        for label in labels
    }

    # ── per_speaker_text — all text merged per real name ──────
    per_speaker_raw: dict[str, list[str]] = defaultdict(list)
    for seg in speaker_segments:
        per_speaker_raw[seg['speaker_label']].append(seg['text'])

    per_speaker_text = {
        normalized_map.get(label, f'Speaker {label}'): ' '.join(texts)
        for label, texts in sorted(per_speaker_raw.items())
    }

    named_segments = [
        {
            'speaker':       normalized_map.get(seg['speaker_label'], f"Speaker {seg['speaker_label']}"),
            'speaker_label': seg['speaker_label'],
            'text':          seg['text'],
            # 'start_ms':      seg.get('start_ms'),
            # 'end_ms':        seg.get('end_ms'),
            # 'confidence':    seg.get('confidence'),
        }
        for seg in speaker_segments
    ]

    metadata = dict(session_data['metadata'])
    metadata['speaker_map'] = normalized_map
    metadata['saved_at'] = datetime.utcnow().isoformat() + 'Z'

    return {
        'metadata':        metadata,
        'full_text':       session_data['full_text'],
        # 'confidence':      session_data['confidence'],
        'per_speaker_text': per_speaker_text,
        'speaker_segments': named_segments,
    }


def _transcribe_audio_sync(source_file: str, api_key: str, source_name: str | None = None) -> dict[str, Any]:
    aai.settings.base_url = 'https://api.assemblyai.com'
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        speech_models=['universal-3-pro', 'universal-2'],
        language_detection=True,
        speaker_labels=True,
    )

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(source_file, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f'Transcription failed: {transcript.error}')

    # Build speaker_segments from AssemblyAI utterances
    # Each utterance = one continuous block of speech from a single speaker
    speaker_segments = [
        {
            'speaker_label': _normalize_label(u.speaker),
            'text':          u.text,
            'start_ms':      u.start,       # audio offset in milliseconds
            'end_ms':        u.end,         # audio offset in milliseconds
            'confidence':    u.confidence,
        }
        for u in (transcript.utterances or [])
    ]

    session_data: dict[str, Any] = {
        'metadata': {
            'transcript_id':          transcript.id,
            'status':                 str(transcript.status),
            'language_code':          transcript.json_response.get('language_code'),
            'audio_duration_seconds': transcript.json_response.get('audio_duration'),
            'speech_models':          ['universal-3-pro', 'universal-2'],
            'source':                 source_name or source_file,
        },
        'full_text':        transcript.text,
        'confidence':       transcript.json_response.get('confidence'),
        'speaker_segments': speaker_segments,   # ← stored as speaker_segments internally too
    }

    # If no speaker segments found → return result immediately, no naming needed
    if not speaker_segments:
        return {
            'requires_speaker_naming': False,
            'result': _build_final_output(session_data, {}),
        }

    # Store session temporarily so finalize_transcription() can retrieve it
    request_id      = uuid4().hex
    speaker_samples = _build_speaker_samples(speaker_segments)

    with _PENDING_LOCK:
        _PENDING_TRANSCRIPTS[request_id] = session_data

    return {
        'requires_speaker_naming': True,
        'request_id':      request_id,
        'speaker_samples': speaker_samples,
    }


async def transcribe_audio(*, source_file: str, source_name: str | None = None) -> dict[str, Any]:
    resolved_api_key = settings.ASSEMBLYAI_API_KEY
    if not resolved_api_key:
        raise ValueError('ASSEMBLYAI_API_KEY is missing in Backend/.env')
    if not Path(source_file).exists():
        raise ValueError(f'Source file not found: {source_file}')

    return await asyncio.to_thread(
        _transcribe_audio_sync,
        source_file,
        resolved_api_key,
        source_name,
    )


def finalize_transcription(*, request_id: str, speaker_map: dict[str, str]) -> dict[str, Any]:
    with _PENDING_LOCK:
        session_data = _PENDING_TRANSCRIPTS.pop(request_id, None)

    if not session_data:
        raise ValueError('Invalid or expired request_id')

    return _build_final_output(session_data, speaker_map)