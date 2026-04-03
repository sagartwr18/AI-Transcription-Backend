from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import MongoClient

from app.core.config import settings
from app.schemas.transcription import SummaryType


class SummaryMongoService:
    def __init__(self) -> None:
        if not settings.MONGODB_URI:
            raise ValueError('MONGODB_URI is required to persist summaries.')

        self._client = MongoClient(settings.MONGODB_URI)
        self._collection = self._client[settings.MONGODB_DB_NAME][settings.MONGODB_SUMMARIES_COLLECTION]
        self._transcripts_collection = self._client[settings.MONGODB_DB_NAME][settings.MONGODB_TRANSCRIPTS_COLLECTION]

    def ping(self) -> dict[str, Any]:
        self._client.admin.command('ping')
        return {
            'connected': True,
            'database': settings.MONGODB_DB_NAME,
            'collection': settings.MONGODB_SUMMARIES_COLLECTION,
        }

    def save_transcript(
        self,
        *,
        tracking_session_id: str,
        provider_session_id: str | None,
        session_name: str,
        transcript_data: dict[str, Any],
        source_type: str,
    ) -> str:
        now = datetime.now(timezone.utc)
        set_document = {
            'tracking_session_id': tracking_session_id,
            'provider_session_id': provider_session_id,
            'session_name': session_name,
            'source_type': source_type,
            'transcript_data': transcript_data,
            'updated_at': now,
        }

        result = self._transcripts_collection.update_one(
            {'tracking_session_id': tracking_session_id, 'source_type': source_type},
            {'$set': set_document, '$setOnInsert': {'created_at': now}},
            upsert=True,
        )

        if result.upserted_id is not None:
            return str(result.upserted_id)

        existing = self._transcripts_collection.find_one(
            {'tracking_session_id': tracking_session_id, 'source_type': source_type},
            {'_id': 1},
        )
        if existing is None:
            raise RuntimeError('Transcript saved but could not resolve record id.')
        return str(existing['_id'])

    @staticmethod
    def _merge_transcript_data(
        base_transcript_data: dict[str, Any],
        incoming_transcript_data: dict[str, Any],
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(base_transcript_data)

        base_full_text = str(base_transcript_data.get('full_text') or '').strip()
        incoming_full_text = str(incoming_transcript_data.get('full_text') or '').strip()
        merged['full_text'] = ' '.join(part for part in [base_full_text, incoming_full_text] if part).strip()

        base_per_speaker = base_transcript_data.get('per_speaker_text') or {}
        incoming_per_speaker = incoming_transcript_data.get('per_speaker_text') or {}
        merged_per_speaker: dict[str, str] = {
            str(speaker): str(text or '').strip()
            for speaker, text in base_per_speaker.items()
        }
        for speaker, text in incoming_per_speaker.items():
            speaker_name = str(speaker)
            new_text = str(text or '').strip()
            existing_text = str(merged_per_speaker.get(speaker_name, '') or '').strip()
            if new_text:
                merged_per_speaker[speaker_name] = f"{existing_text} {new_text}".strip()
            elif speaker_name not in merged_per_speaker:
                # Preserve speaker identity even when transcript text is empty.
                merged_per_speaker[speaker_name] = ''
        merged['per_speaker_text'] = merged_per_speaker

        base_segments = list(base_transcript_data.get('speaker_segments') or [])
        incoming_segments = list(incoming_transcript_data.get('speaker_segments') or [])
        merged['speaker_segments'] = [*base_segments, *incoming_segments]

        base_metadata = dict(base_transcript_data.get('metadata') or {})
        incoming_metadata = dict(incoming_transcript_data.get('metadata') or {})

        merged_metadata = dict(base_metadata)
        merged_metadata['merged_at'] = datetime.now(timezone.utc).isoformat()
        merged_metadata['appended_from_transcript_ids'] = [
            tid
            for tid in dict.fromkeys(
                [
                    *(base_metadata.get('appended_from_transcript_ids') or []),
                    incoming_metadata.get('transcript_id'),
                ]
            )
            if tid
        ]

        try:
            base_duration = float(base_metadata.get('audio_duration_seconds') or 0.0)
            incoming_duration = float(incoming_metadata.get('audio_duration_seconds') or 0.0)
            merged_metadata['audio_duration_seconds'] = base_duration + incoming_duration
        except (TypeError, ValueError):
            pass

        base_models = base_metadata.get('speech_models') or []
        incoming_models = incoming_metadata.get('speech_models') or []
        merged_metadata['speech_models'] = list(dict.fromkeys([*base_models, *incoming_models]))

        base_speakers = [str(s) for s in (base_metadata.get('speakers') or []) if s is not None]
        incoming_speakers = [str(s) for s in (incoming_metadata.get('speakers') or []) if s is not None]
        segment_speakers = [
            str(seg.get('speaker'))
            for seg in merged.get('speaker_segments', [])
            if isinstance(seg, dict) and seg.get('speaker') is not None
        ]
        per_speaker_names = [str(name) for name in merged_per_speaker.keys()]
        merged_metadata['speakers'] = list(
            dict.fromkeys([*base_speakers, *incoming_speakers, *per_speaker_names, *segment_speakers])
        )

        merged['metadata'] = merged_metadata
        return merged

    def list_transcripts(self) -> list[dict[str, Any]]:
        cursor = self._transcripts_collection.find(
            {},
            {
                'tracking_session_id': 1,
                'provider_session_id': 1,
                'session_name': 1,
                'source_type': 1,
                'created_at': 1,
                'updated_at': 1,
            },
        ).sort('created_at', -1)

        results: list[dict[str, Any]] = []
        for doc in cursor:
            created = doc.get('created_at')
            updated = doc.get('updated_at')
            results.append({
                'transcript_record_id': str(doc.get('_id')),
                'tracking_session_id': doc.get('tracking_session_id'),
                'provider_session_id': doc.get('provider_session_id'),
                'session_name': doc.get('session_name', 'Untitled'),
                'source_type': doc.get('source_type'),
                'created_at': created.isoformat() if created else None,
                'updated_at': updated.isoformat() if updated else None,
            })
        return results

    def get_transcript(self, transcript_record_id: str) -> dict[str, Any] | None:
        document = self._transcripts_collection.find_one({'_id': ObjectId(transcript_record_id)})
        if document is None:
            return None

        return {
            'transcript_record_id': str(document.get('_id')),
            'tracking_session_id': document.get('tracking_session_id'),
            'provider_session_id': document.get('provider_session_id'),
            'session_name': document.get('session_name', 'Untitled'),
            'source_type': document.get('source_type'),
            'transcript_data': document.get('transcript_data', {}),
            'created_at': document.get('created_at').isoformat() if document.get('created_at') else None,
            'updated_at': document.get('updated_at').isoformat() if document.get('updated_at') else None,
        }

    def append_transcript_to_existing(
        self,
        *,
        target_transcript_record_id: str,
        source_transcript_record_id: str,
        session_name: str,
    ) -> dict[str, Any]:
        if target_transcript_record_id == source_transcript_record_id:
            raise ValueError('Source and target transcript ids must be different.')

        target = self._transcripts_collection.find_one({'_id': ObjectId(target_transcript_record_id)})
        if target is None:
            raise ValueError('Target transcript record not found.')

        source = self._transcripts_collection.find_one({'_id': ObjectId(source_transcript_record_id)})
        if source is None:
            raise ValueError('Source transcript record not found.')

        merged_transcript_data = self._merge_transcript_data(
            base_transcript_data=target.get('transcript_data') or {},
            incoming_transcript_data=source.get('transcript_data') or {},
        )

        now = datetime.now(timezone.utc)
        self._transcripts_collection.update_one(
            {'_id': target['_id']},
            {
                '$set': {
                    'session_name': session_name.strip(),
                    'transcript_data': merged_transcript_data,
                    'updated_at': now,
                },
                '$push': {
                    'append_history': {
                        'source_transcript_record_id': str(source.get('_id')),
                        'source_tracking_session_id': source.get('tracking_session_id'),
                        'appended_at': now,
                    },
                },
            },
        )

        refreshed = self.get_transcript(str(target.get('_id')))
        if refreshed is None:
            raise RuntimeError('Merged transcript was updated but could not be reloaded.')
        return refreshed

    def save_summary(
        self,
        *,
        summary_type: SummaryType,
        summary_data: dict[str, Any],
        per_speaker_text: dict[str, str],
        full_text: str,
    ) -> str:
        document = {
            'summary_type': summary_type.value,
            'summary_data': summary_data,
            'speaker_count': summary_data.get('speaker_count', len(per_speaker_text)),
            'source_metadata': {
                'speaker_labels': list(per_speaker_text.keys()),
                'speaker_transcript_count': len(per_speaker_text),
                'has_full_text': bool(full_text and full_text.strip()),
            },
            'created_at': datetime.now(timezone.utc),
        }

        result = self._collection.insert_one(document)
        return str(result.inserted_id)

    def update_summary(
        self,
        summary_id: str,
        *,
        summary_type: SummaryType,
        summary_data: dict[str, Any],
        per_speaker_text: dict[str, str],
        full_text: str,
    ) -> bool:
        document = {
            'summary_type': summary_type.value,
            'summary_data': summary_data,
            'speaker_count': summary_data.get('speaker_count', len(per_speaker_text)),
            'source_metadata': {
                'speaker_labels': list(per_speaker_text.keys()),
                'speaker_transcript_count': len(per_speaker_text),
                'has_full_text': bool(full_text and full_text.strip()),
            },
            'updated_at': datetime.now(timezone.utc),
        }

        result = self._collection.update_one(
            {'_id': ObjectId(summary_id)},
            {'$set': document},
        )
        return result.matched_count > 0

    def list_summaries(self) -> list[dict[str, Any]]:
        cursor = self._collection.find(
            {},
            {
                'summary_data.session_name': 1,
                'summary_type': 1,
                'created_at': 1,
                'updated_at': 1,
            },
        ).sort('created_at', -1)

        results: list[dict[str, Any]] = []
        for doc in cursor:
            created = doc.get('created_at')
            updated = doc.get('updated_at')
            results.append({
                'summary_record_id': str(doc['_id']),
                'session_name': doc.get('summary_data', {}).get('session_name', 'Untitled'),
                'summary_type': doc.get('summary_type'),
                'created_at': created.isoformat() if created else None,
                'updated_at': updated.isoformat() if updated else None,
            })
        return results

    def get_summary(self, summary_id: str) -> dict[str, Any] | None:
        document = self._collection.find_one({'_id': ObjectId(summary_id)})
        if document is None:
            return None

        return {
            'summary_record_id': str(document['_id']),
            'summary_type': document.get('summary_type'),
            'summary_data': document.get('summary_data', {}),
            'speaker_count': document.get('speaker_count'),
            'source_metadata': document.get('source_metadata', {}),
            'created_at': document.get('created_at').isoformat() if document.get('created_at') else None,
        }
