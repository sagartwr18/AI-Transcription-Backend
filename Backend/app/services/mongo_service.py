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

    def ping(self) -> dict[str, Any]:
        self._client.admin.command('ping')
        return {
            'connected': True,
            'database': settings.MONGODB_DB_NAME,
            'collection': settings.MONGODB_SUMMARIES_COLLECTION,
        }

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
