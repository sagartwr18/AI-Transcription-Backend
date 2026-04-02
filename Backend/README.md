# Backend Template

Structured FastAPI backend with API versioning and service-based architecture.

## Folder Structure

Backend/
- app/
  - api/
    - v1/
      - endpoints/
        - health.py
        - transcription.py
      - router.py
  - core/
    - config.py
  - schemas/
    - common.py
    - transcription.py
  - services/
    - assembly_service.py
    - real_service.py
  - main.py
- requirements.txt
- .env.example

## APIs

- `GET /api/v1/health`
- `POST /api/v1/transcription/assembly/transcribe` (multipart upload: `audio_file`)
- `POST /api/v1/transcription/assembly/transcribe/finalize` (JSON body with `request_id` + `speaker_map`)
- `POST /api/v1/transcription/real/start-live`

For `/api/v1/transcription/assembly/transcribe`, send `form-data` with key:
- `audio_file` (type: file)

If response says `requires_speaker_naming: true`, call `/api/v1/transcription/assembly/transcribe/finalize` with:
```json
{
  "request_id": "returned_request_id",
  "speaker_map": {
    "A": "Speaker One",
    "B": "Speaker Two"
  }
}
```

## Run

```powershell
cd Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Notes

- Assembly batch and realtime logic are implemented inside `app/services`.
- Realtime live API runs in a background thread.
- Summary generation can also be persisted to MongoDB by setting `MONGODB_URI`.
