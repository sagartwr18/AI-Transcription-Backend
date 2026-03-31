import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect

from app.schemas.common import ApiResponse
from app.schemas.transcription import AssemblyFinalizeRequest, AutoRealtimeStartRequest, RealtimeStartRequest, RegenerateSummaryRequest, SummaryRequest, SummaryType, SpeakerSwitchRequest
from app.services import assembly_service, auto_diarization_service, real_service
from app.services.summary_service import SummaryService
summary_service = SummaryService()

router = APIRouter()


@router.post('/assembly/transcribe', response_model=ApiResponse)
async def transcribe_with_assembly(audio_file: UploadFile = File(...)) -> ApiResponse:
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail='No audio file selected')

    file_suffix = Path(audio_file.filename).suffix or '.wav'
    temp_file_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            while True:
                chunk = await audio_file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        result = await assembly_service.transcribe_audio(
            source_file=temp_file_path,
            source_name=audio_file.filename,
        )
        if result.get('requires_speaker_naming'):
            return ApiResponse(
                success=True,
                message='Transcription completed. Submit speaker names to finalize.',
                data=result,
            )

        return ApiResponse(success=True, message='Transcription completed', data=result.get('result'))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Transcription failed: {exc}') from exc
    finally:
        await audio_file.close()
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post('/assembly/transcribe/finalize', response_model=ApiResponse)
def finalize_transcription(payload: AssemblyFinalizeRequest) -> ApiResponse:
    try:
        result = assembly_service.finalize_transcription(
            request_id=payload.request_id,
            speaker_map=payload.speaker_map,
        )
        return ApiResponse(success=True, message='Transcription finalized', data=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Finalize failed: {exc}') from exc


@router.post('/real/start-live', response_model=ApiResponse)
def start_live_session(payload: RealtimeStartRequest) -> ApiResponse:
    try:
        state = real_service.start_live_session(
            session_name=payload.session_name,
            # number_of_speakers=payload.number_of_speakers,
            speakers=payload.speakers,
            sample_rate=payload.sample_rate,
        )
        return ApiResponse(
            success=True,
            message='Live session started from internal realtime service',
            data=state,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Live session failed to start: {exc}') from exc


@router.post('/real/auto/start-live', response_model=ApiResponse)
def start_auto_live_session(payload: AutoRealtimeStartRequest) -> ApiResponse:
    try:
        state = auto_diarization_service.start_live_session(
            session_name=payload.session_name,
            number_of_speakers=payload.number_of_speakers,
            sample_rate=payload.sample_rate,
        )
        return ApiResponse(
            success=True,
            message='Automatic diarization live session started',
            data=state,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Automatic diarization session failed to start: {exc}') from exc


@router.get('/real/live-updates', response_model=ApiResponse)
def realtime_live_updates(
    since_index: int = Query(default=0, ge=0),
) -> ApiResponse:
    try:
        return ApiResponse(
            success=True,
            message='Live updates fetched',
            data=real_service.live_updates(since_index=since_index),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Live updates failed: {exc}') from exc


@router.get('/real/auto/live-updates', response_model=ApiResponse)
def auto_realtime_live_updates(
    since_index: int = Query(default=0, ge=0),
) -> ApiResponse:
    try:
        return ApiResponse(
            success=True,
            message='Automatic diarization live updates fetched',
            data=auto_diarization_service.live_updates(since_index=since_index),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Automatic diarization live updates failed: {exc}') from exc


@router.post('/real/stop-live', response_model=ApiResponse)
def stop_live_session() -> ApiResponse:
    try:
        data = real_service.stop_live_session()
        return ApiResponse(
            success=True,
            message='Live session stop requested',
            data=data,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Stop live session failed: {exc}') from exc


@router.post('/real/auto/stop-live', response_model=ApiResponse)
def stop_auto_live_session() -> ApiResponse:
    try:
        data = auto_diarization_service.stop_live_session()
        return ApiResponse(
            success=True,
            message='Automatic diarization live session stop requested',
            data=data,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Automatic diarization stop failed: {exc}') from exc


@router.websocket('/real/audio-stream')
async def realtime_audio_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            chunk = await websocket.receive_bytes()
            real_service.push_audio_chunk(chunk)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        await websocket.close(code=1008)
    except Exception:
        await websocket.close(code=1011)


@router.websocket('/real/auto/audio-stream')
async def auto_realtime_audio_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            chunk = await websocket.receive_bytes()
            auto_diarization_service.push_audio_chunk(chunk)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        await websocket.close(code=1008)
    except Exception:
        await websocket.close(code=1011)
        
@router.post("/live/speaker", response_model=ApiResponse)
def switch_speaker(body: SpeakerSwitchRequest):
    try:
        real_service.set_active_speaker(body.speaker_name)
        return ApiResponse(
            success=True,
            message=f"Active speaker set to {body.speaker_name}",
            data={"active_speaker": body.speaker_name}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post('/summary/create', response_model=ApiResponse)
def create_summary(payload: SummaryRequest) -> ApiResponse:
    try:
        summary = summary_service.create_summary(
            summary_type=payload.summary_type,
            per_speaker_text=payload.per_speaker_text,
            full_text=payload.full_text,
        )

        message = (
            'Speaker summary created'
            if payload.summary_type == SummaryType.speaker_summary
            else 'Conference summary created'
        )

        return ApiResponse(
            success=True,
            message=message,
            data=summary,   # ← service already returns a clean dict, pass directly
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f'AI service error: {exc}') from exc

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Summary creation failed: {exc}') from exc


@router.post('/summary/regenerate', response_model=ApiResponse)
def regenerate_summary(payload: RegenerateSummaryRequest) -> ApiResponse:
    try:
        summary = summary_service.regenerate_summary(
            summary_type=payload.summary_type,
            current_summary=payload.current_summary,
            per_speaker_text=payload.per_speaker_text,
            full_text=payload.full_text,
            improvement_instructions=payload.improvement_instructions,
        )

        message = (
            'Speaker summary regenerated'
            if payload.summary_type == SummaryType.speaker_summary
            else 'Conference summary regenerated'
        )

        return ApiResponse(
            success=True,
            message=message,
            data=summary,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f'AI service error: {exc}') from exc

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Summary regeneration failed: {exc}') from exc
