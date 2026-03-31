from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

class AssemblyFinalizeRequest(BaseModel):
    request_id: str = Field(..., description='Request ID returned by /assembly/transcribe')
    speaker_map: dict[str, str] = Field(
        default_factory=dict,
        description='Mapping of speaker labels to speaker names, e.g. {"A":"Rahul","B":"Priya"}',
    )

class SpeakerSwitchRequest(BaseModel):
    speaker_name: str

# class RealtimeStartRequest(BaseModel):
#     session_name: str = Field(..., description='Name of the realtime transcription session')
#     number_of_speakers: int = Field(
#         ...,
#         ge=1,
#         description='Expected speaker count. Backend will use this value + 10 for max_speakers.',
#     )
#     sample_rate: int = Field(
#         default=48000,
#         ge=8000,
#         le=96000,
#         description='Incoming browser microphone sample rate.',
#     )

class RealtimeStartRequest(BaseModel):
    session_name: str = Field(..., description='Name of the realtime transcription session')
    speakers: list[str] = Field(                          # ✅ was: number_of_speakers: int
        ...,
        description='List of speaker names e.g. ["Sagar", "Pratham", "Riya"]',
    )
    sample_rate: int = Field(
        default=48000,
        ge=8000,
        le=96000,
        description='Incoming browser microphone sample rate.',
    )


class AutoRealtimeStartRequest(BaseModel):
    session_name: str = Field(..., description='Name of the automatic diarization realtime session')
    number_of_speakers: int = Field(
        ...,
        ge=1,
        description='Estimated number of speakers expected in the session.',
    )
    sample_rate: int = Field(
        default=48000,
        ge=8000,
        le=96000,
        description='Incoming browser microphone sample rate.',
    )

class SummaryType(str, Enum):
    speaker_summary    = "speaker_summary"
    conference_summary = "conference_summary"

class SummaryRequest(BaseModel):
    summary_type: SummaryType = Field(        # ← add this field
        ...,
        description="'speaker_summary' → per-speaker summaries. 'conference_summary' → full executive summary.",
        example="speaker_summary",
    )
    
    per_speaker_text: dict[str, str] = Field(
        default_factory=dict,
        description='Per speaker transcript text map.',
    )
    full_text: str = Field(
        default='',
        description='Optional full transcript text.',
    )


class RegenerateSummaryRequest(BaseModel):
    summary_type: SummaryType = Field(
        ...,
        description="'speaker_summary' → improve per-speaker summaries. 'conference_summary' → improve full executive summary.",
        example="conference_summary",
    )
    current_summary: Any = Field(
        ...,
        description='Previously generated summary payload to improve.',
    )
    per_speaker_text: dict[str, str] = Field(
        default_factory=dict,
        description='Optional per speaker transcript text map. Recommended for speaker summary regeneration.',
    )
    full_text: str = Field(
        default='',
        description='Optional full transcript text. Recommended for conference summary regeneration.',
    )
    improvement_instructions: str = Field(
        default='Improve clarity, accuracy, and usefulness while preserving the original meaning.',
        description='Optional guidance describing how the regenerated summary should be improved.',
    )
