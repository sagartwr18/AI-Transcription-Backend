from __future__ import annotations
import json
import os
import re
import time
import logging
from typing import Any

from google import genai
from app.schemas.transcription import SummaryType

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"

API_CALL_DELAY   = int(os.getenv("GEMINI_CALL_DELAY",   "6"))
RETRY_BASE_DELAY = int(os.getenv("GEMINI_RETRY_DELAY", "65"))
MAX_RETRIES      = int(os.getenv("GEMINI_MAX_RETRIES",  "4"))

FULL_SUMMARY_MIN_SPEAKERS = 8
FULL_SUMMARY_MAX_SPEAKERS = 10


# ── Service ────────────────────────────────────────────────────────────────────

class SummaryService:
    """
    Exposes two public methods driven by summary_type from the frontend:
        - generate_per_speaker_summary()  → triggered by summary_type='speaker_summary'
        - generate_full_conference_summary() → triggered by summary_type='conference_summary'

    Both are called through the single entry point create_summary().
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key is required. "
                "Pass api_key= or set the GEMINI_API_KEY environment variable."
            )
        self._client = genai.Client(api_key=key)

    # ── Public entry point ─────────────────────────────────────────────────────

    def create_summary(
        self,
        session_name: str,
        summary_type: SummaryType,
        per_speaker_text: dict[str, str],
        full_text: str,
    ) -> dict:
        """
        Route to the correct summary generator based on summary_type.

        summary_type='speaker_summary'
            → calls generate_per_speaker_summary()
            → requires per_speaker_text

        summary_type='conference_summary'
            → calls generate_full_conference_summary()
            → requires full_text + speaker count between 8–10
        """
        if summary_type == SummaryType.speaker_summary:
            if not per_speaker_text:
                raise ValueError(
                    "per_speaker_text must not be empty when summary_type='speaker_summary'."
                )
            return self.generate_per_speaker_summary(session_name, per_speaker_text)

        if summary_type == SummaryType.conference_summary:
            if not full_text or not full_text.strip():
                raise ValueError(
                    "full_text must not be empty when summary_type='conference_summary'."
                )
            speaker_count = len(per_speaker_text) if per_speaker_text else 0

            return self.generate_full_conference_summary(session_name, full_text, speaker_count)

        raise ValueError(f"Unknown summary_type: {summary_type}")

    def regenerate_summary(
        self,
        session_name: str,
        summary_type: SummaryType,
        current_summary: dict[str, Any],
        per_speaker_text: dict[str, str],
        full_text: str,
        improvement_instructions: str,
    ) -> dict:
        """
        Improve an already generated summary using the existing summary payload as input.
        Optional transcript context can be supplied to improve factual grounding.
        """
        if not current_summary:
            raise ValueError("current_summary must not be empty when regenerating a summary.")

        normalized_summary = self._normalize_regeneration_summary(summary_type, current_summary)

        if summary_type == SummaryType.speaker_summary:
            return self.regenerate_per_speaker_summary(
                session_name= session_name,
                current_summary=normalized_summary,
                per_speaker_text=per_speaker_text,
                improvement_instructions=improvement_instructions,
            )

        if summary_type == SummaryType.conference_summary:
            return self.regenerate_full_conference_summary(
                session_name= session_name,
                current_summary=normalized_summary,
                full_text=full_text,
                speaker_count=len(per_speaker_text) if per_speaker_text else normalized_summary.get("speaker_count", 0),
                improvement_instructions=improvement_instructions,
            )

        raise ValueError(f"Unknown summary_type: {summary_type}")

    # ── Per-speaker summary ────────────────────────────────────────────────────

    def generate_per_speaker_summary(self, session_name: str, per_speaker_text: dict[str, str]) -> dict:
        """
        Generate a structured JSON summary for every speaker.
        Key points count is adaptive:
        - 8 to 10 if the speaker's text is long  (>= 150 words)
        - 4 to 5 if the speaker's text is short  (<  150 words)

        Returns:
        {
            "speaker_count": int,
            "speaker_summaries": [
                {
                    "speaker":            str,
                    "contextual_summary": str,
                    "key_points":         [str, ...],  # 4-5 or 8-10 depending on text length
                },
                ...
            ]
        }
        """
        speaker_count     = len(per_speaker_text)
        speaker_summaries: list[dict] = []

        logger.info("Generating per-speaker summaries for %d speaker(s).", speaker_count)

        for idx, (speaker, text) in enumerate(per_speaker_text.items(), start=1):
            logger.info("[%d/%d] Summarising: %s", idx, speaker_count, speaker)

            word_count  = len(text.split())
            is_long     = word_count >= 150
            min_points  = 8 if is_long else 4
            max_points  = 10 if is_long else 5

            logger.debug(
                "%s has %d words — expecting %d–%d key points.",
                speaker, word_count, min_points, max_points
            )

            parsed      = self._call_gemini_json(self._speaker_prompt(speaker, text))
            key_points  = parsed.get("key_points", [])

            # Guard: trim if over, warn if under
            if len(key_points) > max_points:
                logger.warning(
                    "Trimming key_points for %s from %d to %d.",
                    speaker, len(key_points), max_points
                )
                key_points = key_points[:max_points]
            elif len(key_points) < min_points:
                logger.warning(
                    "Expected %d–%d key_points for %s, got %d.",
                    min_points, max_points, speaker, len(key_points)
                )

            speaker_summaries.append({
                "speaker":            parsed.get("speaker", speaker),
                "contextual_summary": parsed.get("contextual_summary", ""),
                "key_points":         key_points,
            })

            if idx < speaker_count:
                logger.debug("Sleeping %ds (rate-limit guard).", API_CALL_DELAY)
                time.sleep(API_CALL_DELAY)

        return {
            "session_name":      session_name,
            "speaker_count":     speaker_count,
            "speaker_summaries": speaker_summaries,
        }

    # ── Full conference summary ────────────────────────────────────────────────

    def generate_full_conference_summary(self,session_name: str, full_text: str, speaker_count: int) -> dict:
        """
        Generate an executive-level full conference summary.

        Returns:
        {
            "speaker_count": int,
            "main_points":   [str, ...],  # 10-12 attributed bullet points
            "conclusions":   [str, ...]
        }
        """
        logger.info("Generating full conference summary (%d speakers).", speaker_count)

        parsed = self._call_gemini_json(self._full_summary_prompt(full_text, speaker_count))

        main_points = parsed.get("main_points", [])

        if len(main_points) > 12:
            logger.warning("Trimming main_points from %d to 12.", len(main_points))
            main_points = main_points[:12]
        elif len(main_points) < 10:
            logger.warning("Expected 10-12 main_points, got %d.", len(main_points))

        return {
            "session_name":      session_name,
            "main_points":   main_points,
            "conclusion":   parsed.get("conclusions", []),
        }

    def regenerate_per_speaker_summary(
        self,
        session_name: str,
        current_summary: dict[str, Any],
        per_speaker_text: dict[str, str],
        improvement_instructions: str,
    ) -> dict:
        speaker_summaries = current_summary.get("speaker_summaries")
        if not isinstance(speaker_summaries, list) or not speaker_summaries:
            raise ValueError(
                "current_summary for speaker_summary must contain a non-empty 'speaker_summaries' list."
            )

        total_speakers = len(speaker_summaries)
        improved_summaries: list[dict[str, Any]] = []

        logger.info("Regenerating per-speaker summaries for %d speaker(s).", total_speakers)

        for idx, entry in enumerate(speaker_summaries, start=1):
            speaker = str(entry.get("speaker", f"Speaker {idx}"))
            logger.info("[%d/%d] Regenerating summary: %s", idx, total_speakers, speaker)

            parsed = self._call_gemini_json(
                self._speaker_regeneration_prompt(
                    speaker_name=speaker,
                    current_entry=entry,
                    speaker_text=per_speaker_text.get(speaker, ""),
                    improvement_instructions=improvement_instructions,
                )
            )

            source_text = per_speaker_text.get(speaker, "")
            word_count = len(source_text.split()) if source_text else 0
            is_long = word_count >= 150 if source_text else len(entry.get("key_points", [])) >= 8
            min_points = 8 if is_long else 4
            max_points = 10 if is_long else 5
            key_points = parsed.get("key_points", [])

            if len(key_points) > max_points:
                logger.warning(
                    "Trimming regenerated key_points for %s from %d to %d.",
                    speaker, len(key_points), max_points
                )
                key_points = key_points[:max_points]
            elif len(key_points) < min_points:
                logger.warning(
                    "Expected %d-%d regenerated key_points for %s, got %d.",
                    min_points, max_points, speaker, len(key_points)
                )

            improved_summaries.append({
                "speaker": parsed.get("speaker", speaker),
                "contextual_summary": parsed.get("contextual_summary", ""),
                "key_points": key_points,
            })

            if idx < total_speakers:
                logger.debug("Sleeping %ds (rate-limit guard).", API_CALL_DELAY)
                time.sleep(API_CALL_DELAY)

        return {
            "session_name":      session_name,
            "speaker_count": current_summary.get("speaker_count", total_speakers),
            "speaker_summaries": improved_summaries,
        }

    def regenerate_full_conference_summary(
        self,
        session_name: str,
        current_summary: dict[str, Any],
        full_text: str,
        speaker_count: int,
        improvement_instructions: str,
    ) -> dict:
        if not current_summary.get("main_points"):
            raise ValueError(
                "current_summary for conference_summary must contain a non-empty 'main_points' list."
            )

        logger.info("Regenerating full conference summary (%d speakers).", speaker_count)

        parsed = self._call_gemini_json(
            self._full_summary_regeneration_prompt(
                current_summary=current_summary,
                full_text=full_text,
                speaker_count=speaker_count,
                improvement_instructions=improvement_instructions,
            )
        )

        main_points = parsed.get("main_points", [])
        if len(main_points) > 12:
            logger.warning("Trimming regenerated main_points from %d to 12.", len(main_points))
            main_points = main_points[:12]
        elif len(main_points) < 10:
            logger.warning("Expected 10-12 regenerated main_points, got %d.", len(main_points))

        return {
            "session_name":      session_name,
            # "speaker_count": speaker_count,
            "main_points": main_points,
            "conclusion": parsed.get("conclusions", []),
        }

    @staticmethod
    def _normalize_regeneration_summary(
        summary_type: SummaryType,
        current_summary: Any,
    ) -> dict[str, Any]:
        if summary_type == SummaryType.speaker_summary:
            if isinstance(current_summary, list):
                return {
                    "speaker_count": len(current_summary),
                    "speaker_summaries": current_summary,
                }

            if not isinstance(current_summary, dict):
                raise ValueError(
                    "For speaker_summary regeneration, current_summary must be a list of speaker summaries "
                    "or an object containing speaker summary entries."
                )

            if isinstance(current_summary.get("speaker_summaries"), list):
                return current_summary

            if all(isinstance(value, dict) for value in current_summary.values()):
                speaker_summaries = list(current_summary.values())
                return {
                    "speaker_count": len(speaker_summaries),
                    "speaker_summaries": speaker_summaries,
                }

            raise ValueError(
                "For speaker_summary regeneration, current_summary must contain 'speaker_summaries' "
                "or be an object whose values are speaker summary entries."
            )

        if summary_type == SummaryType.conference_summary:
            if not isinstance(current_summary, dict):
                raise ValueError(
                    "For conference_summary regeneration, current_summary must be an object containing 'main_points'."
                )

            if "main_points" not in current_summary:
                raise ValueError(
                    "For conference_summary regeneration, current_summary must contain 'main_points'."
                )

            normalized = dict(current_summary)
            if "conclusions" not in normalized and "conclusion" in normalized:
                normalized["conclusions"] = normalized.get("conclusion", [])
            return normalized

        return current_summary
        
    # ── Prompt: per-speaker (improved) ────────────────────────────────────────

    @staticmethod
    def _speaker_prompt(speaker_name: str, speaker_text: str) -> str:
        # Determine key_points count based on speaker text length
        word_count = len(speaker_text.split())
        if word_count >= 150:
            key_points_instruction = "8 to 10"
            key_points_range       = "8–10"
        else:
            key_points_instruction = "4 to 5"
            key_points_range       = "4–5"

        return (
            "You are an expert Conference Rapporteur and Knowledge Synthesizer.\n\n"
            "Task: Analyze the provided transcript segment from a collaborative discussion. "
            "Extract the intellectual substance of the speaker's contribution while ignoring "
            "conversational filler (e.g., 'um,' 'can you hear me,' 'thanks for having me').\n\n"
            "Instructions:\n"
            "1. Identify Speaker: Use the exact label provided in the transcript.\n"
            "2. Contextual Summary: Write a 20-word summary of the speaker's contribution. Do not just list topics; "
            "explain how their contribution moved the discussion forward or what unique "
            "perspective they offered.\n"
            f"3. High-Signal Key Points: Extract exactly {key_points_instruction} bullet points "
            "representing specific data, suggestions, or insights. "
            "Avoid vague points like 'He talked about technology.' "
            "Instead use specific points like 'Proposed a 3-step framework for AI integration.' "
            f"CRITICAL: The 'key_points' array MUST contain exactly {key_points_instruction} items. "
            f"No fewer than {key_points_range.split('–')[0]}, "
            f"no more than {key_points_range.split('–')[1]}.\n"
            "4. Interaction Check: If the speaker is responding directly to a previous point, "
            "briefly note that connection (e.g., 'Building on Speaker A point about cost...'). "
            "If there is no direct interaction, set this field to null.\n\n"
            "CRITICAL: You MUST respond with ONLY a valid JSON object. "
            "No markdown, no code fences, no explanation — raw JSON only.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "speaker": "<exact speaker name>",\n'
            '  "contextual_summary": "<20-word summary>",\n'
            '  "key_points": [\n'
            f'    // Must contain exactly {key_points_instruction} bullet strings\n'
            '    "• <specific insight or data point>",\n'
            '    "• <specific insight or data point>"\n'
            "  ],\n"
            '  "interaction_note": "<connection to previous speaker or null>"\n'
            "}\n\n"
            f"Speaker: {speaker_name}\n"
            "---\n"
            f"{speaker_text.strip()}\n"
            "---\n\n"
            "JSON response:"
        )

    # ── Prompt: full conference (improved) ────────────────────────────────────

    @staticmethod
    def _full_summary_prompt(full_text: str, speaker_count: int) -> str:
        return (
            "You are an expert Knowledge Synthesizer and Executive Assistant. "
            "Your goal is to distill a multi-speaker conference transcript into a "
            "high-level executive summary that captures the collective intelligence "
            "of the session.\n\n"
            f"This transcript contains {speaker_count} speakers.\n\n"
            "Instructions:\n"
            "1. Filter the Noise: Ignore administrative chatter, technical issues, and "
            "introductory pleasantries. Focus only on the core intellectual exchange.\n"
            "2. Extract Main Points: Identify the most important and insightful points "
            "discussed across the entire conference. Each point should be a single, "
            "clear, and concise sentence that captures a key idea, finding, or argument.\n"
            "   IMPORTANT: You MUST generate between 10 and 12 distinct points in "
            "'main_points'. Cover a diverse range of topics discussed — do not repeat "
            "or rephrase similar points.\n"
            "3. Attribute Insights: Each point in 'main_points' must be attributed to "
            "the specific speaker who raised it "
            "(e.g., 'Speaker A highlighted that...', 'Dr. Chen argued that...').\n"
            "4. Identify Outcomes: Strictly distinguish between:\n"
            "   - 'suggestions': something an individual speaker mentioned or proposed.\n"
            "   - 'conclusions': something the group collectively agreed upon or a clear "
            "consensus that emerged.\n"
            "   - 'action_items': concrete next steps, follow-ups, or tasks identified.\n\n"
            "CRITICAL: You MUST respond with ONLY a valid JSON object. "
            "No markdown, no code fences, no explanation — raw JSON only.\n"
            "CRITICAL: The 'main_points' array MUST contain exactly 10 to 12 strings. "
            "No fewer than 10, no more than 12.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "main_points": [\n'
            '    "highlighted that ...",\n'
            '    "argued that ...",\n'
            '    "emphasized that ..."\n'
            "    // Repeat for a total of 10 to 12 point strings\n"
            "  ],\n"
            '  "conclusions": [\n'
            '    "<A point the group collectively agreed upon>"\n'
            "  ]\n"
            "}\n\n"
            "Full Conference Transcript:\n"
            "---\n"
            f"{full_text.strip()}\n"
            "---\n\n"
            "JSON response:"
        )

    @staticmethod
    def _speaker_regeneration_prompt(
        speaker_name: str,
        current_entry: dict[str, Any],
        speaker_text: str,
        improvement_instructions: str,
    ) -> str:
        word_count = len(speaker_text.split()) if speaker_text else 0
        existing_key_points = current_entry.get("key_points", [])
        is_long = word_count >= 150 if speaker_text else len(existing_key_points) >= 8
        key_points_instruction = "8 to 10" if is_long else "4 to 5"
        key_points_range = "8-10" if is_long else "4-5"

        transcript_section = (
            f"Original transcript for this speaker:\n---\n{speaker_text.strip()}\n---\n\n"
            if speaker_text and speaker_text.strip()
            else "Original transcript for this speaker: Not provided. Improve only from the current summary.\n\n"
        )

        return (
            "You are an expert Conference Rapporteur and Summary Editor.\n\n"
            "Task: Improve an existing per-speaker summary so it becomes clearer, more precise, and more useful "
            "without inventing facts or changing the original meaning.\n\n"
            f"Improvement instructions: {improvement_instructions.strip() or 'Improve clarity, accuracy, and usefulness while preserving meaning.'}\n\n"
            "Rules:\n"
            "1. Preserve factual fidelity. Do not add claims that are not supported by the transcript or current summary.\n"
            "2. Keep the exact speaker label.\n"
            "3. Rewrite a 20-word summary of the speaker's contribution.\n"
            f"4. Return exactly {key_points_instruction} high-signal key points. No fewer than {key_points_range.split('-')[0]}, no more than {key_points_range.split('-')[1]}.\n"
            "5. Prefer concrete and specific wording over generic phrases.\n\n"
            "CRITICAL: You MUST respond with ONLY a valid JSON object.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "speaker": "<exact speaker name>",\n'
            '  "contextual_summary": "<20-word improved summary>",\n'
            '  "key_points": [\n'
            '    "<specific improved point>",\n'
            '    "<specific improved point>"\n'
            "  ]\n"
            "}\n\n"
            "Current summary entry to improve:\n"
            f"{json.dumps(current_entry, ensure_ascii=True, indent=2)}\n\n"
            f"{transcript_section}"
            "JSON response:"
        )

    @staticmethod
    def _full_summary_regeneration_prompt(
        current_summary: dict[str, Any],
        full_text: str,
        speaker_count: int,
        improvement_instructions: str,
    ) -> str:
        transcript_section = (
            f"Original full conference transcript:\n---\n{full_text.strip()}\n---\n\n"
            if full_text and full_text.strip()
            else "Original full conference transcript: Not provided. Improve only from the current summary.\n\n"
        )

        return (
            "You are an expert Knowledge Synthesizer and Executive Summary Editor.\n\n"
            "Task: Improve an existing full conference summary so it is clearer, more concise, better organized, "
            "and more actionable without inventing facts or changing the meaning.\n\n"
            f"This conference includes approximately {speaker_count} speakers.\n"
            f"Improvement instructions: {improvement_instructions.strip() or 'Improve clarity, accuracy, and usefulness while preserving meaning.'}\n\n"
            "Rules:\n"
            "1. Preserve factual accuracy and speaker attribution.\n"
            "2. Return 10 to 12 distinct main points.\n"
            "3. Keep the points diverse and non-repetitive.\n"
            "4. Keep conclusions limited to actual consensus or clear outcomes.\n\n"
            "CRITICAL: You MUST respond with ONLY a valid JSON object.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "main_points": [\n'
            '    "<improved attributed point>",\n'
            '    "<improved attributed point>"\n'
            "  ],\n"
            '  "conclusions": [\n'
            '    "<improved group conclusion>"\n'
            "  ]\n"
            "}\n\n"
            "Current conference summary to improve:\n"
            f"{json.dumps(current_summary, ensure_ascii=True, indent=2)}\n\n"
            f"{transcript_section}"
            "JSON response:"
        )

    # ── Gemini call with JSON parsing + 429 retry ──────────────────────────────

    def _call_gemini_json(self, prompt: str, retries: int = MAX_RETRIES) -> dict:
        for attempt in range(1, retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                raw = response.text.strip()

                # Strip accidental markdown code fences Gemini sometimes adds
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw.strip())

                return json.loads(raw)

            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed on attempt %d: %s", attempt, exc)
                if attempt == retries:
                    raise RuntimeError(
                        f"Gemini returned invalid JSON after {retries} attempts."
                    ) from exc
                time.sleep(API_CALL_DELAY * attempt)

            except Exception as exc:
                err    = str(exc)
                is_429 = "429" in err or "RESOURCE_EXHAUSTED" in err

                if is_429:
                    wait = RETRY_BASE_DELAY + (attempt * 10)
                    logger.warning(
                        "Rate limit (429). Waiting %ds, retry %d/%d.",
                        wait, attempt, retries - 1,
                    )
                    time.sleep(wait)
                else:
                    logger.warning("Gemini attempt %d/%d failed: %s", attempt, retries, exc)
                    if attempt < retries:
                        time.sleep(API_CALL_DELAY * attempt)

                if attempt == retries:
                    raise RuntimeError(
                        f"Gemini API call failed after {retries} attempts: {exc}"
                    ) from exc

        raise RuntimeError("Gemini API call failed unexpectedly.")
