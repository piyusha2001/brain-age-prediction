import os
import secrets

from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs


load_dotenv()

ELEVENLABS_API_KEY = (
    os.getenv("ELEVEN_LABS_API_KEY")
    or os.getenv("ELEVENLABS_API_KEY")
)

# Bella is a warm, bright, professional female voice available through
# ElevenLabs. Set ELEVENLABS_VOICE_ID to use a different account voice.
FRIDAY_VOICE_ID = os.getenv(
    "ELEVENLABS_VOICE_ID",
    "hpp4J3VqNfWAUOO0d1Us",
)


def generate_friday_audio(text: str) -> bytes:
    """Generate a natural, privacy-conscious Friday narration as MP3."""

    if not ELEVENLABS_API_KEY:
        raise ValueError(
            "ELEVEN_LABS_API_KEY or ELEVENLABS_API_KEY was not found."
        )

    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        raise ValueError("Narration text cannot be empty.")

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio_chunks = client.text_to_speech.convert(
        voice_id=FRIDAY_VOICE_ID,
        text=cleaned_text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        enable_logging=False,
        voice_settings=VoiceSettings(
            stability=0.58,
            similarity_boost=0.78,
            style=0.18,
            use_speaker_boost=True,
            speed=0.96,
        ),
    )

    return b"".join(audio_chunks)


def build_welcome_narration() -> str:
    introductions = [
        "Hi, I'm Friday. Welcome to Synapse A I. I'll make your brain-age results clear and easy to follow.",
        "Welcome to Synapse A I. I'm Friday, your guide to the measurements behind your brain-age estimate.",
        "Hello, I'm Friday. I'm here to explain your Synapse A I results in a calm, simple way.",
        "Hi there. I'm Friday. Let's turn your brain biomarkers into a clear story you can understand.",
        "Welcome. I'm Friday, your Synapse A I guide. Let's explore what shaped your brain-age estimate.",
    ]
    return secrets.choice(introductions)


def build_report_narration(report: dict) -> str:
    patient = report["patient"]
    summary = report["summary"]
    return (
        f"Friday here. Your age is "
        f"{patient['actual_age']:.0f}, and the model estimated a brain age "
        f"of {patient['predicted_age']:.1f}. {summary['key_takeaway']}"
    )


def build_factor_narration(factor: dict) -> str:
    direction = (
        "a lower brain-age estimate"
        if factor["direction"] == "younger"
        else "a higher brain-age estimate"
    )
    evidence_strength = str(factor.get("evidence_strength", "contextual")).lower()
    evidence_context = {
        "strong": "the cited research closely matches this measurement",
        "moderate": "the research is relevant, though it studied different groups or questions",
        "limited": "the research offers broad context for this measurement",
    }.get(
        evidence_strength,
        "the research provides context for interpreting this measurement",
    )
    return (
        f"Friday here. {factor['feature']} moved the model toward "
        f"{direction}. Research support is {evidence_strength}; "
        f"{evidence_context}."
    )
