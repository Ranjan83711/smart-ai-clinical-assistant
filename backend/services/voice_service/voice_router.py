from .speech_to_text import transcribe_audio
from .llm_handler import generate_response
from .text_to_speech import text_to_speech


def handle_voice_query(audio_file):

    # speech → text
    text = transcribe_audio(audio_file)

    # GPT response
    response_text = generate_response(text)

    # text → speech
    audio_path = text_to_speech(response_text)

    return {
        "transcription": text,
        "response": response_text,
        "audio": audio_path
    }