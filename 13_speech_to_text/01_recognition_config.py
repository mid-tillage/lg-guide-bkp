import os
import dotenv
from langchain_google_community import GoogleSpeechToTextLoader

dotenv.load_dotenv()

from google.cloud.speech_v2 import (
    AutoDetectDecodingConfig,
    RecognitionConfig,
    RecognitionFeatures,
)

config = RecognitionConfig(
    auto_decoding_config=AutoDetectDecodingConfig(),
    language_codes=["en-US"],
    model="long",
    features=RecognitionFeatures(
        enable_automatic_punctuation=False,
        profanity_filter=True,
        enable_spoken_punctuation=True,
        enable_spoken_emojis=True,
    ),
)

from langchain_google_community import GoogleSpeechToTextLoader

project_id = os.getenv("GCP_PROJECT_ID")
location = "global"
recognizer_id = os.getenv("RECOGNIZER_ID")
file_path = os.getenv("FILE_PATH")

loader = GoogleSpeechToTextLoader(
    project_id=project_id,
    location=location,
    recognizer_id=recognizer_id,
    file_path=file_path,
    config=config,
)
