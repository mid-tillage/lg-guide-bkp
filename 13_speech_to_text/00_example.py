import os
import dotenv
from langchain_google_community import GoogleSpeechToTextLoader

dotenv.load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
file_path = os.getenv("FILE_PATH")
# "gs://cloud-samples-data/speech/audio.flac"
# or a local file path: file_path = "./audio.wav"

loader = GoogleSpeechToTextLoader(project_id=project_id, file_path=file_path)

docs = loader.load()
print(docs[0].page_content)
print(docs[0].metadata)
