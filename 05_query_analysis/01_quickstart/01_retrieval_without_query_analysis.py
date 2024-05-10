import dotenv

dotenv.load_dotenv()

from langchain_community.document_loaders import YoutubeLoader

urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    # "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    # "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    # "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    # "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    # "https://www.youtube.com/watch?v=rQdibOsL1ps",
    # "https://www.youtube.com/watch?v=28lC4fqukoc",
    # "https://www.youtube.com/watch?v=es-9MgxB-uc",
    # "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    # "https://www.youtube.com/watch?v=ObIltMaRJvY",
    # "https://www.youtube.com/watch?v=DjuXACWYkkU",
    # "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]
docs = []
for url in urls:
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())

import datetime

# Add some additional metadata: what year the video was published
for doc in docs:
    doc.metadata["publish_year"] = int(
        datetime.datetime.strptime(
            doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y")
    )

import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
chunked_docs = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunked_docs,
    embeddings,
)

search_results = vectorstore.similarity_search("how do I build a RAG agent")
print(search_results[0].metadata["title"])
print(search_results[0].page_content[:500])

search_results = vectorstore.similarity_search("videos on RAG published in 2023") # meeeh... **
# Our first result is from 2024 (despite us asking for videos from 2023)
#  and not very relevant to the input. 
# Since we're just searching against document contents, 
# there's no way for the results to be filtered on any document attributes.
print(search_results[0].metadata["title"])
print(search_results[0].metadata["publish_date"])
print(search_results[0].page_content[:500])

# This is just one failure mode that can arise. 
# Let's now take a look at how a basic form of query analysis can fix it!