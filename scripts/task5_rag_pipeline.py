import os
import requests
from typing import List

from task4_translate import translate_to_english
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

ASR_URL = "http://127.0.0.1:8000/transcribe"
VECTOR_DB_PATH = "vector_db/faiss_index"

#  MUST be same as Task-2 embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# ASR: Call IndicConformer FastAPI
# --------------------------------------------------

def transcribe_audio(audio_file_path: str) -> str:
    """
    Send audio file to IndicConformer ASR FastAPI endpoint
    """

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    with open(audio_file_path, "rb") as audio_file:
        files = {
            "file": (
                os.path.basename(audio_file_path),
                audio_file,
                "audio/wav"
            )
        }
        response = requests.post(ASR_URL, files=files)

    if response.status_code != 200:
        raise RuntimeError(
            f"ASR API failed {response.status_code}: {response.text}"
        )

    return response.json()["transcription"]


# --------------------------------------------------
# Vector DB Retrieval
# --------------------------------------------------

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 2
) -> List[str]:
    """
    Retrieve top-k relevant chunks from FAISS vector DB
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(query, k=top_k)

    return [doc.page_content for doc in docs]


# --------------------------------------------------
# LLM: GitHub Models
# --------------------------------------------------

def generate_answer_from_llm(
    question: str,
    context_chunks: List[str]
) -> str:
    """
    Generate answer using GitHub Models (OpenAI compatible API)
    """

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN not found in environment")

    api_url = "https://models.inference.ai.azure.com/chat/completions"

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json"
    }

    context = "\n\n".join(context_chunks)

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "Answer strictly using the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        "temperature": 0.3
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM API failed {response.status_code}: {response.text}"
        )

    return response.json()["choices"][0]["message"]["content"]


# --------------------------------------------------
# MAIN RAG PIPELINE
# --------------------------------------------------

def run_rag_pipeline(audio_file_path: str) -> str:
    """
    Audio → ASR → Translation → Retrieval → LLM → Answer
    """

    # Step 1: ASR
    hindi_text = transcribe_audio(audio_file_path)
    print("\nASR Output:\n", hindi_text)

    # Step 2: Translate
    english_question = translate_to_english(hindi_text)
    print("\nTranslated Question:\n", english_question)

    # Step 3: Retrieve context
    context_chunks = retrieve_relevant_chunks(english_question)

    # Step 4: Generate answer
    final_answer = generate_answer_from_llm(
        english_question,
        context_chunks
    )

    return final_answer


# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":
    audio_path = "sample.wav"

    answer = run_rag_pipeline(audio_path)

    print("\nFinal Answer:\n")
    print(answer)
