import os
import requests
from typing import List
from task4_translate import translate_to_english
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# ASR: Call FastAPI endpoint (placeholder)
def transcribe_audio(audio_file_path: str) -> str:
    """
    Send audio file to ASR FastAPI endpoint and return transcription.
    """

    asr_url = "http://127.0.0.1:8000/transcribe"

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

        response = requests.post(asr_url, files=files)

    if response.status_code != 200:
        raise RuntimeError(
            f"ASR API failed with status code {response.status_code}: {response.text}"
        )

    return response.json()["transcription"]



# --------------------------------------------------
# Vector DB: Retrieve top-2 chunks (placeholder)

def retrieve_relevant_chunks(
    query: str,
    vector_db_path: str = "vector_db/faiss_index",
    top_k: int = 2
):
    vector_store = FAISS.load_local(
        vector_db_path,
        embeddings=None,
        allow_dangerous_deserialization=True
    )

    results = vector_store.similarity_search_by_vector(
        vector_store.index.reconstruct_n(0, vector_store.index.ntotal)[0],
        k=top_k
    )

    return [doc.page_content for doc in results]


# LLM: Generate answer using retrieved context (placeholder)
# --------------------------------------------------
def generate_answer_from_llm(
    question: str,
    context_chunks: List[str]
) -> str:
    """
    Generate final answer using GitHub Models (OpenAI-compatible API)
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
                "content": "You are a helpful AI assistant. Answer strictly using the provided context."
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
# Main RAG pipeline
# --------------------------------------------------

def run_rag_pipeline(audio_file_path: str) -> str:
    """
    End-to-end RAG pipeline:
    Audio -> ASR -> Translation -> Retrieval -> LLM -> Answer
    """

    # Step 1: Transcribe audio
    transcription = transcribe_audio(audio_file_path)

    # Step 2: Translate to English
    english_question = translate_to_english(transcription)

    # Step 3: Retrieve relevant chunks
    context_chunks = retrieve_relevant_chunks(english_question)

    # Step 4: Generate answer from LLM
    final_answer = generate_answer_from_llm(
        english_question,
        context_chunks
    )

    return final_answer


# -----------------------------------------------
if __name__ == "__main__":
    audio_path = "sample.wav"

    # Step 1: ASR
    asr_text = transcribe_audio(audio_path)
    print("\nASR Output:\n", asr_text)

    # Step 2: Translation
    english_text = translate_to_english(asr_text)
    print("\nTranslated Text:\n", english_text)

    # Step 3: Vector DB Retrieval
    chunks = retrieve_relevant_chunks(english_text)

    # Step 4: LLM Answer
    answer = generate_answer_from_llm(english_text, chunks)
    print("\nFinal Answer (LLM Generated):\n")
    print(answer)