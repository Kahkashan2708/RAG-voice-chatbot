import os
from typing import List
import torch
import nemo.collections.asr as nemo_asr

from scripts.task4_translate import translate_to_english
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

VECTOR_DB_PATH = "vector_db/faiss_index"

# MUST be same as Task-2 embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ASR_MODEL_PATH = "models/indicconformer_stt_hi_hybrid_rnnt_large.nemo"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# LOAD ASR MODEL (DIRECT, NO FASTAPI)
# --------------------------------------------------

asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    restore_path=ASR_MODEL_PATH,
    map_location=DEVICE
)
asr_model.eval()


def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe Hindi audio using IndicConformer (NeMo)
    """

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    with torch.no_grad():
        text = asr_model.transcribe(
            audio=[audio_file_path],
            language_id="hi"
        )

    return text[0]


# --------------------------------------------------
# VECTOR DB RETRIEVAL
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
