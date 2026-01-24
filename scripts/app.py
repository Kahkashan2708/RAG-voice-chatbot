import os
import sys
os.environ["GITHUB_TOKEN"] = "XXXXXXXXXXXXXXXXXXXXXXX"
sys.path.insert(0, "/content/NeMo")
import gradio as gr
from task5_rag_pipeline import run_rag_pipeline


def voice_chatbot(audio_file):
    """
    Takes audio file path from Gradio,
    runs full RAG pipeline,
    returns final answer text.
    """
    if audio_file is None:
        return "❌ Please upload an audio file (.wav)."

    try:
        answer = run_rag_pipeline(audio_file)
        return answer
    except Exception as e:
        return f"⚠️ Error occurred:\n{str(e)}"


with gr.Blocks(
    title="Voice-enabled RAG Chatbot",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🎙️ Voice-enabled RAG Chatbot
        ### Hindi Speech → AI-powered Answer

        This application performs:
        - 🎧 **Automatic Speech Recognition (IndicConformer)**
        - 🌐 **Hindi → English Translation**
        - 📚 **Context Retrieval using FAISS Vector DB**
        - 🤖 **Answer Generation using LLM (GitHub Models)**

        Upload a **Hindi audio (.wav)** file and get an intelligent answer.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="🎧 Upload Audio (.wav)",
                interactive=True
            )

            submit_btn = gr.Button(
                "🚀 Ask Question",
                variant="primary"
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="🤖 Chatbot Answer",
                lines=10,
                interactive=False,
                placeholder="The answer will appear here..."
            )

    submit_btn.click(
        fn=voice_chatbot,
        inputs=audio_input,
        outputs=output_text
    )

    gr.Markdown(
        """
        ---
        **Tech Stack:**
        NeMo IndicConformer · FastAPI · FAISS · LangChain · Gradio · GitHub Models
        """
    )
demo.launch(share=True)



