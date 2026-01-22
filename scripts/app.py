import gradio as gr
from task5_rag_pipeline import run_rag_pipeline


def voice_chatbot(audio_file):
    """
    Takes audio file path from Gradio,
    runs full RAG pipeline,
    returns final answer text.
    """
    if audio_file is None:
        return "Please upload an audio file."

    answer = run_rag_pipeline(audio_file)
    return answer


with gr.Blocks(title="Voice-enabled RAG Chatbot") as demo:
    gr.Markdown("## 🎙️ Voice-enabled RAG Chatbot")
    gr.Markdown("Upload an audio file (Hindi supported) and get the answer.")

    audio_input = gr.Audio(
        type="filepath",
        label="Upload Audio (.wav)"
    )

    output_text = gr.Textbox(
        label="Chatbot Answer",
        lines=6
    )

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        fn=voice_chatbot,
        inputs=audio_input,
        outputs=output_text
    )

demo.launch()



