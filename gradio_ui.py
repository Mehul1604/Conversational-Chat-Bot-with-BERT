import gradio as gr

from get_preds import get_response

demo = gr.Interface(
    fn=get_response,
    inputs=gr.components.Textbox(label='Input'),
    outputs=gr.components.Label(label='Response'),
    allow_flagging='never'
)