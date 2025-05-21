import gradio as gr

def greet(name):
    return f"Hello {name}!"

with gr.Blocks() as demo:
    gr.Markdown("## Gradio Test")
    txt = gr.Textbox(placeholder="Enter your name")
    btn = gr.Button("Greet")
    txt.submit(greet, txt, None)  # Call greet when user presses Enter in the textbox
    btn.click(greet, txt, None)  # Call greet when user clicks the button

print("Starting Gradio app...")  # Ensure the app is starting
demo.launch(server_name="0.0.0.0", server_port=7861)
