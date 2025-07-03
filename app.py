import gradio as gr
from main import run_research_agent
from pdf_chat import load_pdf, chat_with_pdf

chat_history = []
pdf_loaded = False


def handle_query(user_input):
    global pdf_loaded

    if user_input.strip().lower() == "exit":
        return chatbox.update(value=chatbox.value + "\n\n👋 Goodbye!"), ""

    # Run either PDF QA or research agent
    if pdf_loaded:
        response = chat_with_pdf(user_input)
    else:
        response = run_research_agent(user_input)

    # Update chat history for rendering
    chat_history.append(("User", user_input))
    if isinstance(response, dict):
        summary = response.get("summary", "")
        sources = response.get("sources", [])
        tools = response.get("tools_used", [])

        sources_html = ""
        if sources:
            sources_html += "<div style='font-size: 0.85em; color: gray;'>Sources: " + ", ".join(
                f"<span style='background-color:#e0e0e0; padding: 2px 6px; border-radius: 6px;'>{s}</span>"
                for s in sources
            ) + "</div>"

        tools_html = ""
        if tools:
            tools_html += "<div style='font-size: 0.85em; color: gray;'>Tools: " + ", ".join(
                f"<span style='background-color:#d0f0d0; padding: 2px 6px; border-radius: 6px;'>{t}</span>"
                for t in tools
            ) + "</div>"

        full_response = summary + sources_html + tools_html
        chat_history.append(("Assistant", full_response))
    else:
        chat_history.append(("Assistant", response))

    # Format like ChatGPT
    formatted = ""
    for speaker, msg in chat_history:
        if speaker == "User":
            formatted += f"<div style='text-align: right; margin: 8px;'><strong>{speaker}:</strong> {msg}</div>"
        else:
            formatted += f"<div style='text-align: left; margin: 8px; background-color: #f2f2f2; padding: 8px; border-radius: 10px;'><strong>{speaker}:</strong> {msg}</div>"

    return gr.update(value=formatted, visible=True), ""


def handle_file_upload(file):
    global pdf_loaded
    if not file:
        return "No file uploaded."

    try:
        status = load_pdf(file.name)
        pdf_loaded = True
        return f"{file.name} uploaded successfully!"
    except Exception as e:
        return f"Failed to load PDF: {str(e)}"


with gr.Blocks(css=".gradio-container {max-width: 800px !important; margin: auto;}") as demo:
    gr.Markdown("## Agent Chatbot")

    with gr.Row():
        file_input = gr.File(label="Upload a PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    file_input.change(fn=handle_file_upload, inputs=file_input, outputs=upload_status)

    chatbox = gr.HTML(label="Chat History", visible=True)
    user_input = gr.Textbox(label="Type your message and press Enter")

    user_input.submit(fn=handle_query, inputs=user_input, outputs=[chatbox, user_input])

demo.launch()
