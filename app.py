# app.py
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Dummy login logic
def check_login(username, password):
    return username == "admin@gmail.com" and password == "admin123"

def handle_login(username, password):
    if check_login(username, password):
        return gr.update(visible=False), gr.update(visible=True), ""
    else:
        return gr.update(visible=True), gr.update(visible=False), "❌ Invalid login"

def handle_logout():
    return gr.update(visible=True), gr.update(visible=False)

def generate_email(email_type, prompt):
    instruction = f"Write a professional, detailed {email_type.lower()} email for the following topic: {prompt}. Include greeting, body, and closing. Use at least 6-8 sentences."
    inputs = tokenizer(instruction, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def save_email_to_file(email_text):
    with open("generated_email.txt", "w", encoding="utf-8") as f:
        f.write(email_text)
    return "generated_email.txt"

with gr.Blocks(theme=gr.themes.Monochrome(), css="body { font-family: 'Segoe UI', sans-serif; }") as demo:
    gr.Markdown("<h1 style='text-align: center;'>📧 Automated Email Responder</h1>")
    gr.Markdown("<p style='text-align: center; color: lightgray;'>Generate professional emails using FLAN-T5 Large</p>")

    with gr.Row():
        with gr.Column(visible=True, scale=1, min_width=300) as login_section:
            gr.Markdown("### 🔐 Login to Continue")
            username = gr.Textbox(label="Email", placeholder="admin@gmail.com")
            password = gr.Textbox(label="Password", type="password", placeholder="admin123")
            login_msg = gr.Markdown("")
            login_btn = gr.Button("Login", variant="primary")

        with gr.Column(visible=False, scale=2, min_width=400) as email_section:
            gr.Markdown("### 📝 Compose Email")
            email_type = gr.Dropdown(
                label="Select Email Type",
                choices=["Apology", "Request", "Reminder", "Appreciation", "Notification", "Follow-up", "Custom"],
                value="Apology"
            )
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. delay in submitting project", lines=2)
            generate_btn = gr.Button("🚀 Generate Email", variant="primary")
            email_output = gr.Textbox(label="Generated Email", lines=12)

            with gr.Row():
                download_btn = gr.Button("⬇️ Download Email")
                copy_btn = gr.Button("📋 Copy to Clipboard")
                logout_btn = gr.Button("🔒 Logout")

            email_file = gr.File(label="Download File", visible=True)

    # Bind logic
    login_btn.click(fn=handle_login, inputs=[username, password], outputs=[login_section, email_section, login_msg])
    generate_btn.click(fn=generate_email, inputs=[email_type, prompt], outputs=email_output)
    download_btn.click(fn=save_email_to_file, inputs=email_output, outputs=email_file)
    copy_btn.click(None, inputs=email_output, outputs=None, js="navigator.clipboard.writeText(arguments[0])")
    logout_btn.click(fn=handle_logout, outputs=[login_section, email_section])

# Launch the app (Render-ready)
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
