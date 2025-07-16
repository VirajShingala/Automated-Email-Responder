# app.py
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

app = Flask(__name__)
app.secret_key = "supersecret"  # Replace in production!

# Load FLAN-T5 model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Dummy login
USERNAME = "admin@gmail.com"
PASSWORD = "admin123"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.form["username"] == USERNAME and request.form["password"] == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("email"))
        else:
            return render_template("index.html", error="❌ Invalid login")
    return render_template("index.html")

@app.route("/email", methods=["GET", "POST"])
def email():
    if not session.get("logged_in"):
        return redirect(url_for("index"))

    generated_email = ""
    if request.method == "POST":
        email_type = request.form.get("email_type")
        prompt = request.form.get("prompt")
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
        generated_email = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        with open("generated_email.txt", "w", encoding="utf-8") as f:
            f.write(generated_email)

    return render_template("email.html", email=generated_email)

@app.route("/download")
def download():
    return send_file("generated_email.txt", as_attachment=True)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
