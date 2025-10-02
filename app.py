import gradio as gr
import requests
import base64
import json
import os
from openai import OpenAI

# OpenAI-Client initialisieren (API-Key muss als Umgebungsvariable gesetzt sein)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Versteckter Prompt (faktenbasiert, keine Spekulation)
hidden_prompt = """Objective:
Generate a concise, strictly objective description of the technical museum object 
shown in the uploaded images. The description must be based **only on visible evidence** 
in the images. Do not speculate or invent any information.

Rules:
- Describe exclusively what can be clearly observed (shape, materials, labels, markings).
- Do not add assumptions about usage, origin, or history unless explicitly visible or labeled.
- If details are unclear or not visible, omit them entirely.
- Avoid subjective or interpretive language (no “probably”, “might be”, “appears to”).
- Length: As detailed as possible, but only as long as factual content allows. 
  (If only a few facts are visible, keep the text short.)

Required Structure & Content:
1. Identification (only if visibly labeled; otherwise omit)
2. Physical Characteristics (size impression, materials, form, color, condition)
3. Technical Function (only if clearly deducible from visible features; otherwise omit)
4. Distinguishing Details (markings, numbers, accessories, engineering evidence)

Output:
- Write a cohesive, integrated description.
- Use accessible, objective prose.
- Provide the final description in 4 separate sections, in this exact order: 
  English, Polish, German, French.
"""

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"  # Vision-Modell in Ollama

# Funktion für OpenAI
def analyze_with_openai(images):
    image_contents = []
    for image in images:
        with open(image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
            image_contents.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # oder "gpt-4o"
        messages=[
            {"role": "system", "content": hidden_prompt},
            {"role": "user", "content": [{"type": "text", "text": "Analyze the uploaded images accordingly."}] + image_contents}
        ],
        max_tokens=800
    )
    return response.choices[0].message.content


# Funktion für Ollama
def analyze_with_ollama(images):
    img_b64_list = []
    for image in images:
        with open(image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
            img_b64_list.append(img_b64)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": hidden_prompt,
        "images": img_b64_list
    }

    response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
    if response.status_code != 200:
        return f"Ollama API Error: {response.status_code} {response.text}"

    result = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            if '"response":"' in data:
                try:
                    part = data.split('"response":"')[1].split('"')[0]
                    result += part
                except Exception:
                    pass

    return result if result else "Keine Antwort von Ollama erhalten."


# Hauptfunktion mit Auswahl + Fortschrittsanzeige
def analyze_images(images, backend, progress=gr.Progress()):
    if not images:
        return "Bitte ein oder mehrere Bilder hochladen."

    progress(0, desc="Starte Analyse …")
    try:
        if backend == "OpenAI":
            progress(0.5, desc="Sende Bilder an OpenAI …")
            result = analyze_with_openai(images)
        else:
            progress(0.5, desc="Sende Bilder an Ollama …")
            result = analyze_with_ollama(images)

        progress(1, desc="Fertig!")
        return result

    except Exception as e:
        return f"Fehler bei Analyse mit {backend}: {e}"


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Technical Museum Object Analyzer")

    with gr.Row():
        backend_choice = gr.Radio(
            ["OpenAI", "Ollama"],
            value="Ollama",
            label="Backend auswählen"
        )

    with gr.Row():
        image_input = gr.File(
    	file_types=[".jpg", ".jpeg", ".png"],
    	label="Bilder hochladen (mehrere möglich)",
    	file_count="multiple"
	)

    with gr.Row():
        output_text = gr.Textbox(
            lines=25,
            label="Beschreibung (EN, PL, DE, FR)",
            show_copy_button=True
        )

    # Analyse-Button mit Fortschritt
    analyze_btn = gr.Button("Objekt analysieren")
    analyze_btn.click(fn=analyze_images, inputs=[image_input, backend_choice], outputs=output_text)

if __name__ == "__main__":
    demo.launch()