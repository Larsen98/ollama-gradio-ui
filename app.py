import gradio as gr
import requests
import base64

# --- Ollama Setup ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"

# --- Prompt ---
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
- Write a cohesive, integrated description in English.
"""

# --- Analyse mit Ollama ---
def analyze_with_ollama(images, progress=gr.Progress()):
    if not images:
        return "Bitte ein oder mehrere Bilder hochladen."

    progress(0, desc="Starte Analyse …")

    try:
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

        progress(0.5, desc="Sende Bilder an Ollama …")
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

        progress(1, desc="Fertig!")
        return result if result else "Keine Antwort von Ollama erhalten."

    except Exception as e:
        return f"Fehler bei Analyse mit Ollama: {e}"


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Technical Museum Object Analyzer (Ollama)")

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

    # 👉 Direkt starten, wenn Bilder hochgeladen werden
    image_input.change(fn=analyze_with_ollama, inputs=image_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
