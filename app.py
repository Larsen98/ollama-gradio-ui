import gradio as gr
import requests
import base64

# Lokaler Ollama-Server
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"  # Modellname: z. B. "llava", "bakllava", "llava-phi3"

def analyze_image(image):
    # Bild in Base64 konvertieren
    with open(image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Anfrage an Ollama senden
    payload = {
        "model": MODEL_NAME,
        "prompt": "Beschreibe dieses Bild detailliert auf Deutsch.",
        "images": [img_b64]
    }
    response = requests.post(OLLAMA_API_URL, json=payload, stream=True)

    result = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            if '"response":"' in data:
                part = data.split('"response":"')[1].split('"')[0]
                result += part
    return result

# Gradio Interface
demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Ollama Bildanalyse"
)

if __name__ == "__main__":
    demo.launch()