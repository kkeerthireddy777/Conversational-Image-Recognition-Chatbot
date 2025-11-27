# 🤖 Conversational Image Recognition Chatbot  
### YOLOv8 + BLIP-2 + LLaMA (Ollama) + Gradio UI

A conversational multimodal AI system that can **analyze images**, **detect objects**, **describe scenes**, and **answer questions** about the uploaded image — in a smooth, chat-like interface.

This project integrates **YOLOv8**, **BLIP-2**, and **LLaMA (via Ollama)** into a unified pipeline served through a **Gradio web interface**.

---

# 🚀 Features

### 🔍 Image Understanding
- Upload or capture an image  
- (Optional) Provide an image URL  
- Automatic object detection via **YOLOv8**  
- Image captioning via **BLIP-2**  
- Detection visualization with bounding boxes

### 💬 Conversational Interaction
- Ask natural questions about the image  
- Multi-turn memory maintained by **LLaMA/Ollama**  
- Scene reasoning (colors, relationships, positions, counts)  
- Spatial intelligence (e.g., *"What's on the left?"*)

### 🎨 Interface
- Clean, modern **Gradio Blocks UI**  
- Chat history viewer  
- Upload → Analyze → Chat workflow  
- Annotated output panel

---

# 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Object Detection | YOLOv8 (`yolov8x.pt`) |
| Image Captioning | BLIP-2 (Vision-Language) |
| Conversational LLM | LLaMA (via Ollama) |
| Interface | Gradio |
| Backend | Python + PyTorch |
| Environment | Mac M1 local + optional Google Colab GPU |

---

# 🏗️ Architecture

### 🖼️ High-Level Flow

User → Gradio UI
→ app.py
→ ConversationalImageChatbot (main.py)
→ YOLO detector (models/yolov8x.pt)
→ BLIP-2 captioner (utils / models)
→ LLaMA via Ollama (localhost:11434)
→ Chat response → UI


This repository contains:

CONVERSATIONAL-IMAGE-CHATBOT/
│
├── app.py # Gradio UI + user interaction
├── main.py # Core chatbot pipeline
├── config.py # Configs (paths, thresholds, etc.)
│
├── models/ # YOLO + BLIP-2 model weights (gitignored)
│ └── yolov8x.pt
│
├── utils/ # Helper modules (hashing, image utils, etc.)
│
├── tests/ # Smoke tests for components
│
├── requirements.txt # Python packages
├── pyproject.toml
├── uv.lock
│
├── .env # API keys / environment variables
├── .gitignore
├── .python-version
│
├── README.md # Documentation
└── temp_image_* # Temporary files generated during runtime


