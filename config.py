import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configurations
    YOLO_MODEL = "yolov8x.pt"  # Use yolov8x for best accuracy
    BLIP_MODEL = "Salesforce/blip2-opt-2.7b"  # or blip2-flan-t5-xl for better quality
    LLM_MODEL = "llama3-70b-8192"  # Groq's LLaMA 3 70B
    
    # Detection Thresholds
    YOLO_CONFIDENCE = 0.5
    YOLO_IOU = 0.45
    
    # Conversation Settings
    MAX_CONVERSATION_HISTORY = 10  # Keep last 10 messages
    TEMPERATURE = 0.7
    MAX_TOKENS = 1024
    
    # Image Processing
    MAX_IMAGE_SIZE = (1280, 1280)