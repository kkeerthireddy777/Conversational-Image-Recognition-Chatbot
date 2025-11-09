from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from config import Config

class BLIPCaptioner:
    def __init__(self):
        """Initialize BLIP-2 model for image captioning and VQA"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading BLIP-2 model on {self.device}...")
        self.processor = Blip2Processor.from_pretrained(Config.BLIP_MODEL)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            Config.BLIP_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("BLIP-2 model loaded successfully!")
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate a detailed caption for the image
        """
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5
        )
        
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return caption
    
    def answer_question(self, image_path: str, question: str) -> str:
        """
        Answer a specific question about the image using Visual Question Answering
        """
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3
        )
        
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return answer