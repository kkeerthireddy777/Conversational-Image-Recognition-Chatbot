from PIL import Image
import os
from config import Config

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_path: str) -> str:
        """
        Preprocess image (resize if too large, convert format if needed)
        Returns path to processed image
        """
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            if img.size[0] > Config.MAX_IMAGE_SIZE[0] or img.size[1] > Config.MAX_IMAGE_SIZE[1]:
                img.thumbnail(Config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Save processed image
            processed_path = image_path.replace('.', '_processed.')
            img.save(processed_path)
            
            return processed_path
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return image_path
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate if file is a valid image"""
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except:
            return False