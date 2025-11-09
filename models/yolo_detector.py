from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple
from config import Config

class YOLODetector:
    def __init__(self):
        """Initialize YOLOv8 detector"""
        self.model = YOLO(Config.YOLO_MODEL)
        self.confidence = Config.YOLO_CONFIDENCE
        self.iou = Config.YOLO_IOU
        
    def detect_objects(self, image_path: str) -> Dict:
        """
        Detect objects in image with bounding boxes
        
        Returns:
            Dict with detections, annotated_image, and structured info
        """
        results = self.model.predict(
            source=image_path,
            conf=self.confidence,
            iou=self.iou,
            verbose=False
        )
        
        detections = []
        result = results[0]
        
        # Extract detection information
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            
            # Calculate position descriptors
            img_height, img_width = result.orig_shape
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            position = self._get_position_description(
                center_x, center_y, img_width, img_height
            )
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'position': position,
                'center': (float(center_x), float(center_y))
            })
        
        # Get annotated image
        annotated_image = result.plot()
        
        # Generate structured description
        structured_info = self._structure_detections(detections)
        
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'structured_info': structured_info,
            'total_objects': len(detections)
        }
    
    def _get_position_description(self, x: float, y: float, 
                                  width: float, height: float) -> str:
        """Generate natural language position description"""
        # Divide image into 3x3 grid
        col = "left" if x < width/3 else "center" if x < 2*width/3 else "right"
        row = "top" if y < height/3 else "middle" if y < 2*height/3 else "bottom"
        
        if col == "center" and row == "middle":
            return "center of the image"
        elif col == "center":
            return f"{row} center"
        elif row == "middle":
            return f"{col} side"
        else:
            return f"{row} {col}"
    
    def _structure_detections(self, detections: List[Dict]) -> str:
        """Create structured text description of all detections"""
        if not detections:
            return "No objects detected in the image."
        
        # Group by class
        class_counts = {}
        class_positions = {}
        
        for det in detections:
            class_name = det['class']
            position = det['position']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                class_positions[class_name] = []
            
            class_counts[class_name] += 1
            class_positions[class_name].append(position)
        
        # Build description
        description_parts = []
        for class_name, count in class_counts.items():
            positions = class_positions[class_name]
            if count == 1:
                description_parts.append(
                    f"1 {class_name} at {positions[0]}"
                )
            else:
                pos_str = ", ".join(positions[:-1]) + f" and {positions[-1]}"
                description_parts.append(
                    f"{count} {class_name}s at {pos_str}"
                )
        
        return "Detected: " + "; ".join(description_parts) + "."