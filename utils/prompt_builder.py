from typing import Dict

class PromptBuilder:
    @staticmethod
    def build_image_context(yolo_results: Dict, blip_caption: str) -> str:
        """
        Build comprehensive image context for LLM
        """
        context_parts = []
        
        # Add BLIP caption
        context_parts.append(f"Overall Scene Description: {blip_caption}")
        
        # Add YOLO detections
        if yolo_results['total_objects'] > 0:
            context_parts.append(f"\n{yolo_results['structured_info']}")
            
            # Add detailed object information
            context_parts.append("\nDetailed Object Positions:")
            for i, det in enumerate(yolo_results['detections'], 1):
                context_parts.append(
                    f"{i}. {det['class']} - located at {det['position']} "
                    f"(confidence: {det['confidence']:.2f})"
                )
        else:
            context_parts.append("\nNo specific objects detected by the detector.")
        
        return "\n".join(context_parts)