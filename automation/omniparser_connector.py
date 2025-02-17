'''
from typing import Dict, Any, List, Tuple, Optional
import torch
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
import io
import base64
import traceback
import logging
from .OmniParser.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

import warnings
warnings.filterwarnings("ignore", message=".*prepare_inputs_for_generation.*")

logger = logging.getLogger(__name__)

class OmniParserConnector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Base directory: {BASE_DIR}")

        # Define and validate paths
        yolo_model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_detect", "model.pt")
        caption_model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_caption_florence")

        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model file not found at: {yolo_model_path}")
        if not os.path.exists(caption_model_path):
            raise FileNotFoundError(f"Caption model path not found at: {caption_model_path}")

        # Load models
        try:
            self.yolo_model = get_yolo_model(model_path=yolo_model_path)
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=caption_model_path
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def _validate_box(self, box) -> bool:
        """Validate if a box has valid coordinates."""
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        try:
            coords = [float(coord) for coord in box]
            return all(-10000 < coord < 10000 for coord in coords)
        except (ValueError, TypeError):
            return False

    def _handle_ocr_result(self, ocr_result: Any) -> Tuple[List[str], List[List[float]]]:
        """Handle OCR results with enhanced validation and error handling."""
        logger.info(f"Raw OCR result: {ocr_result}")
    
        texts, boxes = [], []
    
        try:
            if isinstance(ocr_result, tuple) and len(ocr_result) > 0 and isinstance(ocr_result[0], tuple):
                inner_tuple = ocr_result[0]
                if len(inner_tuple) >= 2:
                    raw_texts, raw_boxes = inner_tuple[0], inner_tuple[1]
                    
                    # Process texts
                    if isinstance(raw_texts, (list, tuple)):
                        texts = [str(text).strip() for text in raw_texts if text and str(text).strip()]
                    
                    # Process boxes
                    if isinstance(raw_boxes, (list, tuple)):
                        for box in raw_boxes:
                            if isinstance(box, (list, tuple)) and len(box) == 4:
                                try:
                                    box_coords = [float(coord) for coord in box]
                                    if self._validate_box(box_coords):
                                        boxes.append(box_coords)
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Invalid box coordinates: {box}, error: {str(e)}")
                                    continue
            
            # Ensure matching texts and boxes
            min_len = min(len(texts), len(boxes))
            if min_len > 0:
                texts = texts[:min_len]
                boxes = boxes[:min_len]
                logger.info(f"Successfully processed {len(texts)} text elements with valid boxes")
                return texts, boxes
            
            logger.warning("No valid text-box pairs found after processing")
            return [], []
                
        except Exception as e:
            logger.error(f"Error processing OCR result: {str(e)}\n{traceback.format_exc()}")
            return [], []

    def process_image(
        self, 
        image_file, 
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        use_paddleocr: bool = True,
        imgsz: int = 640
    ) -> Dict[str, Any]:
        """Process image and return structured results."""
        temp_path = None
        try:
            logger.info(" Loading image...")
            image = Image.open(image_file)
            
            if image.size[0] < 10 or image.size[1] < 10:
                return {'success': False, 'error': 'Image too small or invalid'}
            
            # Save image with unique name
            temp_name = f'screenshot_{hash(str(image_file))}.png'
            os.makedirs('media', exist_ok=True)
            temp_path = os.path.join('media', temp_name)
            image.save(temp_path)
            logger.info(f" Image saved to: {temp_path}")
            
            # OCR configuration
            ocr_config = {
                'paragraph': False,
                'text_threshold': 0.7,
                'link_threshold': 0.4,
                'low_text': 0.4,
                'min_size': 10,
                'mag_ratio': 1.5
            }
            
            # Run OCR with multiple attempts
            logger.info("🔍 Running OCR...")
            max_attempts = 2
            ocr_result = None
            
            for attempt in range(max_attempts):
                try:
                    ocr_result = check_ocr_box(
                        temp_path,
                        display_img=False,
                        output_bb_format='xyxy',
                        goal_filtering=None,
                        easyocr_args=ocr_config,
                        use_paddleocr=use_paddleocr
                    )
                    if ocr_result:
                        break
                except Exception as e:
                    logger.warning(f"OCR attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        use_paddleocr = not use_paddleocr
                        continue
                    return {'success': False, 'error': 'OCR processing failed after multiple attempts'}

            # Process OCR results
            texts, boxes = self._handle_ocr_result(ocr_result)
            
            if not texts or not boxes:
                return {
                    'success': False, 
                    'error': 'No text detected in the image',
                    'details': {
                        'message': 'Could not detect readable text in the image',
                        'ocr_result_type': str(type(ocr_result)),
                        'raw_result': str(ocr_result)
                    }
                }

            logger.info(f"✅ OCR successful - Found {len(texts)} text elements")
            
            # Run YOLO detection
            logger.info("🔍 Running YOLO detection...")
            try:
                yolo_results = self.yolo_model(temp_path, imgsz=imgsz)
                boxes_yolo = []
                classes_yolo = []
                
                if len(yolo_results) > 0:
                    result = yolo_results[0] 
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            if box.conf.item() > box_threshold:
                                box_coords = box.xyxy[0].cpu().numpy().tolist()
                                boxes_yolo.append(box_coords)
                                class_idx = int(box.cls[0].item())
                                classes_yolo.append(result.names[class_idx])
                
                logger.info(f"✅ YOLO detection successful - Found {len(boxes_yolo)} objects")
            except Exception as e:
                logger.error(f"YOLO detection failed: {str(e)}")
                boxes_yolo = []
                classes_yolo = []

            # Generate labeled image
            try:
                boxes = [box if isinstance(box, list) else box.tolist() for box in boxes]
                boxes_yolo = [box if isinstance(box, list) else box.tolist() for box in boxes_yolo]

                original_image = Image.open(temp_path)
                
                # Draw OCR boxes
                labeled_image = original_image.copy()
                draw = ImageDraw.Draw(labeled_image)
                
                # Draw OCR text boxes in blue
                for text, box in zip(texts, boxes):
                    draw.rectangle(box, outline='blue', width=2)
                    draw.text((box[0], box[1] - 10), text[:20], fill='blue')
                
                # Draw YOLO boxes in red
                for cls, box in zip(classes_yolo, boxes_yolo):
                    draw.rectangle(box, outline='red', width=2)
                    draw.text((box[0], box[1] - 10), cls, fill='red')
                
                # Convert to base64
                buffered = io.BytesIO()
                labeled_image.save(buffered, format="PNG")
                labeled_image_b64 = base64.b64encode(buffered.getvalue()).decode()
                
            except Exception as e:
                logger.error(f"Failed to generate labeled image: {str(e)}\n{traceback.format_exc()}")
                labeled_image_b64 = None


            # Return successful result
            return {
                'success': True,
                'text_elements': [
                    {'text': text, 'box': box}
                    for text, box in zip(texts, boxes)
                ],
                'detected_objects': [
                    {'class': cls, 'box': box}
                    for cls, box in zip(classes_yolo, boxes_yolo)
                ],
                'labeled_image': labeled_image_b64
            }

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }
        finally:
            # Cleanup temporary files
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file: {str(e)}")

'''

#omniparser_connector.py code 

from typing import Dict, Any, List, Tuple, Optional
import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import traceback
import logging
import json
from OmniParser.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

import warnings
warnings.filterwarnings("ignore", message=".*prepare_inputs_for_generation.*")

logger = logging.getLogger(__name__)

class OmniParserConnector:
    # Class constants matching Gradio's configuration
    BOX_THRESHOLD_CONFIG = {
        'minimum': 0.01,
        'maximum': 1.0,
        'step': 0.01,
        'default': 0.05
    }
    
    IOU_THRESHOLD_CONFIG = {
        'minimum': 0.01,
        'maximum': 1.0,
        'step': 0.01,
        'default': 0.1
    }
    
    IMGSZ_CONFIG = {
        'minimum': 640,
        'maximum': 1920,
        'step': 32,
        'default': 640
    }

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Base directory: {BASE_DIR}")

        # Define and validate paths
        yolo_model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_detect", "model.pt")
        caption_model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_caption_florence")

        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model file not found at: {yolo_model_path}")
        if not os.path.exists(caption_model_path):
            raise FileNotFoundError(f"Caption model path not found at: {caption_model_path}")

        # Load models
        try:
            self.yolo_model = get_yolo_model(model_path=yolo_model_path)
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=caption_model_path
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def _validate_step_value(self, value: float, config: dict) -> bool:
        """
        Validate if a value adheres to the step constraints.
        """
        steps = (value - config['minimum']) / config['step']
        return abs(steps - round(steps)) < 1e-10

    def _validate_parameters(
        self,
        box_threshold: float,
        iou_threshold: float,
        imgsz: int
    ) -> Tuple[bool, str]:
        """Validate input parameters against allowed ranges and steps."""
        # Validate box_threshold
        if not (self.BOX_THRESHOLD_CONFIG['minimum'] <= box_threshold <= self.BOX_THRESHOLD_CONFIG['maximum']):
            return False, f"box_threshold must be between {self.BOX_THRESHOLD_CONFIG['minimum']} and {self.BOX_THRESHOLD_CONFIG['maximum']}, got {box_threshold}"
        if not self._validate_step_value(box_threshold, self.BOX_THRESHOLD_CONFIG):
            return False, f"box_threshold must be in steps of {self.BOX_THRESHOLD_CONFIG['step']}, got {box_threshold}"
        
        # Validate iou_threshold
        if not (self.IOU_THRESHOLD_CONFIG['minimum'] <= iou_threshold <= self.IOU_THRESHOLD_CONFIG['maximum']):
            return False, f"iou_threshold must be between {self.IOU_THRESHOLD_CONFIG['minimum']} and {self.IOU_THRESHOLD_CONFIG['maximum']}, got {iou_threshold}"
        if not self._validate_step_value(iou_threshold, self.IOU_THRESHOLD_CONFIG):
            return False, f"iou_threshold must be in steps of {self.IOU_THRESHOLD_CONFIG['step']}, got {iou_threshold}"
            
        # Validate imgsz
        if not (self.IMGSZ_CONFIG['minimum'] <= imgsz <= self.IMGSZ_CONFIG['maximum']):
            return False, f"imgsz must be between {self.IMGSZ_CONFIG['minimum']} and {self.IMGSZ_CONFIG['maximum']}, got {imgsz}"
        if not self._validate_step_value(imgsz, self.IMGSZ_CONFIG):
            return False, f"imgsz must be in steps of {self.IMGSZ_CONFIG['step']}, got {imgsz}"
            
        return True, ""

    def _validate_box(self, box) -> bool:
        """Validate if a box has valid coordinates."""
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        try:
            coords = [float(coord) for coord in box]
            return all(-10000 < coord < 10000 for coord in coords)
        except (ValueError, TypeError):
            return False

    def _handle_ocr_result(self, ocr_result: Any) -> Tuple[List[str], List[List[float]]]:
        """Handle OCR results with enhanced validation and error handling."""
        logger.info(f"Raw OCR result: {ocr_result}")
    
        texts, boxes = [], []
    
        try:
            if isinstance(ocr_result, tuple) and len(ocr_result) > 0 and isinstance(ocr_result[0], tuple):
                inner_tuple = ocr_result[0]
                if len(inner_tuple) >= 2:
                    raw_texts, raw_boxes = inner_tuple[0], inner_tuple[1]
                    
                    # Process texts
                    if isinstance(raw_texts, (list, tuple)):
                        texts = [str(text).strip() for text in raw_texts if text and str(text).strip()]
                    
                    # Process boxes
                    if isinstance(raw_boxes, (list, tuple)):
                        for box in raw_boxes:
                            if isinstance(box, (list, tuple)) and len(box) == 4:
                                try:
                                    box_coords = [float(coord) for coord in box]
                                    if self._validate_box(box_coords):
                                        boxes.append(box_coords)
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Invalid box coordinates: {box}, error: {str(e)}")
                                    continue
            
            # Ensure matching texts and boxes
            min_len = min(len(texts), len(boxes))
            if min_len > 0:
                texts = texts[:min_len]
                boxes = boxes[:min_len]
                logger.info(f"Successfully processed {len(texts)} text elements with valid boxes")
                return texts, boxes
            
            logger.warning("No valid text-box pairs found after processing")
            return [], []
                
        except Exception as e:
            logger.error(f"Error processing OCR result: {str(e)}\n{traceback.format_exc()}")
            return [], []

    def process_image(
        self, 
        image_file, 
        box_threshold: float = BOX_THRESHOLD_CONFIG['default'],
        iou_threshold: float = IOU_THRESHOLD_CONFIG['default'],
        use_paddleocr: bool = True,
        imgsz: int = IMGSZ_CONFIG['default']
    ) -> Dict[str, Any]:
        """
        Process image with parameter validation.
        
        Parameters:
        -----------
        image_file : File
            The input image file
        box_threshold : float, optional
            Confidence threshold for bounding boxes (range: 0.01 to 1.0, step: 0.01)
        iou_threshold : float, optional
            IOU threshold for overlap detection (range: 0.01 to 1.0, step: 0.01)
        use_paddleocr : bool, optional
            Whether to use PaddleOCR
        imgsz : int, optional
            Image size for icon detection (range: 640 to 1920, step: 32)
            
        Returns:
        --------
        Dict[str, Any]
            Processing results including detected elements and labeled image
        """
        temp_path = None
        try:
            # Validate parameters
            is_valid, error_message = self._validate_parameters(
                box_threshold, 
                iou_threshold, 
                imgsz
            )
            
            if not is_valid:
                return {
                    'success': False,
                    'error': error_message
                }

            logger.info(" Loading image...")
            image = Image.open(image_file)
            
            if image.size[0] < 10 or image.size[1] < 10:
                return {'success': False, 'error': 'Image too small or invalid'}
            
            # Save image with unique name
            temp_name = f'screenshot_{hash(str(image_file))}.png'
            os.makedirs('media', exist_ok=True)
            temp_path = os.path.join('media', temp_name)
            image.save(temp_path)
            logger.info(f" Image saved to: {temp_path}")
            
            # OCR configuration
            ocr_config = {
                'paragraph': False,
                'text_threshold': 0.7,
                'link_threshold': 0.4,
                'low_text': 0.4,
                'min_size': 10,
                'mag_ratio': 1.5
            }
            
            # Run OCR with multiple attempts
            logger.info("🔍 Running OCR...")
            max_attempts = 2
            ocr_result = None
            
            for attempt in range(max_attempts):
                try:
                    ocr_result = check_ocr_box(
                        temp_path,
                        display_img=False,
                        output_bb_format='xyxy',
                        goal_filtering=None,
                        easyocr_args=ocr_config,
                        use_paddleocr=use_paddleocr
                    )
                    if ocr_result:
                        break
                except Exception as e:
                    logger.warning(f"OCR attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        use_paddleocr = not use_paddleocr
                        continue
                    return {'success': False, 'error': 'OCR processing failed after multiple attempts'}

            # Process OCR results
            texts, boxes = self._handle_ocr_result(ocr_result)
            
            if not texts or not boxes:
                return {
                    'success': False, 
                    'error': 'No text detected in the image',
                    'details': {
                        'message': 'Could not detect readable text in the image',
                        'ocr_result_type': str(type(ocr_result)),
                        'raw_result': str(ocr_result)
                    }
                }

            logger.info(f"✅ OCR successful - Found {len(texts)} text elements")
            
            # Run YOLO detection
            logger.info("🔍 Running YOLO detection...")
            try:
                yolo_results = self.yolo_model(temp_path, imgsz=imgsz)
                boxes_yolo = []
                classes_yolo = []
                
                if len(yolo_results) > 0:
                    result = yolo_results[0] 
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            if box.conf.item() > box_threshold:
                                box_coords = box.xyxy[0].cpu().numpy().tolist()
                                boxes_yolo.append(box_coords)
                                class_idx = int(box.cls[0].item())
                                classes_yolo.append(result.names[class_idx])
                
                logger.info(f"✅ YOLO detection successful - Found {len(boxes_yolo)} objects")
            except Exception as e:
                logger.error(f"YOLO detection failed: {str(e)}")
                boxes_yolo = []
                classes_yolo = []

            # Generate labeled image
            try:
                boxes = [box if isinstance(box, list) else box.tolist() for box in boxes]
                boxes_yolo = [box if isinstance(box, list) else box.tolist() for box in boxes_yolo]

                original_image = Image.open(temp_path)
                
                # Draw OCR boxes
                labeled_image = original_image.copy()
                draw = ImageDraw.Draw(labeled_image)
                
                # Draw OCR text boxes in blue
                for text, box in zip(texts, boxes):
                    draw.rectangle(box, outline='blue', width=2)
                    draw.text((box[0], box[1] - 10), text[:20], fill='blue')
                
                # Draw YOLO boxes in red
                for cls, box in zip(classes_yolo, boxes_yolo):
                    draw.rectangle(box, outline='red', width=2)
                    draw.text((box[0], box[1] - 10), cls, fill='red')
                
                # Convert to base64
                buffered = io.BytesIO()
                labeled_image.save(buffered, format="PNG")
                labeled_image_b64 = base64.b64encode(buffered.getvalue()).decode()
                
            except Exception as e:
                logger.error(f"Failed to generate labeled image: {str(e)}\n{traceback.format_exc()}")
                labeled_image_b64 = None

            # Return successful result
            return {
                'success': True,
                'text_elements': [
                    {'text': text, 'box': box}
                    for text, box in zip(texts, boxes)
                ],
                'detected_objects': [
                    {'class': cls, 'box': box}
                    for cls, box in zip(classes_yolo, boxes_yolo)
                ],
                'labeled_image': labeled_image_b64
            }

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc()
            }
        finally:
            # Cleanup temporary files
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file: {str(e)}")