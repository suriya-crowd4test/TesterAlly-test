'''
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .omniparser_connector import OmniParserConnector
import traceback
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render

def screenshot_analyzer(request):
    return render(request, 'screenshot_analyzer.html')

@api_view(['POST'])
def analyze_screenshot(request):
    logger.info("\n=== Starting Screenshot Analysis ===")
    logger.info(" Received API request to /api/analyze-screenshot/")

    # Validate image file
    if 'image' not in request.FILES:
        logger.error(" No image file provided")
        return JsonResponse({'success': False, 'error': 'No image file provided'}, status=400)

    try:
        # Get and validate parameters with proper type conversion and error handling
        try:
            box_threshold = float(request.POST.get('box_threshold', 0.05))
            iou_threshold = float(request.POST.get('iou_threshold', 0.1))
            use_paddleocr = request.POST.get('use_paddleocr', 'true').lower() == 'true'
            imgsz = int(request.POST.get('imgsz', 640))
        except ValueError as e:
            logger.error(f"Parameter validation error: {str(e)}")
            return JsonResponse({'success': False, 'error': f'Invalid parameter value: {str(e)}'}, status=400)

        logger.info(f"Parameters: box_threshold={box_threshold}, iou_threshold={iou_threshold}, "
                   f"use_paddleocr={use_paddleocr}, imgsz={imgsz}")

        # Validate parameter ranges
        if not (0 < box_threshold <= 1):
            return JsonResponse({'success': False, 'error': 'box_threshold must be between 0 and 1'}, status=400)
        if not (0 < iou_threshold <= 1):
            return JsonResponse({'success': False, 'error': 'iou_threshold must be between 0 and 1'}, status=400)
        if not (640 <= imgsz <= 1920):
            return JsonResponse({'success': False, 'error': 'imgsz must be between 640 and 1920'}, status=400)

        # Initialize connector with error handling
        try:
            connector = OmniParserConnector()
            logger.info(" OmniParserConnector initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OmniParserConnector: {str(e)}")
            return JsonResponse({'success': False, 'error': f'Failed to initialize image processor: {str(e)}'}, status=500)

        # Process the image
        result = connector.process_image(
            request.FILES['image'],
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz
        )

        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            trace = result.get('trace', '')
            logger.error(f" Processing failed: {error_msg}\n{trace}")
            return JsonResponse(result, status=500)

        logger.info(" Processing completed successfully")
        return JsonResponse(result)

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f" Exception: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JsonResponse({
            'success': False, 
            'error': error_msg,
            'trace': traceback.format_exc()
        }, status=500)


import pyautogui
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time


logger = logging.getLogger(__name__)

@csrf_exempt
def execute_action(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        action = data.get('action')
        coordinates = data.get('coordinates')
        text = data.get('text')
        
        if not coordinates:
            return JsonResponse({'success': False, 'error': 'No coordinates provided'})
        
        # Calculate center point of the box
        x1, y1, x2, y2 = coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Configure PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5
        
        # Move mouse to position smoothly
        pyautogui.moveTo(center_x, center_y, duration=0.5)
        
        if action == 'click':
            # Perform click
            pyautogui.click(center_x, center_y)
            logger.info(f"Clicked at coordinates ({center_x}, {center_y})")
            
        elif action == 'type':
            if not text:
                return JsonResponse({'success': False, 'error': 'No text provided for type action'})
            
            # Click first to focus
            pyautogui.click(center_x, center_y)
            time.sleep(0.5)  # Wait for focus
            
            # Type the text
            pyautogui.typewrite(text)
            logger.info(f"Typed '{text}' at coordinates ({center_x}, {center_y})")
            
        return JsonResponse({'success': True})
        
    except Exception as e:
        logger.error(f"Error executing action: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)})

'''
# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
from PIL import Image
import io
import base64
import os
from django.conf import settings
from .OmniParser.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_detect", "model.pt")
caption_model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_caption_florence")


# Initialize models
yolo_model = get_yolo_model(model_path)
caption_model_processor = get_caption_model_processor(
    model_name="florence2", 
    model_name_or_path=caption_model_path 
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get parameters from request
            image = Image.open(request.FILES['image'])
            box_threshold = float(request.POST.get('box_threshold', 0.05))
            iou_threshold = float(request.POST.get('iou_threshold', 0.1))
            use_paddleocr = request.POST.get('use_paddleocr', 'true').lower() == 'true'
            imgsz = int(request.POST.get('imgsz', 640))

            # Save temporary image
            temp_dir = os.path.join(settings.BASE_DIR, 'media')
            os.makedirs(temp_dir, exist_ok=True)
            image_save_path = os.path.join(temp_dir, 'temp_image.png')
            image.save(image_save_path)

            # Process image
            box_overlay_ratio = image.size[0] / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }

            # OCR and processing
            ocr_bbox_rslt, _ = check_ocr_box(
                image_save_path,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                use_paddleocr=use_paddleocr
            )
            
            text, ocr_bbox = ocr_bbox_rslt
            dino_labled_img, _, parsed_content_list = get_som_labeled_img(
                image_save_path,
                yolo_model,
                BOX_TRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
            )

            # Format results
            parsed_content = '\n'.join([f'icon {i}: {str(v)}' for i, v in enumerate(parsed_content_list)])

            # Clean up temporary file
            if os.path.exists(image_save_path):
                os.remove(image_save_path)

            return JsonResponse({
                'success': True,
                'processed_image': dino_labled_img,  # This is already base64
                'parsed_content': parsed_content
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({
        'success': False,
        'error': 'Invalid request'
    })