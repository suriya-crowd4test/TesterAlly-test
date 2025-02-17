'''
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

'''


from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
from PIL import Image, ImageDraw
import io
import base64
import os
from django.conf import settings
from .OmniParser.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import json

# Initialize models (your existing initialization code)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_detect", "model.pt")
caption_model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_caption_florence")

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

            # Create validation image
            validation_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
            draw = ImageDraw.Draw(validation_image)
            
            # Draw validation boxes
            width, height = validation_image.size
            for i, content in enumerate(parsed_content_list):
                bbox = content['bbox']
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # Draw red box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # Add label
                draw.text((x1, y1-15), f"Box {i+1}: {content['content']}", fill="red")

            # Convert validation image back to base64
            buffered = io.BytesIO()
            validation_image.save(buffered, format="PNG")
            validation_image_base64 = base64.b64encode(buffered.getvalue()).decode()

            formatted_content = '\n'.join([
                f'icon {i}: {{ "bbox": {v["bbox"]}, "content": "{v["content"]}", "interactivity": {v["interactivity"]} }}'
                for i, v in enumerate(parsed_content_list)
            ])


            # Clean up temporary file
            if os.path.exists(image_save_path):
                os.remove(image_save_path)

            return JsonResponse({
                'success': True,
                'processed_image': validation_image_base64,
                'parsed_content': formatted_content,
                'coordinates': [
                    {
                        'bbox': item['bbox'],
                        'content': item['content'],
                        'interactivity': item['interactivity']
                    }
                    for item in parsed_content_list
                ]
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