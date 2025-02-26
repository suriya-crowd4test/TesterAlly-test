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

'''


# from django.shortcuts import render
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import torch
# from PIL import Image
# import io
# import base64
# import os
# from django.conf import settings
# from .OmniParser.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_detect", "model.pt")
# caption_model_path = os.path.join(BASE_DIR, "OmniParser", "weights", "icon_caption_florence")


# # Initialize models
# yolo_model = get_yolo_model(model_path)
# caption_model_processor = get_caption_model_processor(
#     model_name="florence2", 
#     model_name_or_path=caption_model_path 
# )

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def home(request):
#     return render(request, 'home.html')

# @csrf_exempt
# def process_image(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         try:
#             # Get parameters from request
#             image = Image.open(request.FILES['image'])
#             box_threshold = float(request.POST.get('box_threshold', 0.05))
#             iou_threshold = float(request.POST.get('iou_threshold', 0.1))
#             use_paddleocr = request.POST.get('use_paddleocr', 'true').lower() == 'true'
#             imgsz = int(request.POST.get('imgsz', 640))

#             # Save temporary image
#             temp_dir = os.path.join(settings.BASE_DIR, 'media')
#             os.makedirs(temp_dir, exist_ok=True)
#             image_save_path = os.path.join(temp_dir, 'temp_image.png')
#             image.save(image_save_path)

#             # Process image
#             box_overlay_ratio = image.size[0] / 3200
#             draw_bbox_config = {
#                 'text_scale': 0.8 * box_overlay_ratio,
#                 'text_thickness': max(int(2 * box_overlay_ratio), 1),
#                 'text_padding': max(int(3 * box_overlay_ratio), 1),
#                 'thickness': max(int(3 * box_overlay_ratio), 1),
#             }

#             # OCR and processing
#             ocr_bbox_rslt, _ = check_ocr_box(
#                 image_save_path,
#                 display_img=False,
#                 output_bb_format='xyxy',
#                 goal_filtering=None,
#                 easyocr_args={'paragraph': False, 'text_threshold': 0.9},
#                 use_paddleocr=use_paddleocr
#             )
            
#             text, ocr_bbox = ocr_bbox_rslt
#             dino_labled_img, _, parsed_content_list = get_som_labeled_img(
#                 image_save_path,
#                 yolo_model,
#                 BOX_TRESHOLD=box_threshold,
#                 output_coord_in_ratio=True,
#                 ocr_bbox=ocr_bbox,
#                 draw_bbox_config=draw_bbox_config,
#                 caption_model_processor=caption_model_processor,
#                 ocr_text=text,
#                 iou_threshold=iou_threshold,
#                 imgsz=imgsz,
#             )

#             # Format results
#             parsed_content = '\n'.join([f'icon {i}: {str(v)}' for i, v in enumerate(parsed_content_list)])

#             # Clean up temporary file
#             if os.path.exists(image_save_path):
#                 os.remove(image_save_path)

#             return JsonResponse({
#                 'success': True,
#                 'processed_image': dino_labled_img,  # This is already base64
#                 'parsed_content': parsed_content
#             })

#         except Exception as e:
#             return JsonResponse({
#                 'success': False,
#                 'error': str(e)
#             })

#     return JsonResponse({
#         'success': False,
#         'error': 'Invalid request'
#     })


# import os
# import time
# import pyautogui
# import subprocess
# import json
# import requests
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from PIL import Image

# # Paths and API endpoints
# BROWSER_PATH = "/usr/bin/firefox"  # Path to the Firefox executable
# MEDIA_DIR = "media/screenshots"  # Directory to save screenshots
# OMNIPARSER_API = "http://127.0.0.1:7861/process"  # Replace with actual API URL

# # Ensure media/screenshots directory exists
# os.makedirs(MEDIA_DIR, exist_ok=True)

# # Store the browser process globally to prevent reopening
# browser_process = None

# def open_browser(url):
#     """
#     Opens a URL in Firefox. If the browser is already open, it reuses the same session.
#     """
#     global browser_process
#     if browser_process is None:
#         # Open the browser for the first time
#         browser_process = subprocess.Popen([BROWSER_PATH, url])
#     else:
#         # Switch to the already opened browser and navigate to a new URL
#         pyautogui.hotkey("ctrl", "l")  # Focus on the address bar
#         pyautogui.write(url)
#         pyautogui.press("enter")

#     time.sleep(5)  # Wait for the page to load

# def take_screenshot(filename="screenshot.png"):
#     """
#     Takes a screenshot of the current screen and saves it to the media/screenshots folder.
#     """
#     filepath = os.path.join(MEDIA_DIR, filename)
#     screenshot = pyautogui.screenshot()
#     screenshot.save(filepath)
#     return filepath

# def send_to_omnparser(filepath):
#     """
#     Sends the screenshot to OmniParser API and retrieves UI element coordinates.
#     """
#     with open(filepath, "rb") as image_file:
#         files = {"file": image_file}
#         response = requests.post(OMNIPARSER_API, files=files)
        
#         if response.status_code == 200:
#             return response.json()  # Expecting { "x": value, "y": value }
#         else:
#             return None

# def perform_action(command, data):
#     """
#     Performs UI actions based on the command.
#     - Click: Moves the mouse to (x, y) and clicks.
#     - Type: Types the given text.
#     """
#     if command == "click":
#         x, y = data.get("x"), data.get("y")
#         if x is not None and y is not None:
#             pyautogui.moveTo(x, y)
#             pyautogui.click()
#             return JsonResponse({"status": "clicked", "coordinates": {"x": x, "y": y}})
#         return JsonResponse({"error": "Invalid coordinates"}, status=400)

#     elif command == "type":
#         text = data.get("text")
#         if text:
#             pyautogui.write(text)
#             return JsonResponse({"status": "typed", "text": text})
#         return JsonResponse({"error": "No text provided"}, status=400)

#     return JsonResponse({"error": "Invalid action"}, status=400)

# @csrf_exempt
# def handle_command(request):
#     """
#     Django view to handle automation commands from the user.
#     - Open <URL>: Opens the given URL in the browser.
#     - Click <element>: Clicks on the given element based on coordinates.
#     - Type <text>: Types text into a field.
#     """
#     global browser_process
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             command = data.get("command")

#             if not command:
#                 return JsonResponse({"error": "No command provided"}, status=400)

#             # Open a new URL or reuse the existing browser session
#             if command.startswith("open "):  
#                 url = command.split(" ", 1)[1]  # Extract the URL
#                 open_browser(url)

#                 # Capture and process screenshot
#                 screenshot_path = take_screenshot()
#                 omniparser_data = send_to_omnparser(screenshot_path)

#                 return JsonResponse({
#                     "status": "browser opened",
#                     "screenshot": screenshot_path,
#                     "coordinates": omniparser_data
#                 })

#             # Handle UI actions (click, type)
#             elif command in ["click", "type"]:
#                 return perform_action(command, data)

#             return JsonResponse({"error": "Invalid command"}, status=400)

#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON format"}, status=400)

#     return JsonResponse({"error": "Invalid request method"}, status=405)



import os
import time
import pyautogui
import subprocess
import json
import requests
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# Paths and constants
BROWSER_PATH = "/usr/bin/google-chrome"  # Path to Google Chrome
MEDIA_DIR = os.path.join(settings.MEDIA_ROOT, "screenshots")  # Save in Django's media folder
OMNIPARSER_API_URL = "https://omniparser-api.com/parse"  # Replace with actual Omniparser API URL

# Ensure media/screenshots directory exists
os.makedirs(MEDIA_DIR, exist_ok=True)

# Store the browser process globally
browser_process = None

def open_browser(url):
    """Opens a new browser window with the given URL or navigates to it if already open."""
    global browser_process
    if browser_process is None:
        browser_process = subprocess.Popen([BROWSER_PATH, "--new-window", url])
    else:
        pyautogui.hotkey("ctrl", "l")
        pyautogui.write(url)
        pyautogui.press("enter")
    
    time.sleep(8)  # Allow time for browser to open
    os.system("xdotool search --onlyvisible --class chrome windowactivate")  # Ensure browser is focused
    time.sleep(2)
    wait_for_page_load()


def wait_for_page_load():
    """Waits for the page to load by assuming a delay and checking periodically."""
    print("Waiting for page to load...")
    time.sleep(3)
    os.system("xdotool search --onlyvisible --class chrome windowactivate")
    time.sleep(2)
    for _ in range(5):
        print("Checking page load status...")
        time.sleep(3)
    print("Page load assumed complete.")


def take_screenshot(url):
    """Captures a screenshot of the current browser window and saves it."""
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
    screenshot_filename = f"{domain}.png"
    screenshot_path = os.path.join(MEDIA_DIR, screenshot_filename)

    if os.path.exists(screenshot_path):
        os.remove(screenshot_path)

    time.sleep(3)  # Allow time for UI to stabilize before taking a screenshot
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)

    return screenshot_path, f"/media/screenshots/{screenshot_filename}"


def send_to_omnparser(screenshot_path):
    """Sends the screenshot to Omniparser API for UI element recognition."""
    with open(screenshot_path, "rb") as image_file:
        response = requests.post(OMNIPARSER_API_URL, files={"file": image_file})
    if response.status_code == 200:
        return response.json()
    return None


def perform_ui_action(action, element_name, omniparser_response):
    """Performs a UI action (click or type) on an identified element."""
    for element in omniparser_response:
        if element.get("name").lower() == element_name.lower():
            click_point = element.get("click_point")
            if action == "click":
                pyautogui.click(click_point[0], click_point[1])
            elif action.startswith("type "):
                pyautogui.click(click_point[0], click_point[1])
                pyautogui.write(action.split(" ", 1)[1])
            return True
    return False


@csrf_exempt
def handle_command(request):
    """Handles incoming automation commands via API requests."""
    global browser_process
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            command = data.get("command")

            if not command:
                return JsonResponse({"error": "No command provided"}, status=400)

            if command.startswith("open "):
                url = command.split(" ", 1)[1]
                open_browser(url)
                screenshot_path, screenshot_url = take_screenshot(url)
                omniparser_response = send_to_omnparser(screenshot_path)
                return JsonResponse({
                    "status": "browser opened",
                    "screenshot": screenshot_url,
                    "omnparser_data": omniparser_response
                })

            elif command.startswith("click ") or command.startswith("type "):
                parts = command.split(" ", 1)
                action = parts[0]
                element_name = parts[1]
                omniparser_response = send_to_omnparser(screenshot_path)
                if perform_ui_action(action, element_name, omniparser_response):
                    return JsonResponse({"status": f"{action} performed on {element_name}"})
                else:
                    return JsonResponse({"error": "Element not found"}, status=400)

            return JsonResponse({"error": "Invalid command"}, status=400)
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)

