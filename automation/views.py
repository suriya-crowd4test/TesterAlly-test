import webbrowser
import pyautogui
import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os

@csrf_exempt
def open_browser(request):
    if request.method == "POST":
        import json
        data = json.loads(request.body)
        command = data.get("command", "")

        if "open" in command.lower():
            url = command.split("open")[-1].strip()
            if not url.startswith("http"):
                url = "https://" + url
            webbrowser.open(url)  # Open URL
            time.sleep(3)  # Wait for browser to load

            screenshot_path = os.path.join("media", "screenshot.png")
            pyautogui.screenshot(screenshot_path)  # Take screenshot

            return JsonResponse({"message": f"Opened {url}", "screenshot": screenshot_path})

    return JsonResponse({"error": "Invalid request"}, status=400)
