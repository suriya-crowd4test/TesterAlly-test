'''
from django.urls import path
from . import views

urlpatterns = [
    path('check/', views.screenshot_analyzer, name='screenshot_analyzer'),
    path('api/execute-action/', views.execute_action, name='execute-action'),
    path('analyze-screenshot/', views.analyze_screenshot, name='analyze_screenshot'),
]

'''

from django.urls import path
from . import views

urlpatterns = [
    path('check', views.home, name='home'),
    path('process-image/', views.process_image, name='process_image'),
]