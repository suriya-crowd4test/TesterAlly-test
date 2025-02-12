from django.urls import path
from .views import open_browser

urlpatterns = [
    path('open_browser/', open_browser, name='open_browser'),
]
