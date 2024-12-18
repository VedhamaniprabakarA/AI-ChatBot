"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# AI_Chatbot_api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("ask/", views.chat_bot_response, name="chat_bot_response"),   # Accessed at /api/ask/
    path("csrf/", views.csrf_token_view, name="csrf_token"),           # Accessed at /api/csrf/
]
