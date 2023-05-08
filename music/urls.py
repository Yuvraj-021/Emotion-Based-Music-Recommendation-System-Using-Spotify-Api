"""music URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from music import views

urlpatterns = [
    path('admin-panel/', admin.site.urls),
    path('', views.home),
    path('blog/', views.blog),
    path('about/', views.about),
    path('profile/', views.profile),
    path('facecam_feed', views.facecam_feed, name='facecam_feed'),
    path('stop/', views.stop_emotion_detection, name='stop_emotion_detection'),
    path('spotify/',views.spotifyApi,name='spotifyApi'),
    path('stop_streaming/', views.stop_streaming, name='stop_streaming'),
]
