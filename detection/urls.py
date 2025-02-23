from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('Temp_reel', views.Temp_reel, name='Temp_reel'),
    path('video_reel/', views.video_reel, name='video_reel'),
    path('get_detected_objects/', views.get_detected_objects, name='get_detected_objects'),
    
    path('video_feed/', views.video_feed, name='video_feed'),

    # Téléversement de vidéo
    path('upload_video/', views.upload_video, name='upload_video'),

    # API pour obtenir la suggestion de mouvement
    path('get_suggestion/', views.get_suggestion, name='get_suggestion'),

  
]

