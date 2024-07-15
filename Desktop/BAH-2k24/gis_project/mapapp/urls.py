from django.urls import path
from . import views

urlpatterns = [
    path('api/zoom_in', views.zoom_in, name='zoom_in'),
    path('api/zoom_out', views.zoom_out, name='zoom_out'),
    path('api/show_roads', views.show_roads, name='show_roads'),
    path('api/show_satellite', views.show_satellite, name='show_satellite'),
    path('api/locate', views.locate, name='locate'),
    path('api/weather', views.weather, name='weather')
]

