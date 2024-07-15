from django.db import models

# Create your models here.
class Location(models.model):
    name=models.CharField(max_length=100)
    geom=models.PointField()

class Road(models.model):
    name=models.CharField(max_length=100)
    geom=models.LineStringField()

class SatelliteLayer(models.model):
    name=models.CharField(max_length=100)
    geom=models.PolygonField()
    
