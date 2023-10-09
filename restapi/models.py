from django.db import models



class CapturedImage(models.Model):
    image = models.ImageField(upload_to='captured_images/')
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id}"


class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    classification = models.CharField(max_length=1)