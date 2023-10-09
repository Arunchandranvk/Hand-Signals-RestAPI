from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CapturedImageViewSet,ImageClassificationView

router = DefaultRouter()
router.register(r'captured-images', CapturedImageViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('classify/', ImageClassificationView.as_view(), name='classify_image'),
]

