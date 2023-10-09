from django.shortcuts import render

# # Create your views here.
from rest_framework import viewsets,status
import cv2
import numpy as np
import math
import time
from rest_framework.response import Response
from rest_framework.decorators import api_view,APIView
from rest_framework import viewsets
from .models import CapturedImage,Image
from .serializers import CapturedImageSerializer,ImageSerializer
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import CapturedImage
from .serializers import CapturedImageSerializer
from cvzone.HandTrackingModule import HandDetector
from rest_framework.parsers import MultiPartParser
from cvzone.ClassificationModule import Classifier


class CapturedImageViewSet(viewsets.ModelViewSet):
    queryset = CapturedImage.objects.all()
    serializer_class = CapturedImageSerializer



@api_view(['POST'])
def capture_image(request):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imgsize = 300

    folder = "data/#space"
    counter = 0

    while True:
        suc, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
            imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgcropshape = imgcrop.shape

            aspectratio = h / w
            if aspectratio > 1:
                k = imgsize / h
                wcal = math.ceil(k * w)
                imgresize = cv2.resize(imgcrop, (wcal, imgsize))
                imgresizeshape = imgresize.shape
                wgap = math.ceil((imgsize - wcal) / 2)
                imgwhite[:, wgap:wcal + wgap] = imgresize
            else:
                k = imgsize / w
                hcal = math.ceil(k * h)
                imgresize = cv2.resize(imgcrop, (imgsize, hcal))
                imgresizeshape = imgresize.shape
                hgap = math.ceil((imgsize - hcal) / 2)
                imgwhite[hgap:hcal + hgap, :] = imgresize

            cv2.imshow('imgcrop', imgcrop)
            cv2.imshow('imgwhite', imgwhite)

        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            counter += 1
            timestamp = time.time()
            image_filename = f'{folder}/Image_{timestamp:.2f}.jpg'
            saved = cv2.imwrite(image_filename, imgwhite)
            if saved:
                print(f'Saved image as {image_filename}')

                # Save the image data to the database
                captured_image = CapturedImage(image=f'captured_images/{image_filename}')
                captured_image.save()

            else:
                print(f'Failed to save image as {image_filename}')
            print(counter)

        cap.release()
        cv2.destroyAllWindows()
        return Response({'message': 'Image captured and saved to the database.'}, status=status.HTTP_201_CREATED)


class ImageClassificationView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        # Your image classification logic here (similar to your original code)
        # You can use request.data['image'] to access the uploaded image file

        # Example:
        img = cv2.imdecode(np.fromstring(request.data['image'].read(), np.uint8), cv2.IMREAD_COLOR)

        detector = HandDetector(maxHands=1)
        classifier = Classifier("D:\Internship Luminar\Ml Api\sign_language-\model\keras_model.h5", 'D:\Internship Luminar\Ml Api\sign_language-\model\labels.txt')
        offset = 20
        imgsize = 300
        labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                  'w', 'x', 'y', 'z']

        hands, _ = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
            imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            
            imgcropshape = imgcrop.shape
            aspectratio = h / w

            if aspectratio > 1:
                k = imgsize / h
                wcal = math.ceil(k * w)
                imgresize = cv2.resize(imgcrop, (wcal, imgsize))
                imgresizeshape = imgresize.shape
                wgap = math.ceil((imgsize - wcal) / 2)
                imgwhite[:, wgap:wcal + wgap] = imgresize
                prediction, index = classifier.getPrediction(imgwhite, draw=False)
                print(prediction, index)
            else:
                k = imgsize / w
                hcal = math.ceil(k * h)
                imgresize = cv2.resize(imgcrop, (imgsize, hcal))
                imgresizeshape = imgresize.shape
                hgap = math.ceil((imgsize - hcal) / 2)
                imgwhite[hgap:hcal + hgap, :] = imgresize
                prediction, index = classifier.getPrediction(imgwhite, draw=False)
            # Image processing and classification logic (similar to your original code)
            # ...

            # Save classification result to the database
            classification_result = labels[index]  # Replace with your classification result
            image_instance = Image(image=request.data['image'], classification=classification_result)
            image_instance.save()

            return Response({'classification': classification_result}, status=status.HTTP_201_CREATED)

        return Response({'message': 'No hands detected'}, status=status.HTTP_400_BAD_REQUEST)





