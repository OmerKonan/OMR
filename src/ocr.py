from keras.models import load_model
import cv2

import keras
print(keras.__version__)


model = load_model("../model.h5")
img_path = "../m.jpg"
img = cv2.imread(img_path, 0)
mapping = ['0', '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',\
			'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th3 = cv2.subtract(255, th3)
pred_img = th3
pred_img = cv2.resize(pred_img, (28, 28))
pred_img = pred_img / 255.0
pred_img = pred_img.reshape(1,784)
prediction = mapping[model.predict_classes(pred_img)[0]]
print("################################################")
print("\n\nPredicted Value : {}".format(prediction))