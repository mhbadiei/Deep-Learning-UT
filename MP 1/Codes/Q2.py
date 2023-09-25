from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')

model = VGG19()
image = load_img('gdrive/MyDrive/Colab Notebooks/index.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
yhat = model.predict(image)
label = decode_predictions(yhat)
Len = np.array(label).shape[1]
for i in range(Len):
  label_ = label[0][i]
  print('%s (%.2f%%)' % (label_[1], label_[2]*100))