import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
model_url = 
'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_
V1/1'
# model_url = 'on_device_vision_classifier_landmarks_classifier_asia_V1_1'
# label_url = 
'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_
label_map.csv'
labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))
def image_processing(image):
 img_shape = (321, 321)
  classifier = tf.keras.Sequential(
 [hub.KerasLayer(model_url, input_shape=img_shape + (3,), 
output_key="predictions:logits")])
 img = PIL.Image.open(image)
 img = img.resize(img_shape)
 img1 = img
 img = np.array(img) / 255.0
 img = img[np.newaxis]
 result = classifier.predict(img)
 return labels[np.argmax(result)],img1
def get_map(loc):
 geolocator = Nominatim(user_agent="Your_Name")
 location = geolocator.geocode(loc)
 return location.address,location.latitude, location.longitude
def run():
