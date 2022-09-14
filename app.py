import streamlit as st 
import mxnet as mx
from mxnet import image as img 
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
import hashlib
from pylab import rcParams
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import numpy as np
import os
from pathlib import Path
import re
import pickle
import os
import entohi
st.set_option('deprecation.showfileUploaderEncoding', False)

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Machine Learning Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Persons Counter
         """
         )
file= st.file_uploader("Please upload image for counting number of persons", type=("jpg", "png"))

import cv2
from  PIL import Image, ImageOps

rcParams['figure.figsize'] = 5, 10

if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  image.save('test.jpg')
  print(file)
  print(file.name)
  print(image)
  #image=np.array(image)
  #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  #image = cv2.imdecode(file_bytes, 1)
  st.image(image,caption='Uploaded Image.', use_column_width=True)


def load_image(filepath):
    
    im=img.imread(filepath)
    return im  
def transform_image(array):
    norm_image,image=data.transforms.presets.yolo.transform_test(array)
    return norm_image,image
def detect(network, data):
    pred=network(data)
    class_ids,scores,bounding_boxes=pred
    return class_ids, scores, bounding_boxes
def count_object(network, class_ids, scores, bounding_boxes, object_label, threshold=0.75):
    idx=0
    for i in range(len(network.classes)):
        if network.classes[i]==object_label:
            idx=i
    scores=scores[0]
    class_ids=class_ids[0]
    num_people=0
    for i in range(len(scores)):
        proba=scores[i].astype('float32').asscalar()
        if proba>threshold and class_ids[i].asscalar()==idx:
            num_people+=1
    return num_people    
class PersonCounter():
    def __init__(self, threshold):
        self._network = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
        self._threshold = threshold

    def set_threshold(self, threshold):
        self._threshold = threshold
        
    def count(self, filepath, visualize=False):
        image=load_image(filepath)
        norm_image,unnorm_image=transform_image(image)
        network=self._network
        class_ids, scores, bounding_boxes = detect(network, norm_image)
        if visualize:
            self._visualize(unnorm_image, class_ids, scores, bounding_boxes)
        threshold=self._threshold
        object_label='person'
        num_people=count_object(network, class_ids, scores, bounding_boxes, object_label, threshold)
        if num_people == 1:
            print('{} person detected in {}.'.format(num_people, filepath))
            
        else:
            print('{} people detected in {}.'.format(num_people, filepath))
        return num_people
    
    def _visualize(self, unnorm_image, class_ids, scores, bounding_boxes):
        ax = utils.viz.plot_bbox(unnorm_image,
                                 bounding_boxes[0],
                                 scores[0],
                                 class_ids[0],
                                 class_names=self._network.classes)
        fig = plt.gcf()
        fig.set_size_inches(8,8)
        plt.show()
counter = PersonCounter(threshold=0.2)

if st.button("COUNT The Persons"):
  a=counter.count('test.jpg', visualize=True)
  if a == 1:

    print('{} person detected in image.'.format(a))
    text=str('{} person detected in image.'.format(a))
            
  else:

    print('{} people detected in image.'.format(a))
    text=str('{} people detected in image.'.format(a))
  
  
  st.markdown(f"## Output text in English:")
  st.write(text)
  st.markdown(f"## Your audio in English:")
  enaudio = entohi.englishspeech(text)
  print(enaudio)
  audio_file = open(f"{enaudio}","rb")
  audio_bytes = audio_file.read()
  st.audio(audio_bytes,format="audio/mp3/wav", start_time=0)
  st.markdown(f"## Output text in Hindi")
  hitext=entohi.convert(text)
  print(hitext)
  st.write('{}'.format(hitext))
  hiaudio=entohi.hindispeech(hitext)
  print(hiaudio)
  audio_file = open(f"{hiaudio}", "rb")
  audio_bytes = audio_file.read()
  st.markdown(f"## Your audio in Hindi:")
  st.audio(audio_bytes, format="audio/mp3/wav", start_time=0)
  
  
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
