import pandas as pd
import numpy as np
import cv2
import os

import tensorflow as tf
from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

tf.config.run_functions_eagerly(True)


model = load_model(r"./CNN/efficient.h5")

def pred(img):
  im2 = cv2.imread(img)
  im2 = cv2.resize(im2, (224,224)) # resize to 224,224 as that is on which model is trained on
  img2 = tf.expand_dims(im2, 0) # expand the dims means change shape from (224, 224, 3) to (1, 224, 224, 3)
  res = np.argmax(model.predict(img2),axis=1)[0]
  return res

from CNN.data import c, s

def return_info(x):
  questions = []
  monument = c[x]
  for k in s[x]:
    questions.append(k)
  return monument, questions

def show_info(img):
  prediction = pred(img)
  monu, info = return_info(prediction)
  information = s[prediction][info[0]]
  ques_ans = []
  for i in info[1:]:
    ques = i
    ans = s[prediction][i]
    all = {'question': ques, 'answer': ans}
    ques_ans.append(all)

  return  monu, information, ques_ans