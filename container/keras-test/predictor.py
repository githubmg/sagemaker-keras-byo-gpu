# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
#from keras_definitions import *
import flask
import config
import keras
import cv2
import util
from keras.models import model_from_json


# from keras.models import Sequential
# from keras.models import Model
# from keras.layers import Input, Dense, Activation, Lambda
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import Concatenate
from config_reader import config_reader
import scipy
import math
import random
import pandas as pd
import numpy as np
from flask import request
import pprint
from scipy.ndimage.filters import gaussian_filter
from keras import backend as K 
import tensorflow as tf


graph=None
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        global graph
        if cls.model == None:
            weights_path = model_path + "/model.h5" 

            json_file = open(model_path + '/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            K.clear_session()
            model = model_from_json(loaded_model_json)

            model.load_weights(weights_path)
            
            cls.model = model
            graph = tf.get_default_graph()
            print('get default graph')
        return cls.model

    @classmethod
    def predict(cls, input_img):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        
        global graph        
        clf = cls.get_model()
        print('use default graph')
        with graph.as_default():
            preds = clf.predict(input_img) 
        return preds 

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
#     encoded = flask.request.data
#     nparr = np.frombuffer(encoded, np.uint8)
#     oriImg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     param, model_params = config_reader()
#     scale = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']][3]                      
#     heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
#     #paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

#     imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#     imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])  
#     param, model_params = config_reader()
#     scale = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']][3]
#     heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    
    encoded = flask.request.data
    nparr = np.frombuffer(encoded, np.uint8)
    imageToTest_padded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
    print("Input shape: " + str(input_img.shape))
    output_blobs = ScoringService.predict(input_img)
    heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
#     heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
#     heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
#     heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

#     heatmap_avg = heatmap_avg + heatmap
#     all_peaks = []
#     peak_counter = 0
#     for part in range(19-1):
#         map_ori = heatmap_avg[:,:,part]
#         map = gaussian_filter(map_ori, sigma=3)

#         map_left = np.zeros(map.shape)
#         map_left[1:,:] = map[:-1,:]
#         map_right = np.zeros(map.shape)
#         map_right[:-1,:] = map[1:,:]
#         map_up = np.zeros(map.shape)
#         map_up[:,1:] = map[:,:-1]
#         map_down = np.zeros(map.shape)
#         map_down[:,:-1] = map[:,1:]

#         peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
#         peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
#         peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
#         id = range(peak_counter, peak_counter + len(peaks))
#         peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

#         all_peaks.append(peaks_with_score_and_id)
#         peak_counter += len(peaks)
        
    predictions = heatmap.flatten()
    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    pd.DataFrame({'results':list(predictions)}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200)
