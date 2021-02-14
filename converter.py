# import tensorflow-gpu as tf
# model=tf.keras.models.load_model('C:/Users/Administrator/Mask_RCNN/NomeroffNet/mcm/models/Detector/mrcnn/mask_rcnn_numberplate_0640_2019_06_24.h5')
import os
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_DIR = NOMEROFF_NET_DIR
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")

converter = tf.lite.TFLiteConverter.from_keras_model(nnet)
# model=tf.keras.models.load_model('C:/Users/Administrator/Mask_RCNN/NomeroffNet/mcm/models/Detector/mrcnn/mask_rcnn_numberplate_0640_2019_06_24.h5')
# model = tf.keras.models.load_model('C:/Users/Administrator/Mask_RCNN/models/numberplate_options_2019_05_15.h5')
# model = tf.keras.models.load_model('C:/Users/Administrator/Mask_RCNN/models/TextDetector/eu/apr_ocr_eu_2-cpu.h5')
# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file('C:/Users/Administrator/Mask_RCNN/models/anpr_ocr_eu_2-cpu.h5') 
# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
open('C:/Users/Administrator/Mask_RCNN/NomeroffNet/mcm/models/Detector/mrcnn/converted_model.tflite', 'wb').write(tflite_model)