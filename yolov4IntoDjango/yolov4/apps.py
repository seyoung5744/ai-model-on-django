from django.apps import AppConfig
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

class Yolov4Config(AppConfig):
    name = 'yolov4'

    saved_model_loaded = tf.saved_model.load('./yolov4/checkpoints/yolov4-416', tags=[tag_constants.SERVING])
