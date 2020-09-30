#!/usr/bin/env python3
'''class Yolo'''

import tensorflow.keras as K


class Yolo():
    '''class Yolo that uses the Yolo v3 algorithm
    Attributes:
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
    '''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''Constructor
        model_path: is the path to where a Darknet Keras model is stored
        classes_path: is the path to where the list of class names used for the
        Darknet model, listed in order of index, can be found
        class_t: is a float representing the box score threshold for the
                initial filtering step
        nms_t: is a float representing the IOU threshold for non-max
               suppression
        anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing
        all of the anchor boxes:
        '''
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_classes_names(self, file_name):
        '''take file text with class names and return s list
        Args:
            file_name: path with file
        Return: a list with with classes
        '''
        with open(file_name, 'r') as f:
            class_names = f.read().splitlines()
        return class_names
