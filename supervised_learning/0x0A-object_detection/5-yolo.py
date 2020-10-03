#!/usr/bin/env python3
'''class Yolo'''

import tensorflow.keras as K
import numpy as np
import glob
import cv2


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

    def sigmoid(self, x):
        '''sigmoid function'''
        s = 1 / (1 + np.exp(-x))
        return s

    def process_outputs(self, outputs, image_size):
        '''Process Outputs
           Args:
            outputs:is a list of numpy.ndarrays containing the predictions from
            the Darknet model for a single image
            image_size: is a numpy.ndarray containing the image’s original size
            [image_height, image_width]
        '''
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            box_confidences.append(self.sigmoid(output[:, :, :, 4:5]))
            box_class_probs.append(self.sigmoid(output[:, :, :, 5:]))
        return boxes, box_confidences, box_class_probs

    @staticmethod
    def load_images(folder_path):
        '''
        Args:
        folder_path: a string representing the path to the folder holding all
                    the images to load
        Returns: a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        '''
        image_paths = []
        images = []
        for image in glob.glob(folder_path + '/*'):
            image_paths.append(image)
            images.append(cv2.imread(image))
        return (images, image_paths)

    @staticmethod
    def preprocess_images(self, images):
        '''
        Args:
            images: a list of images as numpy.ndarrays
            Resize the images with inter-cubic interpolation
            Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
            containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model Note: this can
                        vary by model
                input_w: the input width for the Darknet model Note: this can
                        vary by model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2) containing the
                        original height and width of the images
            2 => (image_height, image_width)
        '''
        pimages = []
        image_shapes = []
        # new dimensions according to input expected by model
        new_width = self.model.input.shape[1]
        new_height = self.model.input.shape[2]
        for img in images:
            # every image is a pixel matrix so you can get it size from shape
            image_shapes.append([img.shape[0], img.shape[1]])
            # resize image to new dimensions
            resize = cv2.resize(img, (new_width, new_height),
                                interpolation=cv2.INTER_CUBIC)
            # rescale pixel values between 0-1
            rescale = resize / 255
            pimages.append(rescale)
        return (np.array(pimages), np.array(image_shapes))
