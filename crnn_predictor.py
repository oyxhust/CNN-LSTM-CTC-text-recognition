#!/usr/bin/env python2.7
# coding=utf-8
from __future__ import print_function
import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append("../../amalgamation/python/")
sys.path.append("../../python/")
import argparse

from mxnet_predict import Predictor
import mxnet as mx

from symbol.crnn import crnn

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class lstm_ocr_model(object):
    # Keep Zero index for blank. (CTC request it)
    def __init__(self, path_of_json, path_of_params, classes, data_shape, batch_size, num_label, num_hidden, num_lstm_layer):
        super(lstm_ocr_model, self).__init__()
        self.path_of_json = path_of_json
        self.path_of_params = path_of_params
        self.classes = classes
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label
        self.num_hidden = num_hidden
        self.num_lstm_layer = num_lstm_layer
        self.predictor = None
        self.__init_ocr()

    def __init_ocr(self):
        init_c = [('l%d_init_c'%l, (self.batch_size, self.num_hidden)) for l in range(self.num_lstm_layer*2)]
        init_h = [('l%d_init_h'%l, (self.batch_size, self.num_hidden)) for l in range(self.num_lstm_layer*2)]
        init_states = init_c + init_h

        all_shapes = [('data', (batch_size, 1, self.data_shape[1], self.data_shape[0]))] + init_states + [('label', (self.batch_size, self.num_label))]
        all_shapes_dict = {}
        for _shape in all_shapes:
            all_shapes_dict[_shape[0]] = _shape[1]
        self.predictor = Predictor(open(self.path_of_json).read(),
                                    open(self.path_of_params).read(),
                                    all_shapes_dict,dev_type="gpu", dev_id=0)

    def forward_ocr(self, img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.data_shape)
        img = img.reshape((1, data_shape[1], data_shape[0]))
        img = np.multiply(img, 1/255.0)
        self.predictor.forward(data=img)
        prob = self.predictor.get_output(0)
        label_list = []
        for p in prob:
            max_index = np.argsort(p)[::-1][0]
            label_list.append(max_index)
        return self.__get_string(label_list)

    def __get_string(self, label_list):
        # Do CTC label rule
        # CTC cannot emit a repeated symbol on consecutive timesteps
        ret = []
        label_list2 = [0] + list(label_list)
        for i in range(len(label_list)):
            c1 = label_list2[i]
            c2 = label_list2[i+1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        # change to ascii
        s = ''
        for l in ret:
            if l > 0 and l < (len(self.classes)+1):
                c = self.classes[l-1]
            else:
                c = ''
            s += c
        return s

def parse_args():
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--img', dest='img', help='which image to use',
                        default=os.path.join(os.getcwd(), 'data', 'demo', '20150105_14543723_Z.jpg'), type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    json_path = os.path.join(os.getcwd(), 'model', 'crnn_ctc-symbol.json')
    param_path = os.path.join(os.getcwd(), 'model', 'crnn_ctc-0100.params')
    num_label = 9 # Set your max length of label, add one more for blank
    batch_size = 1
    num_hidden = 256
    num_lstm_layer = 2
    data_shape = (100, 32)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", 
        "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    demo_img = args.img

    _lstm_ocr_model = lstm_ocr_model(json_path, param_path, classes, data_shape, batch_size, 
                                    num_label, num_hidden, num_lstm_layer)
    img = cv2.imread(demo_img)
    #img = cv2.bitwise_not(img)
    _str = _lstm_ocr_model.forward_ocr(img)
    print('Result: ', _str)
    plt.imshow(img)
    plt.gca().text(0, 6.8,
                    '{:s} {:s}'.format("prediction", _str),
                    #bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='red')
    plt.show()
