# -*- coding: utf-8 -*-
import glob
import os
import pickle
import cv2
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import *
from keras import backend as K
import re
import codecs

def create_result_subdir(result_dir):
    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase)
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d' % (run_id))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    return result_subdir

def get_dict(label_pkl_path):
    with open(label_pkl_path, 'rb') as f:
        idx_char_dict, char_idx_dict = pickle.load(f)
    return idx_char_dict,char_idx_dict

def pad_image(img, img_size, nb_channels):
    # img_size : (width, height)
    # loaded_img_shape : (height, width)
    img_reshape = cv2.resize(img, (int(img_size[1] / img.shape[0] * img.shape[1]), img_size[1]))
    if nb_channels == 1:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1])), dtype=np.int32)
    else:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1]), nb_channels),
                           dtype=np.int32)
    img = np.concatenate([img_reshape, padding], axis=1)
    return img


def resize_image(img, img_size):
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = np.asarray(img)
    return img

def load_test_sample(img_root, label_root, char_idx_dict):
    label_name_list = os.listdir(label_root)
    sample_list = []
    for label_name in label_name_list:
        label_path = os.path.join(label_root, label_name)
        img_path = os.path.join(img_root, re.sub('txt', 'jpg', label_name))
        with codecs.open(label_path, 'rb', encoding='utf-8') as label_file:
            text = label_file.readline()
            flag = False
            for char in text:
                if char not in char_idx_dict:
                    flag = True
                    break
            if flag:
                continue
            # img = cv2.imread(img_path)
            # try:
            # load_train_img(img_path, 32)
            # except:
            # print(img_path)
            #     continue
            sample_list.append([img_path, text])

    return sample_list