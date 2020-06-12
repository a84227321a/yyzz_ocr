# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Dense, LSTM, add, concatenate, \
    Dropout, Lambda, Flatten
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import resnet
from utils import get_dict, create_result_subdir,load_test_sample
# from stn_utils import STN
from data_generator import TextImageGenerator,ValGenerator
from config import cfg
from GCN import GraphConvolution
from image_generator_py import ImageGenerator
from keras.regularizers import l2

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    return loss

# def bulid_gragh():
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
#
#     return features.todense(), adj, labels

def model_STN(cfg,idx_char_dict):
    if K.image_data_format() == 'channels_first':
        # input_shape = (cfg.nb_channels, cfg.height, cfg.width)
        input_shape = (cfg.nb_channels, cfg.height, None)
    else:
        # input_shape = (cfg.height, cfg.width, cfg.nb_channels)
        input_shape = (cfg.height, None, cfg.nb_channels)
    inputs_data = Input(name='the_input', shape=input_shape, dtype='float32')
    # if cfg.stn:
    #     if K.image_data_format() == 'channels_first':
    #         x = STN(inputs_data, sampling_size=input_shape[1:])
    #     else:
    #         x = STN(inputs_data, sampling_size=input_shape[:2])
    # else:
    x = inputs_data
    y_pred_1 = resnet.ResNet50(x, len(idx_char_dict)+1)

    # H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + G)



    prediction_model = Model(inputs=inputs_data, outputs=y_pred_1)

    # y_pred_2 = GraphConvolution()


    prediction_model.summary()
    # labels = Input(name='the_labels', shape=[cfg.label_len], dtype='float32')
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    ctc_loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred_1, labels, input_length, label_length])
    training_model = Model(inputs=[inputs_data, labels, input_length, label_length],
                  outputs=[ctc_loss_out])
    return training_model, prediction_model

def get_generators(output_subdir,char_idx_dict,idx_char_dict,test_sample_list,image_generator):
    train_generator = TextImageGenerator(train_dir=cfg.train_dir,
                                         image_generator=image_generator,
                                         char_idx_dict=char_idx_dict,
                                         batch_size=cfg.batch_size,
                                         img_h=cfg.height,
                                         bg_image_path=cfg.bg_image_path,
                                         max_string_len=cfg.absolute_max_string_len)

    val_generator = ValGenerator(save_dir=output_subdir,
                                 img_h=cfg.height,
                                 idx_char_dict=idx_char_dict,
                                 prediction_model=prediction_model,
                                 sample_list=test_sample_list
                                 )
    return train_generator, val_generator

def get_optimizer():
    if cfg.optimizer == 'sgd':
        opt = SGD(lr=cfg.lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    elif cfg.optimizer == 'adam':
        opt = Adam(lr=cfg.lr)
    else:
        raise ValueError('Wrong optimizer name')
    return opt

def create_output_directory():
    os.makedirs(cfg.output_dir, exist_ok=True)
    # 自动在output_dir下生成文件夹
    output_subdir = create_result_subdir(cfg.output_dir)
    print('Output directory: ' + output_subdir)
    # files = [f for f in os.listdir('.') if os.path.isfile(f)]
    # for f in files:
    #     shutil.copy(f, output_subdir)
    return output_subdir

def load_weights_if_resume_training(training_model):
    if cfg.resume_training:
        training_model.load_weights(cfg.load_model_path)
    return training_model

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    idx_char_dict, char_idx_dict = get_dict(cfg.label_pkl_path)
    print('dict init')
    test_sample_list = load_test_sample(img_root=os.path.join(cfg.test_dir,'img'), label_root=os.path.join(cfg.test_dir,'txt'),
                                        char_idx_dict=char_idx_dict)
    print('test data init')
    training_model, prediction_model = model_STN(cfg,idx_char_dict)
    print('model init')
    opt = get_optimizer()
    print('optimizer init')
    output_subdir = create_output_directory()
    image_generator = ImageGenerator(cfg.font_path)
    print('image_generator init')
    train_generator, val_generator = get_generators(output_subdir,char_idx_dict,idx_char_dict,test_sample_list,image_generator)
    print('train&val_generator init')
    # training_model.compile(loss={'ctc': lambda y_true, ctc_pred: ctc_pred}, optimizer=opt, metrics=['accuracy'])
    # use the model
    training_model = load_weights_if_resume_training(training_model)
    if cfg.resume_training ==True:
        print('exist model init')
    else:
        print("no exist model")
    training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt, metrics=['accuracy'])

    training_model.fit_generator(generator=train_generator.next_train(),
                                 steps_per_epoch=cfg.val_iter_period,
                                 epochs=cfg.nb_epochs,
                                 max_queue_size=20,
                                 # workers=cfg.nb_workers,
                                 # use_multiprocessing=cfg.use_multiprocessing,
                                 callbacks=[val_generator, train_generator,TensorBoard(log_dir=os.path.join(output_subdir,'outlog'))])


