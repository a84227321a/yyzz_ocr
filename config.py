# -*- coding: utf-8 -*-
import argparse
import string
import multiprocessing

parser = argparse.ArgumentParser()

parser.add_argument('--resume_training', type=bool, default=False)
parser.add_argument('--load_model_path', type=str, default=r'E:\code\yyzz_ocr\model\000\model.h5')
parser.add_argument('--output_dir', type=str, default='model')

parser.add_argument('--gpus', type=str, nargs='*', default='0')
parser.add_argument('--nb_channels', type=int, default=3)
parser.add_argument('--height', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--absolute_max_string_len', type=int, default=15)

parser.add_argument('--label_pkl_path', type=str, default=r'E:\code\yyzz_ocr\data\char\number.pkl')
parser.add_argument('--train_dir', type=str, default=r'F')
parser.add_argument('--test_dir', type=str, default=r'E:\code\yyzz_ocr\data\gen_test')


parser.add_argument('--bg_image_path', type=str, default=r'F:\ocr\bg_image')
parser.add_argument('--font_path', type=str, default=r'E:\code\yyzz_ocr\data\font')

parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--nb_epochs', type=int, default=100)
parser.add_argument('--val_iter_period', type=int, default=10000)
parser.add_argument('--nb_workers', type=int, default=multiprocessing.cpu_count())


cfg = parser.parse_args()
