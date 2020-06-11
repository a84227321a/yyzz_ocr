# -*- coding: utf-8 -*-
# @Time    : 2020/6/11 9:09
# @Author  : W07
# 生成相关数字部分的字符文件
import string
import pickle

def gen_pickle(pkl_path):
    number_list = string.digits
    alber_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chi_num = "壹贰叁肆伍陆柒捌玖拾佰仟万亿元零整卅廿"
    an_str = "至永久人民币年月日止(-)长期美"
    an_num = "一二三四五六七八九十"

    re_str = number_list+alber_list+chi_num+an_num+an_str
    id_str_dic = dict()
    str_id_dic = dict()
    for id,i in enumerate(re_str):
        id_str_dic[id] = i
        str_id_dic[i] = id
    with open(pkl_path,'wb') as f:
        pickle.dump((id_str_dic,str_id_dic),f)

def get_pickle(pkl_path):
    with open(pkl_path,'rb') as f:
        id_str_dic,str_id_dic = pickle.load(f)
    return id_str_dic,str_id_dic
if __name__ == '__main__':
    pkl_path = '../data/char/number.pkl'
    # gen_pickle(pkl_path)
    print(get_pickle(pkl_path))