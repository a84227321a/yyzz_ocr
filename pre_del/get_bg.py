# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 15:13
# @Author  : W07
# 剪切搜集到的营业执照背景图片
import cv2

def cut_bg(im_path,new_path,point):
    img = cv2.imread(im_path)
    #print(img.shape)
    #print(type(img))
    new_img = img[point[0][1]:point[1][1],point[0][0]:point[1][0]]
    # print(new_img)
    #cv2.imshow('bg',img)
    #cv2.imshow('ag',new_img)
    #cv2.waitKey(0)
    cv2.imwrite(new_path,new_img)
if __name__ == '__main__':
    im_path = r'E:\code\yyzz_ocr\data\mess\bg_nodeal.jpg'
    new_path = r'E:\code\yyzz_ocr\data\bg\bg.jpg'
    point = [(387,1887),(891,2455)]
    cut_bg(im_path,new_path,point)


