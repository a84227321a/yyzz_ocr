# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 15:04
# @Author  : W07

from urllib import parse,request
import time
import json
import hashlib
import random
import base64
import string

def GetAccessToken(formdata, app_key):
    '''
    获取签名
    :param formdata:请求参数键值对
    :param app_key:应用秘钥
    :return:返回接口调用签名
    '''
    dic = sorted(formdata.items(), key=lambda d: d[0])
    sign = parse.urlencode(dic) + '&app_key=' + app_key
    m = hashlib.md5()
    m.update(sign.encode('utf8'))
    return m.hexdigest().upper()


def RecogniseGeneral(host,app_id, time_stamp, nonce_str, image, app_key):
    '''
    腾讯OCR通用接口
    :param app_id:应用标识，正整数
    :param time_stamp:请求时间戳（单位秒），正整数
    :param nonce_str: 随机字符串，非空且长度上限32字节
    :param image:原始图片的base64编码
    :return:
    '''
    formdata = {'app_id': app_id, 'time_stamp': time_stamp, 'nonce_str': nonce_str, 'image': image}
    sign = GetAccessToken(formdata=formdata, app_key=app_key)
    formdata['sign'] = sign
    print(formdata)
    req = request.Request(method='POST', url=host, data=parse.urlencode(formdata).encode('utf8'))
    print(req)
    response = request.urlopen(req)
    if (response.status == 200):
        json_str = response.read().decode()
        print(json_str)
        jobj = json.loads(json_str)
        datas = jobj['data']['item_list']
        recognise = {}
        for obj in datas:
            recognise[obj['itemstring']] = obj
        return recognise


def Recognise(host,img_path,app_id,app_key):
    with open(file=img_path, mode='rb') as file:
        base64_data = base64.b64encode(file.read())
    nonce = ''.join(random.sample(string.digits + string.ascii_letters, 32))
    nonce = "11111"
    stamp = int(time.time())
    base64_data = '222'
    recognise = RecogniseGeneral(host,app_id, stamp, nonce, base64_data,
                                 app_key) # 替换成自己的app_id,app_key
    for k, v in recognise.items():
        print(k, v)
    return recognise

if __name__ == '__main__':
    url = u'https://api.ai.qq.com/fcgi-bin/ocr/ocr_bizlicenseoc'
    app_key = "XngTjZ8UR2kr8cWB"
    app_id = 2111156426
    img_path = r'E:\文档\营业执照识别\5.9YYZZ\竖版企业\JPEGImages\45903_4_18.jpg'
    time_stamp = time.time()
    Recognise(url,img_path,app_id,app_key)