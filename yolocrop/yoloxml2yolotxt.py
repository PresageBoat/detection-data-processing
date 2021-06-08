# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from multiprocessing import Pool
import os.path as osp
import glob

classes = ["vehicle"]


class Yoloxml2Yolotxt(object):
    def __init__(self,xml_dir,txt_dir,num_process=8):
        self.in_xml_dir = osp.join(xml_dir)
        self.out_txt_dir = osp.join(txt_dir)
        images = glob.glob(self.in_xml_dir + '*.xml')
        image_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], images)]
        self.image_ids = image_ids

        if not osp.isdir(self.in_xml_dir):
            os.makedirs(self.in_xml_dir)
        if not osp.isdir(self.out_txt_dir):
            os.makedirs(self.out_txt_dir)
        self.num_process = num_process
    
    def _convert(self,size, box):
        """原样保留。size为图片大小
            将ROI的坐标转换为yolo需要的坐标
            size是图片的w和h
            box里保存的是ROI的坐标（x，y的最小值和最大值）
            返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)
    
    def _xml2txt(self,image_id):
        # 现在传进来的只有图片名没有后缀
        in_file = open(osp.join(self.in_xml_dir, image_id + '.xml'),'rb')
        out_file = open(osp.join(self.out_txt_dir,'%s.txt' % (image_id)), 'w+')

        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        # 在一个XML中每个Object的迭代
        for obj in root.iter('object'):
            # iter()方法可以递归遍历元素/树的所有子元素

            difficult = obj.find('difficult').text
            # 属性标签
            cls = obj.find('name').text
            # 如果训练标签中的品种不在程序预定品种，或者difficult = 1，跳过此object
            if cls not in classes or int(difficult) == 1:
                continue
            # cls_id 只等于1
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            # b是每个Object中，一个bndbox上下左右像素的元组
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
            bb = self._convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def run(self):
        with Pool(self.num_process) as p:
            p.map(self._xml2txt,self.image_ids)


if __name__ == "__main__":
    dir="./test"
    lable_dir=dir+"/labels/"
    xml_dir=dir+"/xml/"

    xml2dota=Yoloxml2Yolotxt(xml_dir,lable_dir)
    xml2dota.run()

