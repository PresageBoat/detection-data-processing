# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from multiprocessing import Pool
import os.path as osp
import glob

classes = ["button"]

class Yoloxml2Dotatxt(object):
    def __init__(self,xml_dir,dota_dir,num_process=8):
        self.in_xml_dir = osp.join(xml_dir)
        self.out_dota_dir = osp.join(dota_dir)
        images = glob.glob(self.in_xml_dir + '*.xml')
        image_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], images)]
        self.image_ids = image_ids

        if not osp.isdir(self.in_xml_dir):
            os.makedirs(self.in_xml_dir)
        if not osp.isdir(self.out_dota_dir):
            os.makedirs(self.out_dota_dir)
        self.num_process = num_process
    
    def _convert(size, box):
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
    
    def _xml2dota(self,image_id):
        # 现在传进来的只有图片名没有后缀
        in_file = open(osp.join(self.in_xml_dir, image_id + '.xml'),'rb')
        out_file = open(osp.join(self.out_dota_dir,'%s.txt' % (image_id)), 'w+')

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
            #
            x_min=int(xmlbox.find('xmin').text)
            x_max=int(xmlbox.find('xmax').text)
            y_min=int(xmlbox.find('ymin').text)
            y_max=int(xmlbox.find('ymax').text)
            #p1-p2-p3-p4
            # p1=(x_min,y_min)
            # p2=(x_max,y_min)
            # p3=(x_min,y_max)
            # p4=(x_max,y_max)
            pt_loc=(x_min,y_min,x_max,y_min,x_min,y_max,x_max,y_max,cls)

            out_file.write(" ".join([str(a) for a in pt_loc])+ '\n')

    def run(self):
        with Pool(self.num_process) as p:
            p.map(self._xml2dota,self.image_ids)


if __name__ == "__main__":
    dir="./test"
    lable_dir=dir+"/dotalabels/"
    xml_dir=dir+"/xml/"

    xml2dota=Yoloxml2Dotatxt(xml_dir,lable_dir)
    xml2dota.run()
