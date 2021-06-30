# detection-data-processing
深度学习目标检测标注文件不同格式的相互转换：
1.calc_objcount.py ：统计当前标注文件中标注目标的数量
2.createimages.py：利用目标和背景文件生成新的训练数据
3.dotatxt2yoloxml.py:将dota格式的数据转化成yolo xml格式的数据
4.get_annotataions_obj.py：获取标注好的yolo xml文件中的目标小图和移除目标后的背景图
5.yolo_image_split.py：将dota标注格式的文件，按照切分大小和重叠度生成新的数据
6.yoloxml2cocojson.py :将yolo xml格式的数据，生成coco json格式的文件
7.yoloxml2dotatxt.py：将yolo xml格式的数据，转化为dota格式的数据
8.yoloxml2yolotxt.py: 将yolo xml格式的数据，转化为yolo txt格式的数据

代码进一步优化中


# DOTA 解释文件的格式

DOTA 数据格式：
![image](https://github.com/edificewang/detection-data-processing/raw/main/doc/dota.png)

Yolo xml 数据转化为DOTA txt数据格式流程：
![image](https://github.com/edificewang/detection-data-processing/raw/main/doc/dataprocess.png)
