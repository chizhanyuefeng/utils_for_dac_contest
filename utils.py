# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import xml.dom.minidom as xml_parser


def load_imglst(img_dir):
    """
    加载图片，并返回每个图片的路径
    :param img_dir: 存储图片的根路径
    :return: 返回每个图片文件的绝对路径和文件名（不包含后缀.jpg）
    """
    file_name_lst = [pic for pic in os.listdir(img_dir) if '.jpg' in pic]
    # 按照文件名顺序进行排序
    file_name_lst.sort()
    # 每个图片的绝对路径
    img_lst = [os.path.join(img_dir,filename) for filename in file_name_lst]

    # 去掉图片文件名后缀。例如图片文件名为0001.jpg，去掉后缀后为0001
    for i in range(len(file_name_lst)):
        file_name_lst[i]=file_name_lst[i].strip(".jpg")

    return img_lst,file_name_lst


def load_bbox_from_xml(xml_file_dir):
    """
    从一个xml文件中加载每个图片的bounding box
    一般用于加载ground truth
    :param xml_file_dir: xml文件的绝对路径
    :return: 返回读取的bounding box
    """
    bbox = []
    dom_tree = xml_parser.parse(xml_file_dir)
    root = dom_tree.documentElement

    xmin = root.getElementsByTagName("xmin")[0].firstChild.data
    ymin = root.getElementsByTagName("ymin")[0].firstChild.data
    xmax = root.getElementsByTagName("xmax")[0].firstChild.data
    ymax = root.getElementsByTagName("ymax")[0].firstChild.data

    bbox.append(xmin)
    bbox.append(ymin)
    bbox.append(xmax)
    bbox.append(ymax)

    bbox = np.array(bbox)
    bbox = bbox.astype('float32').astype('int')
    return bbox

def load_bboxlist_from_xml(file_dir):
    """
    加载N个xml文件，从中读取bounding box。
    最后返回的是一个字典。字典每个值为[key = file_name,value = bbox]。
    :param file_dir: xml文件的绝对路径
    :return: 返回生成的字典
    """
    dict = {}

    # 读取file_dir里面的所有xml文件
    file_list = [xml_file for xml_file in os.listdir(file_dir) if '.xml' in xml_file]
    file_list.sort()

    # xml 列表 [[file_dir,file_name],[file_dir,file_name]....]
    xml_list = []

    for file_name in file_list:
        xml_list.append([os.path.join(file_dir, filename),file_name])

    for bbox_xml in xml_list:
        # 从file_dir中读取xml文件bounding box
        bbox = load_bbox_from_xml(bbox_xml[0])
        # 存储到字典中，key为xml文件名字，value为bounding box
        dict[bbox_xml[1]] = bbox

    return dict

def computeIOU(groud_coord,result_coord):
    """
    计算groud truth和result bbox的IoU数值
    :param groud_coord: 真实样本bbox
    :param result_coord: 输出结果bbox
    :return: 返回计算的IoU数值
    """
    x_gt = [groud_coord[0],groud_coord[2]]
    y_gt = [groud_coord[1],groud_coord[3]]
    x_test = [result_coord[0],result_coord[2]]
    y_test = [result_coord[1],result_coord[3]]
    x_gt.sort()
    y_gt.sort()
    x_test.sort()
    y_test.sort()
    if(x_gt[0] >= x_test[1] or y_gt[0] >= y_test[1] or x_gt[1] <= x_test[0] or y_gt[1] <= y_test[0]):
        return 0
    X = [max(x_gt[0],x_test[0]),min(x_gt[1],x_test[1])]
    Y = [max(y_gt[0],y_test[0]),min(y_gt[1],y_test[1])]
    cross_area = float(computeArea(X,Y))
    gt_area = computeArea(x_gt,y_gt)
    test_area = computeArea(x_test,y_test)
    return cross_area/(gt_area+test_area-cross_area)

def computeArea(X,Y):
    return abs(X[0]-X[1])*abs(Y[0]-Y[1])

def write_xml(file_name,image_size,bounding_box):
    """
    将计算的输出结果存储在xml文件中
    :param file_name: 存储的文件名（不包含后缀.xml）
    :param image_size: 图片的大小
    :param bounding_box: 输出的bounding_box
    :return: 无
    """
    filename_write = file_name + ".xml"
    f = open(filename_write, "w")
    doc = xml_parser.Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)

    filename = doc.createElement("filename")
    annotation.appendChild(filename)
    video_filename = doc.createTextNode(file_name)
    filename.appendChild(video_filename)

    size = doc.createElement("size")
    annotation.appendChild(size)

    width = doc.createElement("width")
    size.appendChild(width)
    image_weight = doc.createTextNode(str(image_size[0]))
    width.appendChild(image_weight)

    height = doc.createElement("height")
    size.appendChild(height)
    image_height = doc.createTextNode(str(image_size[1]))
    height.appendChild(image_height)

    object = doc.createElement("object")
    annotation.appendChild(object)

    bndbox = doc.createElement("bndbox")
    object.appendChild(bndbox)

    xmin = doc.createElement("xmin")
    bndbox.appendChild(xmin)
    x_min = doc.createTextNode(str(bounding_box[0]))
    xmin.appendChild(x_min)

    ymin = doc.createElement("ymin")
    bndbox.appendChild(ymin)
    y_min = doc.createTextNode(str(bounding_box[1]))
    ymin.appendChild(y_min)

    xmax = doc.createElement("xmax")
    bndbox.appendChild(xmax)
    x_max = doc.createTextNode(str(bounding_box[2]))
    xmax.appendChild(x_max)

    ymax = doc.createElement("ymax")
    bndbox.appendChild(ymax)
    y_max = doc.createTextNode(str(bounding_box[3]))
    ymax.appendChild(y_max)

    doc.writexml(f, addindent='\t', newl='\n', encoding="utf-8")


def evalue_accuracy(groud_file_path,result_file_path):
    """
    计算算法准确率
    注：标签文件和计算结果文件要求文件名相同，路径不同
    :param groud_file_path: 标签文件路径w
    :return: 返回算法的准确率
    """
    ground_dict = load_bboxlist_from_xml(groud_file_path)
    result_dict = load_bboxlist_from_xml(result_file_path)

    groud_keys = ground_dict.keys()
    result_keys = result_dict.keys()

    result_IoU = []
    count = min(len(groud_keys),len(result_keys))


    for i in range(count):
        key = result_keys[i]
        if ground_dict.has_key(key):
            iou=computeIOU(ground_dict.get(key),result_dict.get(key))
            result_IoU.append(iou)

    accuracy = sum(result_IoU)/len(result_IoU)

    return accuracy











