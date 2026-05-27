#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert the old XML labels and datasets for YOLOv4 to the novel YOLOv8 format.
"""

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

import warnings

warnings.filterwarnings("ignore")

# make it possible to import own modules
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from lib.func import ensure_directory, write_json_file_boundingbox_as_polygon

from xml.etree import ElementTree
from skimage.transform import resize  # scikit-image
from skimage.io import imread, imsave
from skimage.color import gray2rgb
import numpy as np
from pathlib import Path
from tqdm import tqdm

OUTPUT_PATH = "F:/Bov2Test/xmlconvert/"  # erstellt einen ordner images und einen ordner labels_converted in das er alle der bilder packt, die unten angegeben werden (die bilder werden kopiert!)
INPUT_PATHS = [  # liste von zuordnungen, sodass in "images" die Bilder liegen und in "labels" die zugehörigen label (xml dateien, wie früher)
    {
        "images": "F:/Bov2Test/Daten_Melina/xml/Bilder/Nashorn-Panzer/Amersfoort/1/",
        "labels": "F:/Bov2Test/Daten_Melina/xml/Label/Nashorn-Panzer/Amersfoort/1/",
    },
    {"images": "", "labels": ""},
    {"images": "", "labels": ""},
    {"images": "", "labels": ""},
    {"images": "", "labels": ""},
    {"images": "", "labels": ""},
]


def convert_label():

    for dataset in INPUT_PATHS:
        img_folder = dataset["images"]
        label_folder = dataset["labels"]

        if not os.path.exists(label_folder) or not os.path.exists(img_folder):
            print("ERROR: unknown paths", img_folder, label_folder)
            continue
        for labelfile in tqdm(os.listdir(label_folder), desc=label_folder):

            filetype = Path(labelfile).suffix
            if not filetype == ".xml":
                continue

            basename = Path(labelfile).stem
            img_path = img_folder + basename + ".jpg"

            if not os.path.exists(img_path):
                continue

            _convert_one_image(
                imgpath=img_path,
                xmlpath=label_folder + labelfile,
                output_folder=OUTPUT_PATH,
            )


def _get_boxes_from_xml(label_file, xrescale, yrescale, xadd, yadd):
    """
    Gets a path to a yolov4 xml label file as an input
    returns [ [classname, [xmin, ymin, xmax, ymax]] ]
    """
    ret = []
    tree = ElementTree.parse(
        label_file,
    )
    root = tree.getroot()
    obj_list = root.findall("object")
    for obj in obj_list:
        name = obj.find("name")
        ind_name = name.text.rstrip("\r\n")
        for box in obj.iter("bndbox"):

            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
            coors = [
                int(xmin * xrescale) + xadd,
                int(ymin * yrescale) + yadd,
                int(xmax * xrescale) + xadd,
                int(ymax * yrescale) + yadd,
            ]

            xcenter = int((coors[0] + coors[2]) / 2)
            ycenter = int((coors[1] + coors[3]) / 2)
            w = xrescale * (xmax - xmin)
            h = yrescale * (ymax - ymin)
            ret.append([ind_name, [xcenter, ycenter, w, h]])

    return ret


def _read_image(imagepath, h=640, w=640):
    """
    returns the image but rescaled to 640 times 640 by adding black borders
    also return height factor and width factor
    """
    image = imread(imagepath)
    if len(image.shape) == 2:
        image = gray2rgb(image)

    img_h, img_w, _ = image.shape
    ret = np.zeros((h, w, 3))

    if img_h / h > img_w / w:
        # img height = 640
        # img width = int(img_w*h/img_h))

        img_r = resize(image, (h, int(img_w * h / img_h)))
        missing_width = w - int(img_w * h / img_h)
        if missing_width > 0:
            ret[
                :, int(missing_width / 2) : int(missing_width / 2) + img_r.shape[1], :
            ] = img_r

        hscale = h / img_h
        wscale = int(img_w * h / img_h) / img_w
        hadd = 0
        wadd = int(missing_width / 2)

    else:
        img_r = resize(image, (int(img_h * w / img_w), w))
        missing_height = h - int(img_h * w / img_w)
        if missing_height > 0:
            ret[
                int(missing_height / 2) : int(missing_height / 2) + img_r.shape[0], :, :
            ] = img_r

        hscale = int(img_h * w / img_w) / img_h
        wscale = w / img_w
        hadd = int(missing_height / 2)
        wadd = 0

    ret = (255 * ret).astype(np.uint8)
    return ret, hscale, hadd, wscale, wadd


def _convert_one_image(imgpath, xmlpath, output_folder):
    """
    Given a path to an image, and to an old label file, it converts the image to 640x640 and saves it at a new location.
    Moreover, the method creates a json label file accordingly.
    """
    img, hscale, hadd, wscale, wadd = _read_image(imgpath)
    bboxes = _get_boxes_from_xml(xmlpath, wscale, hscale, wadd, hadd)

    basename = Path(imgpath).stem
    rng = int(1000000 * np.random.random())
    json_path = f"{output_folder}labels_converted/{basename}+{rng}.json"
    image_path = f"{output_folder}images/{basename}+{rng}.jpg"

    ensure_directory(json_path)
    ensure_directory(image_path)
    imsave(image_path, img)
    write_json_file_boundingbox_as_polygon(
        json_path, bboxes, f"../images/{basename}+{rng}.jpg", (640, 640)
    )


convert_label()
