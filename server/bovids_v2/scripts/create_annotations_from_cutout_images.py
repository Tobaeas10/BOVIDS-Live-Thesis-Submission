#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script that allows to convert a folder of images and their "cut out" individual images by e.g. GIMP
into a folder of the actual image and a corresponding .json file
as used to train the object detection / object segmentation network of BOVIDS

Use create_annotations_from_images() to run the method.
"""

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"


# make it possible to import own modules
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")

from lib.func import write_json_file_polygon, ensure_directory
from skimage import measure
from skimage.io import imread
from skimage.filters import threshold_triangle
from shutil import copy
import numpy as np

INPUT_FOLDER_RAW_IMAGES = "F:/Nextcloud/Dierkes/Ausschneiden_Photoshop/input/"
OUTPUT_FOLDER = "F:/Nextcloud/Dierkes/Ausschneiden_Photoshop/output/"

ORIGINAL_IMAGE_EXTENSION = "00"
INDIVIDUAL_NAME_MAPPING = {
    "01": "Giraffe_Arnheim_1",
    "02": "Giraffe_Arnheim_2",
    "03": "Giraffe_Arnheim_3",
    "04": "Giraffe_Arnheim_4",
    "05": "Giraffe_Arnheim_5",
    "06": "Giraffe_Arnheim_6",
    "07": "Giraffe_Arnheim_7",
    "08": "Giraffe_Arnheim_8",
}


def create_annotations_from_images():

    polygon_list, image_list, res_list = _fetch_image_list(INPUT_FOLDER_RAW_IMAGES)

    # write new images
    ensure_directory(OUTPUT_FOLDER + "images/")
    ensure_directory(OUTPUT_FOLDER + "labels/")

    for img_base_name, src_path in image_list.items():
        copy(src=src_path, dst=OUTPUT_FOLDER + "images/" + img_base_name + ".jpg")

    for img_base_name, contour_obj in polygon_list.items():
        write_json_file_polygon(
            output_path=OUTPUT_FOLDER + "labels/" + img_base_name + ".json",
            polygons=contour_obj,
            rel_image_path="../images/" + img_base_name + ".jpg",
            image_res=res_list[img_base_name],
        )
        # output_path, polygons, rel_image_path, image_res = (640,640)


def _get_polygon(image):
    img_2d = (imread(image, as_gray=True) * 255).astype(int)
    thresh = threshold_triangle(img_2d)
    binary_img = (img_2d > thresh).astype(int)
    contours = measure.find_contours(binary_img, 0)
    contour_sizes = [len(c) for c in contours]
    max_size = max(contour_sizes)
    largest_contour = contours[contour_sizes.index(max_size)]

    ret_contour = []
    for con in largest_contour:
        ret_contour.append((con[1], con[0]))

    shape = binary_img.shape

    return np.array(ret_contour), shape


def _fetch_image_list(path_name):
    """
    Returns a dictionary {main_image_name : source_path }
    and a dictionary {main_image_name: [ (ind_code, polygon) ] }
    and a dictionary {main_image_name : img_shape}
    """
    ret = {}
    polygon_list = {}
    shape_list = {}

    for path, subdirs, files in os.walk(path_name):
        for name in files:

            ind = len(ORIGINAL_IMAGE_EXTENSION) + 4
            base_name = name[: -1 * ind - 1]
            ending = name[-1 * ind : -4]

            if ending == ORIGINAL_IMAGE_EXTENSION:
                if base_name in ret.keys():
                    print(f"WARNING: Multiple images have the name {base_name}.")
                    continue
                ret[base_name] = path + name
                polygon_list[base_name] = []

            elif ending not in INDIVIDUAL_NAME_MAPPING.keys():
                print(f"WARNING: Invalid name {name}.")
                continue

            else:
                individual_name = INDIVIDUAL_NAME_MAPPING[ending]
                contour, shape = _get_polygon(path + name)

                shape_list[base_name] = (shape[1], shape[0])
                polygon_list[base_name].append((individual_name, contour))

    return polygon_list, ret, shape_list
