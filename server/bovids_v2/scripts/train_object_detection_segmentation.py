#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains methods to 
    - prepare and merge datasets to train detection and segmentation networks
        - converting .json into required YOLO format
        - merge datasets
        - renaming labels
    - train detection and segmentation networks
    
Methods:
    generate_training_set(): use to merge training sets and to rename labels, make YOLO format out of json
    train_neural_network(): use to actually train a neural network given a prepared training set
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from lib.func import (
    ensure_directory,
    write_yaml_file,
    get_polygons_from_json,
    polygon_to_bounding_box,
)

# import yolov8
from ultralytics import YOLO

# import standard libraries
import glob, json, shutil, random, pathlib
import numpy as np


# Preparation of datasets
RENAME_ALL_LABELS = ""  # if string, all labels will be called after that string, if empty string, this is ignored
RENAME_SPECIFIC_LABELS = {}  # dictionary old_label:new_label - might be useful
PREPARATION_MODE = "segment"  # detect, segment - in detection, bounding boxes are created out of segments

INPUT_FOLDERS = [
    "F:/Bov2Test/images_from_video_ohem_sort/2023-05-22-yolov8x/Saebelantilope_Leipzig_Test_1287_sz0/Yolov8x-seg/"
]  # list with paths to image folders with labels. Iterates through all files in the folder (and subdirectories) and matches .json files to existing images
OUTPUT_FOLDER_DATASET = "F:/Bov2Test/Testtraining-seq/"  #  path to the output folder: creates automatically /images/train/, /images/val/, /labels/train/, /labels/val/ and the .yaml file
VALIDATION_SPLIT = (
    0.3  # automatically leave out a fraction of images as a validation set if required
)


# Training of neural network
NETWORK_WEIGHTS = "F:/Nextcloud/Dierkes/bovids2/server/bovids_v2/res/imagesegmentation/yolov8x-seg.pt"  # path to basenetwork, will probably end with .pt
YAML_FILE = "F:/Bov2Test/Testtraining-seq/dataset_information.yaml"  # input yaml configuration file to train the network
NETWORK_OUTPUT_FOLDER = "F:/Bov2Test/od-netz/trainingsvorgang2/"  # path to the newly trained neural network, will contain checkpoints etc
NETWORK_OUTPUT_NAME = "somenetworkname-segment"  # i suggest "YYYYMMDD-SOMENAME-segment" or "YYYYMMDD-SOMENAME-detect"
TRAINING_MODE = "segment"  # detect, segment - take care that NETWORK_WEIGHTS is compatible to this mode!


NUM_EPOCHS = 2
OPTIMIZER = "AdamW"  # Adam, AdamW, or SGD
BATCH_SIZE = 8
DEVICE = "cpu"  # "0" / "cpu"


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


def generate_training_set():
    """
    Method to actually generate a training set for detection / segmentation
    It uses the input arguments to collect images and labels
        - internally swaps classnames and converts json to yolo format
    """
    ensure_directory(OUTPUT_FOLDER_DATASET + "images")
    ensure_directory(OUTPUT_FOLDER_DATASET + "labels")

    image_files = {}
    json_files = {}

    # extract all images and json files to dictionaries
    for inp_folder in INPUT_FOLDERS:
        d = pathlib.Path(inp_folder)
        for file_path in d.rglob("*"):
            filename = str(file_path)
            if os.path.isfile(filename):
                basename = os.path.basename(filename)
                if basename.endswith(".json"):
                    if basename[:-5] in json_files.keys():
                        print(
                            "ERROR: file exists multiple times. Please check!", filename
                        )
                        continue
                    json_files[basename[:-5]] = filename
                elif filename.endswith(".jpg"):
                    if basename[:-4] in image_files.keys():
                        print(
                            "ERROR: file exists multiple times. Please check!", filename
                        )
                        continue
                    image_files[basename[:-4]] = filename

    # choose images and files to work with
    use_files = []
    for basename, filepath in json_files.items():
        if not basename in image_files.keys():
            continue
        use_files.append([basename, filepath, image_files[basename]])

    # copy files to destination folder
    classnames = {}
    for basename, jsonpath, imagepath in use_files:
        polygons = get_polygons_from_json(jsonpath)
        if len(polygons) == 0:
            continue

        txt_lines = ""
        for clsname, polygon in polygons:

            if len(RENAME_ALL_LABELS) > 0:
                clsname = RENAME_ALL_LABELS
            elif clsname in RENAME_SPECIFIC_LABELS.keys():
                clsname = RENAME_SPECIFIC_LABELS[clsname]

            if not clsname in classnames.keys():
                if len(classnames.keys()) == 0:
                    classnames[clsname] = 0
                else:
                    classnames[clsname] = 1 + max(classnames.values())

            if PREPARATION_MODE == "detect":
                # prepare yolo object detection variant
                x_center, y_center, width, height = polygon_to_bounding_box(polygon)
                txt_lines += "{} {} {} {} {}\n".format(
                    classnames[clsname],
                    float(x_center) / 640,
                    float(y_center) / 640,
                    float(width) / 640,
                    float(height) / 640,
                )
            elif PREPARATION_MODE == "segment":
                # prepare segmentation variant
                txt_lines += str(classnames[clsname])
                for x, y in list(polygon):
                    txt_lines += " {} {}".format(float(x) / 640, float(y) / 640)
                txt_lines += "\n"

        trainval = "training"
        if random.random() < VALIDATION_SPLIT:
            trainval = "validation"
        # write txt file
        ensure_directory(
            "{}{}/labels/{}.txt".format(OUTPUT_FOLDER_DATASET, trainval, basename)
        )
        with open(
            "{}{}/labels/{}.txt".format(OUTPUT_FOLDER_DATASET, trainval, basename), "w+"
        ) as f:
            f.write(txt_lines)
        # write image file
        ensure_directory(
            "{}{}/images/{}.jpg".format(OUTPUT_FOLDER_DATASET, trainval, basename)
        )
        shutil.copy2(
            src=imagepath,
            dst="{}{}/images/{}.jpg".format(OUTPUT_FOLDER_DATASET, trainval, basename),
        )

        # create yaml file
        write_yaml_file(
            path=OUTPUT_FOLDER_DATASET,
            train="training/images/",
            val="validation/images/",
            clsnames=classnames,
            output_file=OUTPUT_FOLDER_DATASET + "dataset_information.yaml",
        )


def train_neural_network():

    if not os.path.exists(YAML_FILE):
        print("ERROR: {} does not exist.".format(YAML_FILE))
        return

    if len(NETWORK_WEIGHTS) > 0 and not os.path.exists(NETWORK_WEIGHTS):
        print("ERROR: {} does not exist.".format(NETWORK_WEIGHTS))
        return

    # Load a pretrained model oder an empty one
    if len(NETWORK_WEIGHTS) == 0:  # train from scratch:
        model = YOLO(task=TRAINING_MODE)
    else:
        model = YOLO(NETWORK_WEIGHTS, task=TRAINING_MODE)

    # Train the model
    model.train(
        data=YAML_FILE,
        epochs=NUM_EPOCHS,
        save_period=10,
        optimizer=OPTIMIZER,
        verbose=True,
        batch=BATCH_SIZE,
        exist_ok=True,
        device=DEVICE,
        project=NETWORK_OUTPUT_FOLDER,
        name=NETWORK_OUTPUT_NAME,
        imgsz=640,
    )

    # Validate on validation set
    metrics = model.val()
    print("MAP50-95:", metrics.box.maps)  # a list contains map50-95 of each category
