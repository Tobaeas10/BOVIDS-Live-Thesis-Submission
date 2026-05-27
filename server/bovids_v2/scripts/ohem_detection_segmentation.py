#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains methods to 
    - evaluate the segmentation / bounding boxes drawn by a neural network
    - sort the images and corresponding json labels into subfolders according to their given evaluation label
    - create a small statistics file given evaluation labels
    - create a standard dataset out of the sorted images/labels after adjusting bad/swapped labels with labelMe
    
Methods:
    - start_evaluation()
    - generate_statistics()
    - sort_images_by_evaluation()
    - create_training_dataset()
    
Important changes over version 1:
    - images without a created label can be shown and the evaluation status "missing" can be assigned
    - one can move to the next unlabelled / previous unlabelled image
"""

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"


# makes it possible to import own modules
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from lib.func import ensure_directory, get_polygons_from_json, polygon_to_bounding_box

# import standard libraries
import glob, json, shutil, random
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.draw import polygon, polygon_perimeter
from skimage.io import imread
import cv2


# input variables
### general
INPUT_BASE = "F:/Bov2Test/images_from_video/"  # folder that contains datasets per enclosure code ( e.g. enclosure_code_1/images and enclosure_code_1/labelfolder1, enclosure_code_1/labelfolder2, ... )
WHICH_FOLDERS = {
    "Saebelantilope_Leipzig_Test_1287_sz0": [
        "Yolov8x-seg"
    ],  # determines which label folders will be included in the set to evaluate, normally one does only want to show one folder at a time
}
BASENAME_EVALUATION = "2023-05-22-yolov8x"  # can be any name that determines the savefile name. the savefile will we called 'BASENAMEEVALUATION_enclosureID.csv', e.g. one could use a date or the label names


### sort_images_by_evaluation()
OUTPUT_FOLDER_EVALUATION = "F:/Bov2Test/images_from_video_ohem_sort/"  # output folder for sort_images_by_evaluation()

### create_training_dataset()
FOLDER_EVALUATED_IMAGES = "F:/Bov2Test/images_from_video_ohem_sort/2023-05-22-yolov8x/"  # normally it is the same as OUTPUT_FOLDER_EVALUATION with the BASENAME_EVALUATION appended
OUTPUT_FOLDER_DATASET = "F:/Bov2Test/images_from_video_ohem_training/ "  # output folder for create_training_dataset()
# COPY_WHICH_CLASSES is a dictionary of tuples. For each enclosure ID, we can say which labels (good, bad, swap, missing) after using labelMe to correct those labels
# from which subfolder (classifier) we want to copy. E.g: {'Saebelantilope_Leipzig_1' : ('od_network_name', ['good', 'bad', 'swap', 'missing'])}
# normally, we copy good, bad, swap, missing and the last three had been annotated by labelMe.
COPY_WHICH_CLASSES = {
    "Saebelantilope_Leipzig_Test_1287_sz0": (
        "Yolov8x-seg",
        ["good", "bad", "swap", "missing"],
    ),
}


### color codes for displaying boxes / segmentations {individual_id : color_code}
# color code (blue, green, red) ranging from 0 to 255
COLORCODES_INDIVIDUALS = {
    "default": (
        0,
        0,
        255,
    ),  # set the default to red - do not remove, adjustment of color is fine
    "bear": (255, 255, 0),
    "cat": (200, 0, 0),
    "sheep": (0, 200, 0),
}

COLORCODES_EVALUATION = {
    "none": (100, 100, 100),
    "good": (0, 180, 0),
    "medium": (0, 250, 250),
    "bad": (0, 0, 180),
    "swap": (180, 0, 0),
    "missing": (180, 0, 180),
}

# script control keys
MOVE_LEFT = "4"
MOVE_RIGHT = "5"
END_EVALUATION = "p"
MOVE_NEXT_UNLABELLED = "9"
MOVE_PREV_UNLABELLED = "8"

SET_GOOD = "a"
SET_MEDIUM = "s"
SET_BAD = "d"
SET_SWAPPED = "f"
SET_UNLABELLED = "u"
SET_MISSING = "m"

# if ... not in WHICH_FOLDERS[enccode]: continue

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


def start_evaluation():
    """
    Main method to actually perform the evaluation.
    - reads in existing savefiles or creates new ones (_update_evaluation_files)
    - creates GUI to evaluate
    """

    def _update_evaluation_file(enc_id, df):
        """
        - updates df according to images/labels avaiable - only in folders written in configuration part
        - if images are not avaiable anymore, row is deleted
        - if labels are not avaiable anymore, evaluation is overwritten
        - new images and labels are added
        """
        global INPUT_BASE, WHICH_FOLDERS
        if not os.path.exists(INPUT_BASE + enc_id):
            print("ERROR: path {} not existent.".format(INPUT_BASE + enc_id))
            return df
        if not os.path.exists(INPUT_BASE + enc_id + "/images"):
            print(
                "ERROR: path {} not existent.".format(INPUT_BASE + enc_id + "/images/")
            )
            return df

        # get all labels existing
        enc_folder = INPUT_BASE + enc_id + "/"
        images_folder = enc_folder + "images/"
        all_labels = {}
        ignore_classifiers = []
        for label_folder in WHICH_FOLDERS[enc_id]:
            if not os.path.exists(enc_folder + label_folder):
                print("ERROR: {} not existent for {}".format(label_folder, enc_id))
                print(
                    "WARNING: This means that no such labels are loaded. Please check that this is correct!"
                )
                ignore_classifiers.append(label_folder)
                continue
            for jsonfile in sorted(os.listdir(enc_folder + label_folder)):
                if not jsonfile.endswith("json"):
                    continue
                basename = jsonfile[:-5]
                if not basename in all_labels.keys():
                    all_labels[basename] = []
                all_labels[basename].append(label_folder)

        # get all images existing
        images = [
            f[:-4] for f in sorted(os.listdir(images_folder)) if f.endswith("jpg")
        ]

        # remove non existing images
        rows_to_del = [row for row in df.index if not df.loc[row, "basename"] in images]
        df = df.drop(rows_to_del)

        # set label to none if no json file is present
        for row in df.index:
            if not df.loc[row, "basename"] in all_labels.keys():
                df.loc[row, "evaluation"] = "none"
                continue
            if not df.loc[row, "label_subfolder"] in WHICH_FOLDERS[enc_id]:
                continue
            if (
                not df.loc[row, "label_subfolder"]
                in all_labels[df.loc[row, "basename"]]
            ):
                df.loc[row, "evaluation"] = "none"
                continue

        # add those that do not exist
        for basename in images:
            for label_folder in WHICH_FOLDERS[enc_id]:
                if label_folder in ignore_classifiers:
                    continue
                if (
                    len(
                        df[
                            (df["basename"] == basename)
                            & (df["label_subfolder"] == label_folder)
                        ]
                    )
                    > 0
                ):
                    continue
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "enclosure_id": [enc_id],
                                "basename": [basename],
                                "label_subfolder": [label_folder],
                                "evaluation": ["none"],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        return df

    def _add_shapes(curr_image_path, curr_json_path):
        """
        Gets the polygons from the json file (imported method)
        and draws a shape on the image given the configured class color
        """
        global COLORCODES_INDIVIDUALS

        # gets polygon and image
        img = imread(curr_image_path)
        polygons = get_polygons_from_json(curr_json_path)
        ret = img.copy()

        # adds polygon data to the image
        for classname, poly in polygons:
            rr, cc = polygon(poly[:, 1], poly[:, 0], img.shape)
            col = (
                COLORCODES_INDIVIDUALS["default"]
                if not classname in COLORCODES_INDIVIDUALS.keys()
                else COLORCODES_INDIVIDUALS[classname]
            )
            ret[rr, cc, :] = col

            xc, yc, w, h = polygon_to_bounding_box(poly)
            box = [int(xc - w / 2), int(xc + w / 2), int(yc - h / 2), int(yc + h / 2)]
            r = [box[0], box[1], box[1], box[0], box[0]]
            c = [box[3], box[3], box[2], box[2], box[3]]
            rr, cc = polygon_perimeter(
                c,
                r,
                img.shape,
            )
            ret[rr, cc] = col

        # creates image with polygon
        img_masked = (0.3 * ret + 0.7 * img).astype(np.uint8)
        return img_masked

    def _add_borders(img, evaluation):
        """
        Extends the image by a border visualising the given evaluation
        """
        global COLORCODES_EVALUATION

        # creation of image border
        img_border = np.zeros((700, 700, 3), np.uint8)
        img_border[0:700, 0:700] = COLORCODES_EVALUATION[evaluation]
        img_border[30:670, 30:670] = img

        return img_border

    def _save_changes(df, enc_id):
        """
        Saves the part of df into the savefile csv for the corresponding enclosure id.
        """
        global INPUT_BASE, BASENAME_EVALUATION
        df_save = df[df["enclosure_id"] == enc_id]
        df_save.to_csv(
            "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enc_id), index=False
        )

    # prepare all savefiles / make sure they exist
    for enclosure_id in WHICH_FOLDERS.keys():
        # checks if path to .csv file exists
        if not os.path.exists(
            "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id)
        ):
            # creates empty data frame, if path does not exist
            df = pd.DataFrame(
                {
                    "enclosure_id": [],
                    "basename": [],
                    "label_subfolder": [],
                    "evaluation": [],
                }
            )
        else:
            # reads csv file with predicted actions
            df = pd.read_csv(
                "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id)
            )
            # make a copy to be sure nothing unplanned happens...
            ensure_directory(
                "{}tmp/{}_{}-{}.csv".format(
                    INPUT_BASE,
                    BASENAME_EVALUATION,
                    enclosure_id,
                    datetime.now().strftime("%Y%m%d%H%M%S"),
                )
            )
            df.to_csv(
                "{}tmp/{}_{}-{}.csv".format(
                    INPUT_BASE,
                    BASENAME_EVALUATION,
                    enclosure_id,
                    datetime.now().strftime("%Y%m%d%H%M%S"),
                ),
                index=False,
            )
        df = _update_evaluation_file(enclosure_id, df)
        df.to_csv(
            "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id),
            index=False,
        )

    # prepare one pandas dataframe
    df = pd.DataFrame(
        {"enclosure_id": [], "basename": [], "label_subfolder": [], "evaluation": []}
    )
    for enclosure_id in WHICH_FOLDERS.keys():
        df_enc = pd.read_csv(
            "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id)
        )
        df_enc = df_enc[df_enc["label_subfolder"].isin(WHICH_FOLDERS[enclosure_id])]
        df = pd.concat([df, df_enc], ignore_index=True)
    df.reset_index().drop("index", axis=1)

    number_of_images = len(df.index)
    current_image_id = 0
    display = True
    if number_of_images == 0:
        return

    # creates window to show image and receive user input
    while display:
        # sets current image and json path
        curr_image_path = "{}{}/images/{}.jpg".format(
            INPUT_BASE,
            df.loc[current_image_id, "enclosure_id"],
            df.loc[current_image_id, "basename"],
        )
        curr_json_path = "{}{}/{}/{}.json".format(
            INPUT_BASE,
            df.loc[current_image_id, "enclosure_id"],
            df.loc[current_image_id, "label_subfolder"],
            df.loc[current_image_id, "basename"],
        )

        image_to_display = _add_shapes(curr_image_path, curr_json_path)
        image_to_display = _add_borders(
            image_to_display, df.loc[current_image_id, "evaluation"]
        )

        print(
            "{} - {} of {}.".format(
                df.loc[current_image_id, "basename"],
                current_image_id + 1,
                number_of_images,
            )
        )

        # creates window where the current image is shown
        cv2.imshow("Please evaluate the label.", image_to_display)

        # receives user input from keyboard
        key_pressed = False
        while not key_pressed:
            key = cv2.waitKey(0)
            # keyboard input to move between the images
            if key == ord(MOVE_LEFT) and current_image_id > 0:
                current_image_id -= 1
                key_pressed = True
            elif key == ord(MOVE_RIGHT) and current_image_id < number_of_images - 1:
                current_image_id += 1
                key_pressed = True
            elif key == ord(END_EVALUATION):
                display = False
                key_pressed = True
            elif key == ord(MOVE_NEXT_UNLABELLED):
                while current_image_id < number_of_images - 1:
                    current_image_id += 1
                    if df.loc[current_image_id, "evaluation"] == "none":
                        break
                key_pressed = True
            elif key == ord(MOVE_PREV_UNLABELLED):
                while current_image_id > 0:
                    current_image_id -= 1
                    if df.loc[current_image_id, "evaluation"] == "none":
                        break
                key_pressed = True

            # keyboard input to set and save user evaluation for images
            elif key == ord(SET_SWAPPED):
                df.loc[current_image_id, "evaluation"] = "swap"
                _save_changes(df, df.loc[current_image_id, "enclosure_id"])
                key_pressed = True
            elif key == ord(SET_UNLABELLED):
                df.loc[current_image_id, "evaluation"] = "none"
                _save_changes(df, df.loc[current_image_id, "enclosure_id"])
                key_pressed = True
            elif key == ord(SET_MISSING):
                df.loc[current_image_id, "evaluation"] = "missing"
                _save_changes(df, df.loc[current_image_id, "enclosure_id"])
                key_pressed = True
            elif key == ord(SET_GOOD):
                df.loc[current_image_id, "evaluation"] = "good"
                _save_changes(df, df.loc[current_image_id, "enclosure_id"])
                key_pressed = True
            elif key == ord(SET_MEDIUM):
                df.loc[current_image_id, "evaluation"] = "medium"
                _save_changes(df, df.loc[current_image_id, "enclosure_id"])
                key_pressed = True
            elif key == ord(SET_BAD):
                df.loc[current_image_id, "evaluation"] = "bad"
                _save_changes(df, df.loc[current_image_id, "enclosure_id"])
                key_pressed = True
            elif key == ord(END_EVALUATION):
                display = False
                key_pressed = True
    cv2.destroyAllWindows()


def create_training_dataset():
    """
    After application of sort_images_by_evaluation and by manually annotating bad examples,
    this procedure allows to merge the single datasets to one dataset.
    """
    ensure_directory(OUTPUT_FOLDER_DATASET + "images/")
    ensure_directory(OUTPUT_FOLDER_DATASET + "labels/")
    for enclosure_id in COPY_WHICH_CLASSES.keys():
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), enclosure_id)
        classifier, which_folders = COPY_WHICH_CLASSES[enclosure_id]

        for evaluation in which_folders:
            curr_images = "{}{}/{}/{}/images/".format(
                FOLDER_EVALUATED_IMAGES, enclosure_id, classifier, evaluation
            )
            curr_labels = "{}{}/{}/{}/labels/".format(
                FOLDER_EVALUATED_IMAGES, enclosure_id, classifier, evaluation
            )
            if not os.path.exists(curr_images):
                print("ERROR: not found {}".format(curr_images))
                continue
            if not os.path.exists(curr_labels):
                print("ERROR: not found {}".format(curr_labels))
                continue
            for image in os.listdir(curr_images):
                if not image.endswith(".jpg"):
                    continue
                labelfile = curr_labels + image[:-4] + ".json"
                if not os.path.exists(labelfile):
                    print(
                        "WARNING: label for image not found {}".format(
                            curr_images + image
                        )
                    )
                    continue
                shutil.copy2(
                    curr_images + image, OUTPUT_FOLDER_DATASET + "images/" + image
                )
                shutil.copy2(
                    labelfile, OUTPUT_FOLDER_DATASET + "labels/" + image[:-4] + ".json"
                )
    print("Finished processing.")


def sort_images_by_evaluation():
    """
    Given the human evaluation (as csv), creates /BASENAME_EVALUATION/good/ etc
    as datasets that, after using labelme, can be used to train a new network
    """
    for enclosure_id in WHICH_FOLDERS.keys():
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), enclosure_id)
        if not os.path.exists(
            "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id)
        ):
            print(
                "WARNING: Path does not exist.",
                "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id),
            )
            continue
        else:
            df = pd.read_csv(
                "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id)
            )

        for classifier in WHICH_FOLDERS[enclosure_id]:
            df_cls = df[df["label_subfolder"] == classifier]
            if len(df_cls.index) == 0:
                print(
                    "WARNING: For {} there is no classifier {}.".format(
                        enclosure_id, classifier
                    )
                )
                continue

            for current_image_id in df_cls.index:
                curr_image_path = "{}{}/images/{}.jpg".format(
                    INPUT_BASE,
                    df_cls.loc[current_image_id, "enclosure_id"],
                    df_cls.loc[current_image_id, "basename"],
                )
                curr_json_path = "{}{}/{}/{}.json".format(
                    INPUT_BASE,
                    df_cls.loc[current_image_id, "enclosure_id"],
                    df_cls.loc[current_image_id, "label_subfolder"],
                    df_cls.loc[current_image_id, "basename"],
                )

                if not os.path.exists(curr_image_path):
                    print("WARNING: image not found {}".format(curr_image_path))
                    continue

                if not os.path.exists(curr_json_path) and not df_cls.loc[
                    current_image_id, "evaluation"
                ] in ["missing", "none"]:
                    print("WARNING: json not found {}".format(curr_json_path))
                    continue

                if not os.path.exists(curr_json_path) and df_cls.loc[
                    current_image_id, "evaluation"
                ] in ["missing", "none"]:
                    dst_img = "{}{}/{}/{}/{}/images/{}.jpg".format(
                        OUTPUT_FOLDER_EVALUATION,
                        BASENAME_EVALUATION,
                        enclosure_id,
                        classifier,
                        df_cls.loc[current_image_id, "evaluation"],
                        df_cls.loc[current_image_id, "basename"],
                    )
                    ensure_directory(dst_img)
                    shutil.copy2(curr_image_path, dst_img)
                else:
                    dst_img = "{}{}/{}/{}/{}/images/{}.jpg".format(
                        OUTPUT_FOLDER_EVALUATION,
                        BASENAME_EVALUATION,
                        enclosure_id,
                        classifier,
                        df_cls.loc[current_image_id, "evaluation"],
                        df_cls.loc[current_image_id, "basename"],
                    )
                    dst_json = "{}{}/{}/{}/{}/labels/{}.json".format(
                        OUTPUT_FOLDER_EVALUATION,
                        BASENAME_EVALUATION,
                        enclosure_id,
                        classifier,
                        df_cls.loc[current_image_id, "evaluation"],
                        df_cls.loc[current_image_id, "basename"],
                    )
                    ensure_directory(dst_img)
                    ensure_directory(dst_json)
                    shutil.copy2(curr_image_path, dst_img)
                    shutil.copy2(curr_json_path, dst_json)
    print("Finished processing.")


def generate_statistics():
    """
    Outputs an xlsx file which contains an overview per enclosure_id and per classifier
    Contains information on how many labels found and how they were evaluated
    """

    def _get_stats(df):
        """
        Returns 'Images': int, 'good' : int, 'medium': int, 'bad': int, 'swap': int, 'missing': int, 'none' : int
        """
        ret = {
            evaluation: len(df[df["evaluation"] == evaluation].index)
            for evaluation in COLORCODES_EVALUATION.keys()
        }
        ret["Images"] = len(df.index)

        return ret

    df_ret = {
        "Enclosure": [],
        "Classifier": [],
        "Images": [],
        "good": [],
        "medium": [],
        "bad": [],
        "swap": [],
        "missing": [],
        "none": [],
    }

    for enclosure_id in WHICH_FOLDERS.keys():
        if not os.path.exists(
            "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id)
        ):
            print(
                "WARNING: Path does not exist.",
                "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id),
            )
            continue
        else:
            df = pd.read_csv(
                "{}{}_{}.csv".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id)
            )

        for classifier in WHICH_FOLDERS[enclosure_id]:
            df_cls = df[df["label_subfolder"] == classifier]
            if len(df_cls.index) == 0:
                print(
                    "WARNING: For {} there is no classifier {}.".format(
                        enclosure_id, classifier
                    )
                )
                continue

            tmp = _get_stats(df_cls)
            for k, v in tmp.items():
                df_ret[k].append(v)
            df_ret["Enclosure"].append(enclosure_id)
            df_ret["Classifier"].append(classifier)

    pd.DataFrame(df_ret).to_excel(
        "{}{}_{}_statistics.xlsx".format(INPUT_BASE, BASENAME_EVALUATION, enclosure_id),
        index=False,
    )
