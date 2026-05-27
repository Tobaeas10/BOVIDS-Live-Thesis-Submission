# -*- coding: utf-8 -*-
"""
Contains methods to 
    - prepare and merge datasets to the different Action Classifiers
        - given BORIS annotations: return balanced images from video files
        - a dataset is always of the form {MODE}/{action_label}/{enclosure_id}/images/0/image_file.jpg
        - create and merge existing datasets (from BORIS and OHEM)
    - train action classifiers for three tasks
        - standing / lying
        - LHU / LHD
        - standing_no_food / food
    - creates automatically both representations of one time interval 
    (one classifier per task)
    
Methods:
    generate_training_set()
        Given the dataset (one dataset) DATASET_PATH, it creates a training and validation set at
        TRAINING_VALIDATION_PATH using VALIDATION_SPLIT as the proportional size of the validation set (0.15-0.25 suggested!)
        Does auto-balance the classes. Can be disables by upsampling=False as an argument (use with care)
    
    create_dataset_from_videos()
        Method to create a novel dataset to train action classifiers from annotated video files.
            INPUT_XLSX_DATASETCREATION: uses to excel file containing dates and enclosures that will be used
            OUTPUT_PATH_NEW_DATASETCREATION: path in which the new dataset is created
        Global variables used (from this script file):
            ANCHORPATH_BORIS_ANNOTATIONS directory such that config/boris_information.txt contains the relative path to the boris annotation files
            ANCHORPOINT_VIDEOFILES directory such that enclosure_information contains the relative path to the videofiles
            OBJECT_DETECTION_MODE detect or segment, datasets should not be merged, train different classifiers! requires that for all used enclosures a respective network exists.
            MAXIMUM_NUMER_SAMPLES_PER_VIDEO caps the number of time intervals (balanced) sampled from one video
            REMOVE_TEMPORARY_FILES if false, the samples (intervals) and corresponding difference images as a whole enclosure image are not deleted
        IO Output:
            creates {out}{MODE}/{action_label}/{individual_id}/images/0/ for every used action label (e.g. 1 and 2 for StLy or 11 and 12 for StFo) and any individual_id
     
    merge_datasets()
        Gets a list of multiple datasets as well as the information which fraction of which class should be taken for each dataset (MERGE_DATASETS)
        Outputs (IO) a joint dataset into  OUTPUT_PATH_MERGED_DATASET
        If AUTO_BALANCE is enabled, it upsamples the images such that classes are balanced. Slight augmentations are applied.
        
    train_actionclassifier()
"""

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

# make it possible to import own modules
import os, sys, shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from lib.image_manipulation import (
    augment_action_classifier_image,
    IO_video_timeinterval,
    save_image_to_file,
)
from lib.func import get_videoname, ensure_directory, get_boris_annotation
from lib.object_detection import predict_folder_differences
from config.get_config import (
    get_enclosure_information,
    BEHAVIORS_BY_MODE,
    use_frames_per_interval,
    SECONDS_PER_INTERVAL,
)
from lib.func import read_inputfile_datasetcreation

import numpy as np
from torchvision.io import read_image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torchvision.transforms as T
import torch, pickle, os
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm.contrib.concurrent import thread_map, cpu_count
from tqdm import tqdm

from datetime import datetime
from skimage.io import imsave

###################

"""
# general configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # needs to be adjusted
MODE = 'StLy' # StLy = Standing / Lying, LHULHD = LHD/LHU, StFo = Standing / Food # CAUTION: needs to be adjusted for creating datasets, merging them, and to train a network!

# Creation of a novel dataset
INPUT_XLSX_DATASETCREATION = 'G:/BOVIDS2_BACKUP/2023-10-24/server/bovids_v2/stuff/ac_test.xlsx'
OUTPUT_PATH_NEW_DATASETCREATION = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-Gamma30/'
ANCHORPATH_BORIS_ANNOTATIONS = 'G:/BOVIDS2_DATA/Auswertung/' # directory such that config/boris_information.txt contains the relative path to the boris annotation files
ANCHORPOINT_VIDEOFILES = 'G:/BOVIDS2_DATA/Videomaterial/' #directory such that enclosure_information contains the relative path to the videofiles
OBJECT_DETECTION_MODE = 'detect' # detect or segment, datasets should not be merged, train different classifiers! requires that for all used enclosures a respective network exists.

MAXIMUM_NUMER_SAMPLES_PER_VIDEO = 30 # caps the number of time intervals (balanced) sampled from one video
REMOVE_TEMPORARY_FILES = False # True or False: if false, the samples (intervals) and corresponding difference images as a whole enclosure image are not deleted

MERGE = False # True or False: if false, you dont want to merge datasets
# Merging of existing datasets
MERGE_DATASETS = {
    'datasets': [ 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu/', 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-Klein/', 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-Gamma/' ],
    'fraction_images_per_class': [{1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0}],
    'dismiss_individuals': [ [], [], [] ]
    }
# list of dictionaries that determines which fraction of images of a specific class should be taken from a specific dataset. if the class does not exist, no images are taken from this class
# can be used to either balance classes after ohem or to take only a part of a dataset to not overfit the network with too many hard examples (or something similar)
# a path to a dataset is a directory such that MODE/action_label/enclosure_id/images/0/image.jpg is the structure of the directory.
AUTO_BALANCE = True # if True, balances classes by upsampling with augmentation after merging datasets
OUTPUT_PATH_MERGED_DATASET = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-merged/' 

# preparing a training and validation set from a dataset
DATASET_PATH = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-merged/' 
TRAINING_VALIDATION_PATH = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-Trainingset/' 
VALIDATION_SPLIT = 0.2

# training of an action classifier
DATA_DIR_TRAINING_AND_VALIDATION = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-Trainingset/StLy/'  # contains the folders train/ and val/
PRETRAINED_NETWORK = '' # .pth file that can be used as a base network. if nothing is given, imagenet pretrained network will be chosen

MODELNAME = '2023-12-15_just_one_try-StLy' # e.g. 2023-07-12_StLy_Bovids
SAVEPATH_MODEL = 'G:/BOVIDS2_TEST/try-training2/' # directory in which the model, the training history and the checkpoints will be saved
"""

# general configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # needs to be adjusted
MODE = "StFo"  # StLy = Standing / Lying, LHULHD = LHD/LHU, StFo = Standing / Food # CAUTION: needs to be adjusted for creating datasets, merging them, and to train a network!

# Creation of a novel dataset
# Judith

INPUT_XLSX_DATASETCREATION = (
    "C:/Users/Judith/Uni/HIWI/BOVIDS2/server/bovids_v2/stuff/ac_test.xlsx"
)
OUTPUT_PATH_NEW_DATASETCREATION = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/Zebra-Steppen-merged/"
# OUTPUT_PATH_NEW_DATASETCREATION = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-Gamma30/'
ANCHORPATH_BORIS_ANNOTATIONS = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/boris_data/"  # directory such that config/boris_information.txt contains the relative path to the boris annotation files
ANCHORPOINT_VIDEOFILES = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/"  # directory such that enclosure_information contains the relative path to the videofiles
OBJECT_DETECTION_MODE = "detect"  # detect or segment, datasets should not be merged, train different classifiers! requires that for all used enclosures a respective network exists.

MERGE = True  # True or False: if false, you dont want to merge datasets


MAXIMUM_NUMER_SAMPLES_PER_VIDEO = (
    30  # caps the number of time intervals (balanced) sampled from one video
)
REMOVE_TEMPORARY_FILES = False  # True or False: if false, the samples (intervals) and corresponding difference images as a whole enclosure image are not deleted


# Merging of existing datasets
MERGE_DATASETS = {
    "datasets": [
        r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/Zebra-Steppen-3/",
        r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/Zebra-Steppen/",
    ],
    "fraction_images_per_class": [{10: 1.0, 11: 1.0}, {10: 1.0, 11: 1.0}],
    "dismiss_individuals": [[], []],
}

# list of dictionaries that determines which fraction of images of a specific class should be taken from a specific dataset. if the class does not exist, no images are taken from this class
# can be used to either balance classes after ohem or to take only a part of a dataset to not overfit the network with too many hard examples (or something similar)
# a path to a dataset is a directory such that MODE/action_label/enclosure_id/images/image.jpg is the structure of the directory.
AUTO_BALANCE = True  # if True, balances classes by upsampling with augmentation after merging datasets
OUTPUT_PATH_MERGED_DATASET = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/Zebra-Steppen-merged/"

# preparing a training and validation set from a dataset
DATASET_PATH = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/Zebra-Steppen-merged/"
# DATASET_PATH = 'C:/Users/lmoel/OneDrive/Dokumente/Lea/BOVIDS/ac_dataset_create/'
# DATASET_PATH = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-merged/'
# TRAINING_VALIDATION_PATH = 'G:/BOVIDS2_TEST/ac_dataset_create/Kudu-Trainingset/'
TRAINING_VALIDATION_PATH = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/Zebra-Steppen-merged-Trainingsset/"
# TRAINING_VALIDATION_PATH = 'C:/Users/lmoel/OneDrive/Dokumente/Lea/BOVIDS/ac_dataset_create/'
VALIDATION_SPLIT = 0.2

DATA_DIR_TRAINING_AND_VALIDATION = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/Zebra-Steppen-merged-Trainingsset/StFo/"  # contains the folders train/ and val/
PRETRAINED_NETWORK = ""  # .pth file that can be used as a base network. if nothing is given, imagenet pretrained network will be chosen
MODELNAME = "2024-08-02_StFo_Bovids-merged"  # e.g. 2023-07-12_StLy_Bovids
SAVEPATH_MODEL = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_network/merged/"  # directory in which the model, the training history and the checkpoints will be saved

NUM_EPOCHS = 3
BATCH_SIZE = 16
SAVE_EVERY_EPOCH = 3


######################################
######################################
######################################
######################################
######################################
######################################
######################################


#####################################
### class names used in training ####
#####################################
CLASS_NAMES = list(BEHAVIORS_BY_MODE[MODE].keys())
CLASS_INDICES = list(BEHAVIORS_BY_MODE[MODE].values())


################################################
## code to create and merge training datasets ##
################################################


def create_dataset_from_videos(
    f=INPUT_XLSX_DATASETCREATION, out=OUTPUT_PATH_NEW_DATASETCREATION
):
    """
    Method to create a novel dataset to train action classifiers from annotated video files.
        f: uses to excel file containing dates and enclosures that will be used
        out: path in which the new dataset is created
    Global variables used (from this script file):
        ANCHORPATH_BORIS_ANNOTATIONS directory such that config/boris_information.txt contains the relative path to the boris annotation files
        ANCHORPOINT_VIDEOFILES directory such that enclosure_information contains the relative path to the videofiles
        OBJECT_DETECTION_MODE detect or segment, datasets should not be merged, train different classifiers! requires that for all used enclosures a respective network exists.
        MAXIMUM_NUMER_SAMPLES_PER_VIDEO caps the number of time intervals (balanced) sampled from one video
        REMOVE_TEMPORARY_FILES if false, the samples (intervals) and corresponding difference images as a whole enclosure image are not deleted
    IO Output:
        creates {out}{MODE}/{action_label}/{individual_id}/images/0/ for every used action label (e.g. 1 and 2 for StLy or 11 and 12 for StFo) and any individual_id
    """

    def get_balanced_intervals(sequence, possible_behaviors):
        """
        Given a sequence of integers (behavior time intervals),
        returns per class the indices that should be taken such that
        we have balanced classes
        """

        def get_indices(sequence, value):
            indexes = []
            i = -1
            while True:
                try:
                    i = sequence.index(value, i + 1)
                    indexes.append(i)
                except ValueError:
                    break
            return indexes

        tmp = {val: get_indices(sequence, val) for val in possible_behaviors}
        min_size = min(
            MAXIMUM_NUMER_SAMPLES_PER_VIDEO, min([len(x) for x in tmp.values()])
        )

        ret = {
            val: sorted(np.random.choice(tmp[val], size=min_size, replace=False))
            for val in tmp.keys()
        }
        return ret

    to_process = read_inputfile_datasetcreation(f)
    df_enc_info = get_enclosure_information(to_process.keys())

    video_to_process = []

    for enclosure_id, enc_dict in to_process.items():

        if not enclosure_id in df_enc_info.index:
            print("ERROR: no configuration available for {}".format(enclosure_id))
            continue
        else:
            print(
                "{}: Creating images for {}".format(
                    datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), enclosure_id
                )
            )

        dates = enc_dict["dates"]
        desired_starts = enc_dict["desired_starts"]
        desired_ends = enc_dict["desired_ends"]
        dismissed_individuals = enc_dict["dismissed_individuals"]

        if dismissed_individuals == ["nan"]:
            dismissed_individuals = []

        # Step 1: extract images per video (preparation such that this will be conducted in parallel)
        ## saves temporary data to out/tmp/action_id/enclosure_id/yyyymmdd-mmhhss_enclosure_id.jpg

        for iCounter in range(len(dates)):

            date = dates[iCounter]
            des_start = desired_starts[iCounter]
            des_end = desired_ends[iCounter]
            dis_ind = dismissed_individuals[iCounter]

            yyyy, mm, dd = date.split("-")
            start_date = datetime(
                year=int(yyyy),
                month=int(mm),
                day=int(dd),
                hour=int(df_enc_info.loc[enclosure_id, "recording_start"]),
            )

            # get individual ids that need to be processed
            individual_ids_enclosure = [
                ind
                for ind in df_enc_info.loc[enclosure_id, "individual_ids"].split(";")
            ]

            for ind_id in individual_ids_enclosure:
                if ind_id in dis_ind:
                    continue

                # load boris annotation
                sequence = get_boris_annotation(
                    anchorpath=ANCHORPATH_BORIS_ANNOTATIONS,
                    individual_id=ind_id,
                    date=date,
                    mode=MODE,
                    start_time=des_start,
                    end_time=des_end,
                )

                balanced_indices = get_balanced_intervals(
                    sequence, possible_behaviors=CLASS_INDICES
                )

                video_list = []
                for j in range(
                    len(df_enc_info.loc[enclosure_id, "video_stream_folder"].split(";"))
                ):
                    video_list.append(
                        get_videoname(
                            date,
                            ANCHORPOINT_VIDEOFILES,
                            j,
                            df_enc_info.loc[enclosure_id, "video_stream_folder"].split(
                                ";"
                            ),
                            df_enc_info.loc[enclosure_id, "video_name"].split(";"),
                        )
                    )

                for behavior_code, time_intervals in balanced_indices.items():
                    video_to_process.append(
                        [
                            time_intervals,
                            start_date,
                            out + "tmp/",
                            behavior_code,
                            enclosure_id,
                            video_list,
                        ]
                    )

    # We need to be cautious as the same video is opened twice. While the computational cost does not matter at this point
    # it might generate conflicts in the parallel processing. If this is the case, we need to fix it (TODO)
    np.random.shuffle(video_to_process)
    thread_map(
        IO_video_timeinterval,
        video_to_process,
        max_workers=cpu_count(),
        desc="Extract behavioral sequences to images.",
    )

    # Step 2: apply object detector to images of one enclosure
    ## images are saved here as {action_label}/{enc_id}/differences/{enc_id}_{curr_time}.jpg' and images (respectively)
    ## saves the images in the subfolder out/mode/action_id/individual_id/yyyymmdd-hhmmss_individual_id.jpg
    ensure_directory(out + "tmp/")
    if not os.path.exists(out + "tmp/"):
        print(
            "FATAL ERROR. Something went wrong, object detection/segmentation skipped."
        )
        return
    for action_label in sorted([int(x) for x in os.listdir(out + "tmp/")]):
        if not action_label in CLASS_INDICES:
            continue

        for enclosure_id in tqdm(
            sorted(os.listdir(f"{out}tmp/{action_label}/")),
            desc=f"Object {OBJECT_DETECTION_MODE} of class {action_label}",
        ):
            if not (
                os.path.exists(f"{out}tmp/{action_label}/{enclosure_id}/images/")
                and os.path.exists(
                    f"{out}tmp/{action_label}/{enclosure_id}/differences/"
                )
            ):
                print(
                    f"WARNING. Folder {out}tmp/{action_label}/{enclosure_id}/ skipped."
                )
                continue

            predict_folder_differences(
                folder_path=f"{out}tmp/{action_label}/{enclosure_id}/images/",
                difference_folder_path=f"{out}tmp/{action_label}/{enclosure_id}/differences/",
                output_images=f"{out}{MODE}/{action_label}/",
                output_segment=None,
                enclosure_id=enclosure_id,
                mode=OBJECT_DETECTION_MODE,
                device=DEVICE,
            )

    # Step 3: remove tmp folder
    if REMOVE_TEMPORARY_FILES:
        shutil.rmtree(f"{out}tmp/")


def merge_datasets(merge=MERGE_DATASETS, out=OUTPUT_PATH_MERGED_DATASET):
    """
    Gets a list of multiple datasets as well as the information which fraction of which class should be taken for each dataset.
    Outputs (IO) a joint dataset into out/
    If AUTO_BALANCE is enabled, it upsamples the images such that classes are balanced. Slight augmentations are applied.
    """
    if not len(merge["datasets"]) == len(merge["fraction_images_per_class"]):
        print("ERROR. Length of dataset input and sample fractions do not coincide.")
        return

    if not len(merge["datasets"]) == len(merge["dismiss_individuals"]):
        print(
            "ERROR. Length of dataset input and sets of dismissed individuals do not fit."
        )
        return

    invalid_sampling = False
    for sample_dict in merge["fraction_images_per_class"]:
        for frac in sample_dict.values():
            if not 0 <= frac <= 1:
                invalid_sampling = True
    if invalid_sampling:
        print("ERROR. Not all sample rates are between zero and one.")
        return

    for sample_dict in merge["fraction_images_per_class"]:
        if not set(sample_dict.keys()) == set(CLASS_INDICES):
            invalid_sampling = True
    if invalid_sampling:
        print(
            f"ERROR. In some datasets, there are invalid sample indices (requires {CLASS_INDICES})."
        )
        return

    for j in tqdm(range(len(merge["datasets"])), desc="Datasets processed"):
        d_path = merge["datasets"][j]
        samples = merge["fraction_images_per_class"][j]
        dismissed_individuals = merge["dismiss_individuals"][j]
        if not os.path.exists(d_path):
            print(f"ERROR. Dataset does not exist: {d_path}. Skipped this dataset.")
            continue

        if not os.path.exists(f"{d_path}{MODE}"):
            print(
                f"ERROR. Dataset is invalid for mode {MODE}: {d_path}. Skipped this dataset."
            )
            continue

        if len(os.listdir(f"{d_path}{MODE}")) == 0:
            print(f"WARNING. Dataset is empty: {d_path}.")
            continue

        try:
            available_labels = set([int(x) for x in os.listdir(f"{d_path}{MODE}")])
        except:
            print(
                f"ERROR. Some strange subfolders in {d_path}{MODE}. Skipped the dataset."
            )
            continue

        if not available_labels == set(CLASS_INDICES):
            print(
                f"ERROR. Dataset {d_path}{MODE} has action labels {available_labels} while we expect {CLASS_NAMES}. Skipped the dataset."
            )
            continue

        for action_label in os.listdir(f"{d_path}{MODE}"):
            for ind_id in os.listdir(f"{d_path}{MODE}/{action_label}/"):
                if ind_id in dismissed_individuals:
                    continue
                if not os.path.exists(
                    f"{d_path}{MODE}/{action_label}/{ind_id}/images/0/"
                ):
                    print(
                        f"WARNING: Path does not exist. {d_path}{MODE}/{action_label}/{ind_id}/images/0/. Skipped."
                    )
                    continue
                ensure_directory(f"{out}{MODE}/{action_label}/{ind_id}/images/0/")
                images = [
                    x
                    for x in os.listdir(
                        f"{d_path}{MODE}/{action_label}/{ind_id}/images/0/"
                    )
                    if x.endswith("jpg")
                ]
                num_images = int(len(images) * samples[int(action_label)])
                images = np.random.choice(images, size=num_images, replace=False)
                for filename in images:
                    shutil.copy(
                        src=f"{d_path}{MODE}/{action_label}/{ind_id}/images/0/{filename}",
                        dst=f"{out}{MODE}/{action_label}/{ind_id}/images/0/{filename}",
                    )

    if not AUTO_BALANCE:  # upsampling with slightly augmented images
        return

    _auto_upsampling(f"{out}{MODE}")


def _auto_upsampling(d_path):
    from fnmatch import fnmatch

    labels = [int(x) for x in os.listdir(d_path)]
    samples = {k: [] for k in labels}
    for action_label in labels:
        for path, subdirs, files in os.walk(f"{d_path}/{action_label}/"):
            for name in files:
                if fnmatch(name, "*.jpg"):
                    samples[action_label].append(os.path.join(path, name))
    sample_sizes = {k: len(v) for k, v in samples.items()}

    if min(sample_sizes.values()) == 0:
        print("ERROR in upsampling, one class is empty.")
        return

    upsampling_ratio = {
        k: max(sample_sizes.values()) / v for k, v in sample_sizes.items()
    }

    if max(upsampling_ratio.values()) == 1:
        print("Classes are already balanced, no upsampling required.")
        return

    print(f"Upsampling ratios for {d_path}:", upsampling_ratio)

    for action_label, ups_rat in upsampling_ratio.items():
        if ups_rat >= 0.999:
            continue
        chosen_images = np.random.choice(
            samples[action_label],
            size=round(ups_rat * sample_sizes[action_label])
            - sample_sizes[action_label],
            replace=True,
        )
        if len(chosen_images) == 0:
            continue
        for img_path in tqdm(chosen_images, f"Upsampling class {action_label}"):
            augmented_img = augment_action_classifier_image(img_path)
            new_filename = f"{img_path[:-4]}-{np.random.random()}.jpg"
            save_image_to_file(augmented_img, new_filename)


def generate_training_set(
    d_path=DATASET_PATH, out=TRAINING_VALIDATION_PATH, upsampling=True
):
    """
    Given a dataset, turns it into a training set for an action classifier
    This means that we get the structure
        out/train/action_label/img.jpg
        out/val/action_label/img.jpg
    Automatically does upsampling if not specified explicitly with a function call
    (this makes sure that users normally use balanced classes but experts might change this)
    """
    if not os.path.exists(d_path):
        print("ERROR. Input dataset not existent.")
        return

    print(d_path, MODE)
    if not os.path.exists(f"{d_path}{MODE}"):
        print(f"ERROR. Dataset is invalid for mode {MODE}.")
        return

    if len(os.listdir(f"{d_path}{MODE}")) == 0:
        print("ERROR. Dataset is empty.")
        return

    try:
        available_labels = set([int(x) for x in os.listdir(f"{d_path}{MODE}")])
    except:
        print(f"ERROR. Some strange subfolders in {d_path}{MODE}.")
        return

    if not available_labels == set(CLASS_INDICES):
        print(
            f"ERROR. Dataset {d_path}{MODE} has action labels {available_labels} while we expect {CLASS_INDICES}."
        )
        return

    for action_label in tqdm(os.listdir(f"{d_path}{MODE}"), desc="Preparing classes"):
        for enc_id in os.listdir(f"{d_path}{MODE}/{action_label}/"):
            if not os.path.exists(f"{d_path}{MODE}/{action_label}/{enc_id}/images/0/"):
                print(
                    f"WARNING: Path does not exist. {d_path}{MODE}/{action_label}/{enc_id}/images/0/. Skipped."
                )
                continue
            ensure_directory(f"{out}{MODE}/train/{action_label}/")
            ensure_directory(f"{out}{MODE}/val/{action_label}/")

            images = [
                x
                for x in os.listdir(f"{d_path}{MODE}/{action_label}/{enc_id}/images/0/")
                if x.endswith("jpg")
            ]
            num_images_val = int(len(images) * VALIDATION_SPLIT)
            images_val = np.random.choice(images, size=num_images_val, replace=False)
            for filename in images:
                src = f"{d_path}{MODE}/{action_label}/{enc_id}/images/0/{filename}"
                if filename in images_val:
                    dst = f"{out}{MODE}/val/{action_label}/{filename}"
                else:
                    dst = f"{out}{MODE}/train/{action_label}/{filename}"
                shutil.copy(src=src, dst=dst)

    if upsampling:
        _auto_upsampling(f"{out}{MODE}/train/")
        _auto_upsampling(f"{out}{MODE}/val/")


################################################
###### code to train and evaluate a model ######
################################################


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    """
    Method to conduct one epoch of training.
    model: pytorch model
    Input data_loader:
    Input loss_fn: loss function, standard is CrossEntropyLoss
    Input optimizer: which optimizer is used
    Input device: device to use, can be GPU or some cude device
    Input n_examples: number of images in the data_loader
    """

    model = model.train()
    losses = []
    correct_predictions = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_data_loaders():
    """
    creates data loaders for training and validation set,
    transformations as augmentation are defined here.
    Uses the global path to the training set DATA_DIR_TRAINING_AND_VALIDATION/train/ and DATA_DIR_TRAINING_AND_VALIDATION/val/
    Returns the data_loaders (PyTorch DataLoader) as well as dataset_sizes and class_names
    """

    # Inference transforms EfficientNetV2_S
    # resize_size=[384] using interpolation=InterpolationMode.BILINEAR,
    # followed by a central crop of crop_size=[384].
    # Finally the values are first rescaled to [0.0, 1.0]
    # and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

    # transformations conducted in order to augment dataset
    transforms = {
        "train": T.Compose(
            [
                T.Resize(size=384, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomApply([T.RandomRotation(degrees=15)], p=0.2),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=(0.7, 1.3),
                            contrast=(0.8, 1.2),
                            saturation=(0.8, 1.2),
                        )
                    ],
                    p=0.2,
                ),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                T.RandomPerspective(distortion_scale=0.3, p=0.2),
                T.RandomResizedCrop(size=384, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
                T.RandomHorizontalFlip(p=0.3),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize(size=384, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(size=384),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        d: ImageFolder(f"{DATA_DIR_TRAINING_AND_VALIDATION}{d}", transforms[d])
        for d in ["train", "val"]
    }

    data_loaders = {
        d: DataLoader(
            image_datasets[d], batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        for d in ["train", "val"]
    }

    dataset_sizes = {d: len(image_datasets[d]) for d in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return data_loaders, dataset_sizes, class_names


def train_model(model, device, checkpointpath, n_epochs, modelname, save_every_epoch):
    """
    Method to train a model.
    Input model: pytorch model
    Input data_loaders:
    Input dataset_sizes:
    Input device: device to use, can be GPU or some cude device
    Input checkpointpath:
    Input n_epochs: number of training epochs
    Input modelname: name of the model to save
    Input save_every_epoch: every n-th epoch, model will be saved to checkpointpath even if it is not the currently best performing

    IO operations: save checkpoints and best model, save history file .pb
    Returns the best model.
    """

    ensure_directory(SAVEPATH_MODEL)

    # load data_loaders and classes
    data_loaders, dataset_sizes, class_names = get_data_loaders()

    optimizer = optim.Adam(model.parameters(), amsgrad=True)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    model.to(device)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            model,
            data_loaders["train"],
            loss_fn,
            optimizer,
            device,
            dataset_sizes["train"],
        )

        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_model(
            model, data_loaders["val"], loss_fn, device, dataset_sizes["val"]
        )

        print(f"Val   loss {val_loss} accuracy {val_acc}")

        # story history
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if epoch % save_every_epoch == 0:
            torch.save(
                model.state_dict(),
                checkpointpath + "{}-epoch{}.pth".format(modelname, epoch),
            )

        if val_acc > best_accuracy:
            torch.save(
                model.state_dict(), checkpointpath + "{}-best.pth".format(modelname)
            )
            best_accuracy = val_acc

        print(f"Best val accuracy: {best_accuracy}")

    model.load_state_dict(torch.load(checkpointpath + "{}-best.pth".format(modelname)))

    # save history (pickle)
    with open(checkpointpath + "{}-history.pb".format(modelname), "wb") as file:
        pickle.dump(history, file)

    plot_training_history(checkpointpath + "{}-history.pb".format(modelname), modelname)

    return model


def plot_training_history(history_path, modelname):
    """
    Input a history file (history_path, ends with .pb) and the model name.
    Plots accuracy, loss for training and validation against the epochs and saves it as an image.
    """

    if not os.path.exists(history_path):
        return
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.scatter(
        list(range(len(history["train_loss"]))),
        history["train_loss"],
        label="train loss",
    )
    ax1.scatter(
        list(range(len(history["train_loss"]))),
        history["val_loss"],
        label="validation loss",
    )
    ax1.set_ylim([-0.05, 2.05])
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax2.scatter(
        list(range(len(history["train_loss"]))),
        history["train_acc"],
        label="train accuracy",
    )
    ax2.scatter(
        list(range(len(history["train_loss"]))),
        history["val_acc"],
        label="validation accuracy",
    )
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend()
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    fig.suptitle("Training history")

    dirname = os.path.dirname(history_path)
    plt.savefig("{}/{}-history.jpg".format(dirname, modelname), bbox_inches="tight")


def train_actionclassifier():
    """
    Script to actually train the action classifier.
    """

    def create_fresh_model(n_classes=2):

        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
        n_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(n_features, n_classes)

        return model.to(DEVICE)

    def load_pretrained_model(path_state_dictionary):
        # save:
        # torch.save(model.state_dict(), PATH) .pth
        # if loaded for inference
        # model.eval()
        # if loaded for training
        # model.train()

        # load
        model = efficientnet_v2_s()
        model.load_state_dict(torch.load(path_state_dictionary))
        model.train()

        return model

    if len(PRETRAINED_NETWORK) == 0:
        model = create_fresh_model(n_classes=len(CLASS_INDICES))
    elif not os.path.exists(PRETRAINED_NETWORK):
        print(f"ERROR. Path {PRETRAINED_NETWORK} not found.")
        return
    else:
        model = load_pretrained_model(PRETRAINED_NETWORK)

    train_model(
        model=model,
        device=DEVICE,
        checkpointpath=SAVEPATH_MODEL,
        n_epochs=NUM_EPOCHS,
        modelname=MODELNAME,
        save_every_epoch=SAVE_EVERY_EPOCH,
    )


if __name__ == "__main__":
    create_dataset_from_videos()
    if MERGE:
        merge_datasets()
    generate_training_set()
    train_actionclassifier()
