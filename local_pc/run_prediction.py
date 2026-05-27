# -*- coding: utf-8 -*-
"""
Actual script to run the BOVIDS prediction.
The only file that might be stored on the local computer.
Contains only a wrapper to the actual program logic and the
paths on the computer.
"""
__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

import time

import torch
import sys

"""
ANCHORPATH_VIDEOFILES = '' # anchorpath to video files such that in enclosure_information the relative path starts.
SERVERPATH_OBJECTDETECTION_FILES = '' # anchor to the saved object detection and segmentation images, contains folders with the enclosure code
SERVERPATH_ACTION_CLASSIFICATION = '' # same but for action classification


TEMPORAL_LOCAL_STORAGE = '' # temporary files as videos and cutout images will be stored here (should really be a local path)

BOVIDS_AC= ''  # path that leads to local storage
"""

# Judith & Lea
#ANCHORPATH_VIDEOFILES = r"C:/Users/judit/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/"  # anchorpath to video files such that in enclosure_information the relative path starts.
ANCHORPATH_VIDEOFILES = 'C:/Users/Tobae/Desktop/Programming/PyCharm/MyBOVIDS2/server/bovids_v2/res/Video/'
SERVERPATH_OBJECTDETECTION_FILES = 'C:/Users/Tobae/Desktop/Programming/PyCharm/MyBOVIDS2/paths/SERVERPATH_OBJECTDETECTION_FILES/' # anchor to the saved object detection and segmentation images, contains folders with the enclosure code
SERVERPATH_ACTION_CLASSIFICATION = 'C:/Users/Tobae/Desktop/Programming/PyCharm/MyBOVIDS2/paths/SERVERPATH_ACTION_CLASSIFICATION/' # same but for action classification

#SERVERPATH_OBJECTDETECTION_FILES = r"C:/Users/judit/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/Server/object_detection/predicted_images/"
#SERVERPATH_ACTION_CLASSIFICATION = r"C:/Users/judit/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/Server/action_classification/"

#TEMPORAL_LOCAL_STORAGE = r"C:/Users/judit/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/objection_detection/"  # temporary files as videos and cutout images will be stored here (should really be a local path)
TEMPORAL_LOCAL_STORAGE = 'C:/Users/Tobae/Desktop/Programming/PyCharm/MyBOVIDS2/paths/TEMPORAL_LOCAL_STORAGE/'
BOVIDS_AC = 'C:/Users/Tobae/Desktop/Programming/PyCharm/MyBOVIDS2/paths/BOVIDS_AC/'  # path that leads to local storage
#BOVIDS_AC = r"C:/Users/judit/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/action_classification/"

BOVIDS_STATS = 'C:/Users/Tobae/Desktop/Programming/PyCharm/MyBOVIDS2/paths/BOVIDS_STATS/'
BOVIDS_VISUAL = r'C:/Users/Tobae/Desktop/Programming/PyCharm/MyBOVIDS2/paths/BOVIDS_VISUAL/'

#BOVIDS_STATS = r"C:/Users/judit/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/Server/statistics/"
#BOVIDS_VISUAL = r"C:/Users/judit/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/Server/visualization/"

BATCH_SIZE = 15  # to speed up prediction, up to BATCH_SIZE videos are fetched and translated into single images in parallel. execution of OD / AC is always sequential. post-processing is also parallelised.

BOVIDS_LIBRARY = "../server/bovids_v2/"  # path to the BOVIDS library on the server
PREDICTION_XLSX = "../server/bovids_v2/stuff/predict_example.xlsx"  # excel file that contains the information which nights should be predicted

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # can be cuda:0, cuda:1, ..., depending on number of GPUs.
# Note that AMD GPUs only work on Linux.

if __name__ == "__main__":
    print("Used Device:", DEVICE)

    sys.path.append(BOVIDS_LIBRARY)
    sys.path.append(PREDICTION_XLSX)
    from server.bovids_v2.lib.bovids_prediction import predict_excel_file

    predict_excel_file(
        anchor_video=ANCHORPATH_VIDEOFILES,
        server_od=SERVERPATH_OBJECTDETECTION_FILES,
        server_ac=SERVERPATH_ACTION_CLASSIFICATION,
        pred_xlsx=PREDICTION_XLSX,
        local_stor=TEMPORAL_LOCAL_STORAGE,
        device=DEVICE,
        num_parallel=BATCH_SIZE,
        path_ac_save=BOVIDS_AC,
        path_stats=BOVIDS_STATS,
        path_visual=BOVIDS_VISUAL
    )
