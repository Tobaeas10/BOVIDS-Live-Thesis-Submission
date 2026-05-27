# -*- coding: utf-8 -*-
"""
Program logic behind the prediction of BOVIDS.
Contains the single steps conducted by BOVIDS.
Is called by run_prediction from a local pc.
"""

__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

import os, sys

from local_pc.thesis_tests import timer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")

from server.bovids_v2.lib.func import ensure_directory
from server.bovids_v2.config.get_config import (
    get_enclosure_information,
    get_individual_information,
)

from server.bovids_v2.lib.prediction_information import _read_prediction_file
from server.bovids_v2.lib.pipeline_functions import (
    create_images_from_video, conduct_object_detection, conduct_action_classification, apply_postprocessing,
    create_statistics, create_visualization, delete_local_data, detect_moving)

from server.bovids_v2.lib.copy_functions import (
    copy_videos_to_local_storage, copy_od_to_server, copy_od_to_local_storage, copy_ac_to_server,
    copy_ac_to_local_storage, copy_postprocessing_to_server)

from datetime import datetime


@timer
def predict_excel_file(
    anchor_video,
    server_od,
    server_ac,
    pred_xlsx,
    local_stor,
    device,
    num_parallel,
    path_ac_save,
    path_stats,
    path_visual,
):
    """
    Actual method triggered to predict the nights given in an excel file (pred_xlsx).
    Gets the path to the video files (base folder), the storage on the server, the local storage and the device to compute on.
    Num_parallel is a batch parameter: such a number of nights is conducted "partially on parallel" meaning that cpu operations are in parallel
    like getting videos or sampling images from videos, but the gpu operations are sequentially conducted.
    """

    if not os.path.exists(pred_xlsx):
        print(f"ERROR. Prediction file {pred_xlsx} not found.")
        return

    enclosure_information = get_enclosure_information()
    individual_information = get_individual_information()

    #ensure directories
    ensure_directory(server_ac)
    ensure_directory(server_od)
    ensure_directory(local_stor)
    ensure_directory(path_stats)
    ensure_directory(path_visual)

    # function to store all information as dictionary
    nightwise_information = _read_prediction_file(
        pred_xlsx=pred_xlsx,
        enclosure_information=enclosure_information,
        individual_information=individual_information,
        anchorpath_od=server_od,
        anchorpath_ac=server_ac,
        anchorpath_video=anchor_video,
        local_storage=local_stor,
        num_parallel=num_parallel,
    )

    for batch_processing_nights in nightwise_information:
        print(
            f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}: Starting prediction on batches of nights {[ (x["enclosure_id"], x["date"]) for x in batch_processing_nights ]}.'
        )
        # fyi: batch_processing_nights is a list of 1...num_parallel many night_info dictionaries (nights),
        # each having 1..N items <=> rows with same date in predict_example.xlsx. Then saved to nightwise_information.

        # Step 1: copy videos to local storage, if necessary [in parallel], return number of videos to copy
        # noo_videos_to_copy = copy_videos_to_local_storage(batch_processing_nights)
        noo_videos_to_copy = 420 # uncoment this line and comment the one above to skip this process (for testing)

        # Step 2: create images from videos [in parallel]
        images_to_create = create_images_from_video(noo_videos_to_copy, batch_processing_nights, enclosure_information)
        # images_to_create = [69] * 420 # uncoment this line and comment the one above to skip this process (for testing)

        # Step 3: conduct object detection / segmentation if necessary [cpu: in parallel, gpu: sequentially]
        # method to actually do the object detection:
        od_conduct_folders = conduct_object_detection(images_to_create, batch_processing_nights, device)

        # Step 4: zip and copy the outcome of od to the server if necessary [in parallel] - creates cache
        copy_od_to_server(od_conduct_folders, batch_processing_nights)

        # Step 5: copy the necessary od from the server to the local storage if necessary [in parallel] - prepares usage of cache for next steps
        # We make sure that only the required files are copied, e.g., if only moving is to predict in AC step, then only the bounding box csvs which boosts performance
        copy_od_to_local_storage(batch_processing_nights)

        # Step 6: conduct action classification if necessary and save output to server [cpu: in parallel, gpu: sequentially]
        # creating folder to save ac predictions
        conduct_action_classification(path_ac_save, batch_processing_nights, enclosure_information, device)


        # Step 7: move prediction (ac) to server and delete local files
        # 1. ac hochladen auf server
        copy_ac_to_server(batch_processing_nights, path_ac_save)
        # 2.ac runterladen falls notwendig
        # Step 7a
        copy_ac_to_local_storage(batch_processing_nights, path_ac_save)
        # Step 8: apply postprocessing if necessary on the server [in parallel]
        apply_postprocessing(batch_processing_nights, path_ac_save)

        # Step 8b:
        detect_moving(batch_processing_nights, path_ac_save, server_od)

        # Step 8b: save pp on server
        copy_postprocessing_to_server(batch_processing_nights, path_ac_save)

        # Step 9: create statistic files and activity budget images on the server [in parallel]
        create_statistics(batch_processing_nights, path_stats, path_ac_save)

        # Step 10: create heatmap / enclosure activity overview
        create_visualization(batch_processing_nights, path_stats, path_visual, individual_information)
        # Step 11: delete everything local (after each night)
        # delete_local_data(batch_processing_nights, path_ac_save)


