__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"


import os, sys
import pandas as pd
import numpy as np

from server.bovids_v2.lib.func import (
    get_video_files,
    _ac_operations_conducted
)
from server.bovids_v2.config.get_config import (
    get_postprocessing_rules,
    AC_MODES,
)
from server.bovids_v2.lib.availability_checks import (
    get_ac_networkpath,
    _get_savepath_ac_images,
    _video_cache_avaiable,
    _ac_cache_avaiable,
    _postprocessing_possible,
    _statistics_stly_possible,
    _statistics_subactions_possible
)


def _read_prediction_file(
    pred_xlsx,
    enclosure_information,
    individual_information,
    anchorpath_od,
    anchorpath_ac,
    anchorpath_video,
    local_storage,
    num_parallel,
):
    """
    Method to read the prediction excel file and to collect information which tasks will be conducted.
    Checks for the existence of the required networks, gets required paths and checks for the caching of items.
    It returns a list of lists, the inner lists have size num_parallel for batch processing.

    Each entry is a dictionary of the following elements
                {'enclosure_id': enclosure_id,
                  'dismissed_individuals': list of individuals belonging to the enclosure but being dismissed from this night by the user,
                  'date': date,
                  'eval_start': start time of the evaluation,
                  'eval_end': end time of the evaluation,
                  'savepath_od': path on which object detection elements will be saved or are cached (for this night), '{anchorpath}{od_task}/{enclosure_id}/{date}/'
                  'savepath_ac': ..., dictionary with key = individual_id, '{anchorpath}{ind}/'
                  'video_list': list of the video files required for od this night,
                  'temporal_storage': local_storage (just a path),
                  # od
                  'detection_mode' segment / detect
                  'copy_videos': copy_videos, True/False, depending on if OD needs to be carried out
                  'copy_od_tmp_to_server_images' : copy_od_tmp_to_server_images,  list of individuals for which od images need to be transferred to the server
                  'copy_od_server_to_tmp_images' : copy_od_server_to_tmp_images,  list of individuals for which od images need to be transferred to the temporal storage
                  'copy_od_tmp_to_server_bbinfo' : copy_od_tmp_to_server_bbinfo,  list of individuals for which od bounding box information need to be transferred to the server
                  'copy_od_server_to_tmp_bbinfo' : copy_od_server_to_tmp_bbinfo,  list of individuals for which od bounding box information need to be transferred to the local storage
                  # ac
                  'ac_predictions': ac_predictions, dictionary of ac_modes as keys, each value is a list of individuals {'StLy': [IndId1]}
                  'ac_networkpaths': ac_network_paths, to each individual a dictionary of required networks {IndId: {'StLy': path}}
                  'moving_iou': moving_iou, {individual_id: moving_iou_threshold}
                  # post_processing
                  'apply_postprocessing': dictionary with ac_modes as keys, each value is a list of individuals {'StLy': [IndId1]}
                  'postprocessing_rules': post_processing_rules[individual_id][ac_mode] contains postprocessing-dataframe (i = df[ (df['previous'] == x) & (df['current'] == y) & (df['next']==z)  ].index[0] --> min_length, behavior_new = df.loc[i, 'min_length'], df.loc[i, 'current_corrected'])
                  'individual_postprocessing':
                  'individual_moving':
                  # create statistics file and draw images
                  'statistic_tasks': create_statistics dictionary ac_mode : list, for ac_modes the list is just a list of individuals for which it is necessary to do,
                                      for Final a list of lists [individual_id, [x for x in ac_modes if int(df.loc[row, f'stats_{x}']) == 1] ])
    """

    required_columns = [
        "enclosure_id",
        "date",
        "evaluation_start",
        "evaluation_end",
        "dismiss_individuals",
        "StLy",
        "StFo",
        "LHULHD",
        "Moving",
        "use_cached_od",
        "use_cached_StLy",
        "use_cached_StFo",
        "use_cached_LHULHD",
        "use_cached_Moving",
        "apply_postprocessing",
        "use_cached_pp_StLy",
        "use_cached_pp_subactions",
        "stats_StLy",
        "stats_subactions",
    ]
    resources_folder = os.path.abspath(os.path.dirname(__file__) + "/../res/")

    ret = []

    # read excel file
    try:
        df = pd.read_excel(pred_xlsx)
    except:
        print(f"ERROR. Prediction file {pred_xlsx} does not exist.")
        return ret

    # check for correct format
    columns_missing = False
    for r in required_columns:
        if not r in df.keys():
            columns_missing = True
    if columns_missing:
        print(f"ERROR. Prediction file {pred_xlsx} is invalid.")
        return ret

    for row in df.index:
        enclosure_id = df.loc[row, "enclosure_id"]

        # gets date for enclosure
        try:
            date = df.loc[row, "date"].strftime("%Y-%m-%d")
        except:
            print(
                f'ERROR. Invalid date format {df.loc["date"]} at enclosure {enclosure_id}. Skip the night.'
            )
            continue

        if not enclosure_id in enclosure_information.index:
            print(
                f"ERROR. {enclosure_id} is not contained in the global configuration file. Skipped the night."
            )
            continue

        # get evaluation interval
        eval_start = df.loc[row, "evaluation_start"]
        eval_end = df.loc[row, "evaluation_end"]

        # gets task for enclosure
        od_task = enclosure_information.loc[enclosure_id, "task"]

        # checks for dismissed individuals and creates a list of remaining individuals
        remaining_individuals, dismissed_individuals = check_dismissed_individuals(df, row, enclosure_information, enclosure_id)

        # creates list containing all missing individuals and skips them at future calculations
        missing_individuals = []
        for ind in remaining_individuals:
            if not ind in individual_information.index:
                missing_individuals.append(ind)
        if len(missing_individuals) > 0:
            print(
                f"ERROR. For the following individuals, we have no individual configuration. Skip the night. {missing_individuals}"
            )
            continue

        # paths to save object detection and action classification
        savepath_od = f"{anchorpath_od}{od_task}/{enclosure_id}/{date}/"
        savepath_ac = _get_savepath_ac_images(
            anchorpath_ac, enclosure_id, dismissed_individuals, enclosure_information
        )

        if not int(df.loc[row, "od"]) and not int(df.loc[row, "use_cached_od"]):
            print(
                "ERROR. Either object detection must be done or od cache must be used. Skip the night. "
            )
            continue

        if int(df.loc[row, "od"]) and int(df.loc[row, "use_cached_od"]):
            print(
                "Warning. Only od or use_cached_od can be used. Now the cache will be used."
            )


        if int(df.loc[row, "od"]):
            perform_od = True
        else:
            perform_od = False


        video_cache = _video_cache_avaiable(
            enclosure_id, date, savepath_od, dismissed_individuals, enclosure_information)

        copy_videos, copy_od_tmp_to_server_images, copy_od_server_to_tmp_images, copy_od_tmp_to_server_bbinfo, copy_od_server_to_tmp_bbinfo = check_od_cache(
            video_cache, df, row, remaining_individuals)

        # check if od networks are available if required and get their path
        if copy_videos:
            if od_task == "segment":
                od_network_path = (
                        resources_folder
                        + "/imagesegmentation/"
                        + enclosure_information.loc[enclosure_id, "image_segmentor"]
                        + ".pt"
                )

            else:
                od_network_path = (
                        resources_folder
                        + "/objectdetection/"
                        + enclosure_information.loc[enclosure_id, "object_detector"]
                        + ".pt"
                )


            if not os.path.exists(od_network_path):
                print(
                    f"ERROR. Expected object detection/segmentation network {od_network_path} is not available but required. Skip the night."
                )
                continue
        else:
            od_task, od_iou_config, od_certainty_threshold = None, None, None


        videopaths = get_video_files(
            enclosure_id, anchorpath_video, enclosure_information, date
        )  # from lib.func

        # checks if video files are found
        if len(videopaths) == 0:
            print(
                f"ERROR. No video files found for {enclosure_id} at {date}. Skip the night."
            )
            continue


        # figure out which ac operations are necessary per individual
        ac_predictions = {ac_mode: [] for ac_mode in AC_MODES}

        for ac_mode in AC_MODES:
            ac_predictions[ac_mode] = _ac_operations_conducted(
                remaining_individuals, date, ac_mode, savepath_ac, df, row
            )

        # sanity check whether all ac networks which are required are also available, and if so, load their paths
        ac_network_paths = {}
        missing_paths = False
        for ac_mode in AC_MODES:
            for individual_id in ac_predictions[ac_mode]:
                ac_network = get_ac_networkpath(
                    individual_information, individual_id, ac_mode
                )

                if ac_network == None:
                    print(
                        f"ERROR. The action classifier {ac_mode} is not available for individual {individual_id} but required. Skip the night."
                    )
                    missing_paths = True

                if not individual_id in ac_network_paths.keys():
                    ac_network_paths[individual_id] = {}
                ac_network_paths[individual_id][ac_mode] = ac_network
        if missing_paths:
            continue

        # sanity check whether StLy is either predicted or, cached and allowed to use cache
        sanity_ac = True
        for individual_id in remaining_individuals:
            if individual_id in ac_predictions["StLy"]:
                continue
            #and individual_id not in ac_predictions["Moving"]
            if (
                individual_id not in ac_predictions["StFo"]
                and individual_id not in ac_predictions["LHULHD"]

            ):
                continue
            if int(df.loc[row, f"use_cached_{ac_mode}"]) == 0:
                # StLy not predicted, but required. Moreover caching forbidden.
                sanity_ac = False
                continue
            if not _ac_cache_avaiable(individual_id, date, "StLy", savepath_ac):
                sanity_ac = False
                continue
        if not sanity_ac:
            print(
                f"ERROR. AC predictions wanted but there is no StLy mode activated or cached."
            )


        # collect moving iou
        try:
            moving_iou = {
                individual_id: float(
                    individual_information.loc[individual_id, "IOU_Moving"]
                )
                for individual_id in remaining_individuals
            }
        except:
            print(
                f"ERROR. For one of the individual_ids in {remaining_individuals}, the IOU_Moving entry is invalid."
            )
            continue

        #check for post processing information
        postprocessing_invalid, unknown_rules, apply_postprocessing, post_processing_rules, individual_post_proc = check_postprocessing(
            df, row, remaining_individuals, savepath_ac, date, individual_information)

        if postprocessing_invalid:
            print(
                "ERROR. Postprocessing rules are partly non existent. Skip the night.",
                unknown_rules,
            )
            continue

        # check for moving information
        moving = check_moving(df, row, remaining_individuals)

        # decide which statistics will be created
        stats = check_statistics(remaining_individuals, df, row, savepath_ac, date)

        # dictionary with all information for one enclosure
        night_info = {
            "enclosure_id": enclosure_id,
            "dismissed_individuals": dismissed_individuals,
            "date": date,
            "eval_start": eval_start,
            "eval_end": eval_end,
            "savepath_od": savepath_od,
            "savepath_ac": savepath_ac,
            "video_list": videopaths,
            "temporal_storage": local_storage,
            # od
            "perform_od": perform_od,
            "copy_videos": copy_videos,
            "detection_mode": od_task,
            "copy_od_tmp_to_server_images": copy_od_tmp_to_server_images,
            "copy_od_server_to_tmp_images": copy_od_server_to_tmp_images,
            "copy_od_tmp_to_server_bbinfo": copy_od_tmp_to_server_bbinfo,
            "copy_od_server_to_tmp_bbinfo": copy_od_server_to_tmp_bbinfo,
            # ac
            "ac_predictions": ac_predictions,
            "ac_networkpaths": ac_network_paths,
            "moving_iou": moving_iou,
            # post_processing
            "apply_postprocessing": apply_postprocessing,
            "postprocessing_rules": post_processing_rules,
            "individual_postprocessing": individual_post_proc,
            "individual_moving": moving,
            # create statistics file
            "statistic_tasks": stats,
        }

        # adds all enclosures to one list
        ret.append(night_info)

    # partition ret into chunks
    final = [
        ret[i * num_parallel : (i + 1) * num_parallel]
        for i in range((len(ret) + num_parallel - 1) // num_parallel)
    ]
    return final

def check_dismissed_individuals(df, row, enclosure_information, enclosure_id):
    # checks for dismissed individuals and creates a list of remaining individuals
    if not np.isnan(df.loc[row, "dismiss_individuals"]):
        dismissed_individuals = [
            x for x in df.loc[row, "dismiss_individuals"].split(";")
        ]
        remaining_individuals = sorted(
            [
                x
                for x in enclosure_information.loc[
                    enclosure_id, "individual_ids"
                ].split(";")
                if not x in dismissed_individuals
            ]
        )
    else:
        remaining_individuals = [
            x
            for x in enclosure_information.loc[
                enclosure_id, "individual_ids"
            ].split(";")
        ]
        dismissed_individuals = []
    return remaining_individuals, dismissed_individuals

def check_postprocessing(df, row, remaining_individuals, savepath_ac, date, individual_information):
    # decide if postprocessing is applied
    apply_postprocessing = {ac_mode: [] for ac_mode in AC_MODES}
    if int(df.loc[row, "apply_postprocessing"]):
        for individual_id in remaining_individuals:
            for ac_mode in AC_MODES:
                if _postprocessing_possible(
                        individual_id, ac_mode, df, row, savepath_ac, date
                ):
                    apply_postprocessing[ac_mode].append(individual_id)

    individual_post_proc = {}
    for individual in remaining_individuals:
        individual_post_proc[individual] = bool(df.loc[row, "apply_postprocessing"])

    # find the post-processing rules
    post_processing_rules = {
        ind: {"StLy": 0, "StFo": 0, "LHULHD": 0, "Moving": 0}
        for ind in remaining_individuals
    }
    unknown_rules = []
    for individual_id in remaining_individuals:
        for ac_mode in AC_MODES:
            pp_rulename = individual_information.loc[
                individual_id, f"postproc_{ac_mode}"
            ]
            post_processing_rules[individual_id][ac_mode] = (
                get_postprocessing_rules(pp_rulename, ac_mode)
            )
        # old version (with error):
        # if None in post_processing_rules[individual_id].values():
        if any(
                value is None for value in post_processing_rules[individual_id].values()
        ):
            unknown_rules.append({ac_mode: individual_id})

    postprocessing_invalid = False
    for ur in unknown_rules:
        for ac_mode, individual_id in ur.items():
            if individual_id in apply_postprocessing[ac_mode]:
                postprocessing_invalid = True

    return postprocessing_invalid, unknown_rules, apply_postprocessing, post_processing_rules, individual_post_proc

def check_statistics(remaining_individuals, df, row, savepath_ac, date):
    # dict for which indvidual statistics should be created  [StLy, subactions]
    stats = {individual_id: [] for individual_id in remaining_individuals}
    for individual_id in remaining_individuals:
        # check for StLy statistics
        if not int(df.loc[row, "stats_StLy"]) :
            stats[individual_id].append(False)
        elif int(df.loc[row, "stats_StLy"]):
            if not _statistics_stly_possible(
                    individual_id, df, row, savepath_ac, date
            ):
                print(
                    "ERROR. No postprocessing available. Skip statistics for this individual."
                )
            stats[individual_id].append(
                _statistics_stly_possible(individual_id, df, row, savepath_ac, date)
            )

        # check for subactions statistics
        if not int(df.loc[row, "stats_subactions"]):
            stats[individual_id].append(False)
        elif (
                not int(df.loc[row, "stats_StLy"])
                and int(df.loc[row, "stats_subactions"])
        ):
            print(
                "ERROR. stats_subactions can not be executed while stats_StLy is not available."
            )
            stats[individual_id].append(False)
        elif int(df.loc[row, "stats_subactions"]):
            if not _statistics_stly_possible(
                    individual_id, df, row, savepath_ac, date
            ):
                print(
                    "ERROR. No postprocessing available. Skip statistics for this individual."
                )
                stats[individual_id].append(False)
            else:
                stats[individual_id].append(
                    _statistics_subactions_possible(
                        individual_id, df, row, savepath_ac, date
                    )
                )
    return stats

def check_od_cache(video_cache, df, row, remaining_individuals):
    if int(df.loc[row, "use_cached_od"]) == 0 or not video_cache:
        copy_videos = True
        copy_od_tmp_to_server_images = remaining_individuals
        copy_od_server_to_tmp_images = []
        copy_od_tmp_to_server_bbinfo = remaining_individuals
        copy_od_server_to_tmp_bbinfo = []

    else:
        copy_videos = False
        copy_od_tmp_to_server_images = []
        copy_od_server_to_tmp_images = remaining_individuals
        copy_od_tmp_to_server_bbinfo = []
        copy_od_server_to_tmp_bbinfo = remaining_individuals

    return copy_videos, copy_od_tmp_to_server_images, copy_od_server_to_tmp_images, copy_od_tmp_to_server_bbinfo, copy_od_server_to_tmp_bbinfo

def check_moving(df, row, remaining_individuals):
    moving = []
    if int(df.loc[row, "Moving"]):
        for individual_id in remaining_individuals:
            moving.append(individual_id)

    return moving