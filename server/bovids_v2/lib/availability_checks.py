#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of methods to check data avaibility.
"""

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

import pandas as pd
import os


def get_ac_networkpath(individual_information, individual_id, ac_task):
    """
    Given the individual and the task, the ac network path is returned.
    Return None if the network is not available.
    """
    # print(ac_task)
    network_name = individual_information.loc[
        individual_id, f"action_classifier_{ac_task}"
    ]
    # print(network_name)
    path = os.path.abspath(
        os.path.dirname(__file__) + "/../res/actionclassification/" + network_name
    )

    if not os.path.exists(path):
        return None
    return path


def get_availabe_video_dates(enclosure_id, base_vid, enc_conf, enc_conf_obj=None):
    """

    Parameters
    ----------
    enclosure_id : String
        Enclosure_id as defined in the configuration files..
    base_vid : String
        Anchorpoint for the videos.
    enc_conf : String
        path to enclosure information file
    enc_conf_obj : DataFrame, optional
        DataFrame of enclosure_information

    Returns
    -------
    List
        List of dates such that for any date there are all videos
        required for the given enclosure id present.

    """

    if enc_conf_obj is None:
        df = pd.read_csv(enc_conf, encoding="latin-1", index_col="enclosure_id")
        if not enclosure_id in df.index:
            return []
    else:
        df = enc_conf_obj.copy()

    vid_stream_folders = df.loc[enclosure_id, "video_stream_folder"].split(";")
    vid_stream_folders = [base_vid + x for x in vid_stream_folders]
    video_name_suffix = df.loc[enclosure_id, "video_name"].split(";")

    if len(vid_stream_folders) != len(video_name_suffix):
        print("ERROR in configuration {}. Mismatch in the number of videofiles.")
        return []

    datelists = [[] for j in range(len(vid_stream_folders))]
    for j in range(len(vid_stream_folders)):
        for f in sorted(os.listdir(vid_stream_folders[j])):
            if not f.endswith(video_name_suffix[j]):
                continue
            datelists[j].append(f.split("_")[0])

    if not len(list(set([len(x) for x in datelists]))) == 1:
        print(
            "WARNING: not all video folders contain the same dates ({})".format(
                enclosure_id
            )
        )

    return sorted(list(set.intersection(*map(set, datelists))))

def _get_savepath_ac_images(
    anchorpath, enclosure_id, dismissed_individuals, enclosure_information
):
    """Given the savepath of ac evaluation on the server, it returns the path to the cached files per individual"""
    individuals = sorted(
        [
            x
            for x in enclosure_information.loc[enclosure_id, "individual_ids"].split(
                ";"
            )
        ]
    )
    ret = {}
    for ind in individuals:
        if ind in dismissed_individuals:
            continue
        ret[ind] = f"{anchorpath}{ind}/"
    return ret

def _video_cache_avaiable(
    enclosure_id, date, savepath_od, dismissed_individuals, enclosure_information
):
    """Checks whether all od predictions of the input night are present in the cache"""

    all_available = True
    individuals = sorted(
        [
            x
            for x in enclosure_information.loc[enclosure_id, "individual_ids"].split(
                ";"
            )
        ]
    )
    for ind in individuals:
        if ind in dismissed_individuals:
            continue
        if not os.path.exists(f"{savepath_od}{ind}/{date}_{ind}_odimages.zip"):
            all_available = False
        if not os.path.exists(
            f"{savepath_od}{ind}/{date}_{ind}_boundingbox-positions.csv"
        ):
            all_available = False
    return all_available


def _ac_cache_avaiable(individual_id, date, ac_mode, savepath_ac):

    if not os.path.exists(
        f"{savepath_ac[individual_id]}/raw/{ac_mode}/prediction/{date}_{individual_id}_{ac_mode}.csv"
    ):
        return False

    return True


def _pp_stly_cache_avaiable(individual_id, date, savepath_ac):
    if not os.path.exists(
        f"{savepath_ac[individual_id]}post_processed/StLy/{date}_{individual_id}_StLy_behavior_info_post_processed.csv"
    ):

        return False

    return True


def _pp_subactions_cache_avaiable(individual_id, date, savepath_ac):
    if not os.path.exists(
        f"{savepath_ac[individual_id]}post_processed/{date}_{individual_id}_behavior_info_post_processed.csv"
    ):
        return False
    return True

def _requires_od_images(ac_classification):
    """Returns True if the od images are actually required (StandingLying, ...)
    Returns False if either no ac is conducted or in case only moving should be identified
    """
    if len(ac_classification) == 0:
        return False

    if set(ac_classification) == set(["Moving"]):
        return False

    # TODO: if we implement heatmaps, add this option.

    return True

def _postprocessing_possible(individual_id, ac_mode, df, row, savepath_ac, date):
    """checks whether post processing can be conducted in this night"""
    if int(df.loc[row, f"{ac_mode}"]):
        return True
    if int(df.loc[row, f"use_cached_{ac_mode}"]) and _ac_cache_avaiable(
        individual_id, date, ac_mode, savepath_ac
    ):
        return True

    return False


def _statistics_stly_possible(individual_id, df, row, savepath_ac, date):
    """checks whether statistics can be conducted"""
    if int(df.loc[row, "apply_postprocessing"]):
        return True
    if int(df.loc[row, "use_cached_pp_StLy"]) and _pp_stly_cache_avaiable(
        individual_id, date, savepath_ac
    ):
        return True
    return False


def _statistics_subactions_possible(individual_id, df, row, savepath_ac, date):
    """checks whether statistics can be conducted"""
    if int(df.loc[row, "apply_postprocessing"]):
        return True
    if int(
        df.loc[row, "use_cached_pp_subactions"]
    ) and _pp_subactions_cache_avaiable(individual_id, date, savepath_ac):
        return True
    return False

