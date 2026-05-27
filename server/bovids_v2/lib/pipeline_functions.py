__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

import traceback

import statistics
import pandas as pd

from server.bovids_v2.lib.func import ensure_directory, get_images_ac_mode

from server.bovids_v2.config.get_config import (
    get_required_timeintervals,
    SECONDS_PER_INTERVAL,
    BEHAVIOR_STATISTICS,
    AC_MODES,
)

from server.bovids_v2.lib.image_manipulation import IO_video_to_img_prediction
from server.bovids_v2.lib.object_detection import predict_folder_differences_par
from server.bovids_v2.lib.action_classification import predict_folder_ac
from server.bovids_v2.lib.post_processor import PostProcessor, PostProcessorSubactions
from server.bovids_v2.lib.visualizer import NightVisualizer

from tqdm.contrib.concurrent import process_map, cpu_count
from datetime import datetime, timedelta, time
import os

def create_images_from_video(noo_videos_to_copy, batch_processing_nights, enclosure_information):
    """
    Collect needed information to create images from video.
    """
    images_to_create = []
    if noo_videos_to_copy > 0:
        # requires list of lists [ [[video_files], savepath, start_time_recording, enclosure_id, required_intervals] ]
        for night_info in batch_processing_nights:
            if night_info["perform_od"]:

                if not night_info["copy_videos"]:
                    continue
                # variables for image creation
                enclosure_id = night_info["enclosure_id"]
                vid_files_night = [
                    f'{night_info["temporal_storage"]}videofiles/{os.path.basename(x)}'
                    for x in night_info["video_list"]
                ]
                savepath = f'{night_info["temporal_storage"]}/single_images/{enclosure_id}/{night_info["date"]}/'
                required_time_intervals = get_required_timeintervals(
                    enclosure_information, enclosure_id, night_info["eval_start"], night_info["eval_end"]
                )
                y, m, d = night_info["date"].split("-")
                start_time_video = datetime(
                    year=int(y),
                    month=int(m),
                    day=int(d),
                    hour=int(
                        enclosure_information.loc[enclosure_id, "recording_start"]
                    ),
                    minute=0,
                    second=0,
                )
                # list of variables to perform image creation
                images_to_create.append(
                    [
                        required_time_intervals, # intervals can be numbers from 0 to VIDEO_LENGTH_IN_SECONDS // SECONDS_PER_INTERVAL
                        start_time_video,
                        savepath,
                        enclosure_id,
                        vid_files_night,
                    ]
                )
    if len(images_to_create) > 0:
        _ = process_map(IO_video_to_img_prediction, images_to_create, max_workers=cpu_count(),
                        desc='Calculate images and difference images ')
    return images_to_create


def conduct_object_detection(images_to_create, batch_processing_nights, device):
    """
    Collect needed information to conduct object detection.
    """

    od_conduct_folders = []

    if len(images_to_create) > 0:
        for night_info in batch_processing_nights:
            if night_info["perform_od"]:
                if not night_info["copy_videos"]:
                    continue
                # variables to perform od
                enclosure_id = night_info["enclosure_id"]
                images_folder = f'{night_info["temporal_storage"]}/single_images/{enclosure_id}/{night_info["date"]}/images/'
                differences_folder = f'{night_info["temporal_storage"]}/single_images/{enclosure_id}/{night_info["date"]}/differences/'
                output_images = f'{night_info["temporal_storage"]}/predicted_images/{enclosure_id}/{night_info["date"]}/'
                output_bb_information = f'{night_info["temporal_storage"]}/predicted_images/{enclosure_id}/{night_info["date"]}/'
                detection_mode = night_info["detection_mode"]
                dismissed_individuals = night_info["dismissed_individuals"]

                # list with variables to perform od
                # od_conduct_folders.append([images_folder, differences_folder, output_images, output_bb_information, detection_mode, dismissed_individuals, night_info["date"] ] )

                od_conduct_folders.append(
                    [
                        images_folder,
                        differences_folder,
                        output_images,
                        output_bb_information,
                        enclosure_id,
                        detection_mode,
                        device,
                        dismissed_individuals,
                        night_info["date"],
                    ]
                )


    # performing od
    if len(od_conduct_folders) > 0:
        _ = process_map(predict_folder_differences_par, od_conduct_folders,
                        max_workers=cpu_count() if device == 'cpu' else 1, desc='Conduct object detection ')
    # output folder of the predicted images will be temporal_storage/predicted_images/enclosure_id/date/{individualid}/images/{time_val}_{individualid}.jpg

    return od_conduct_folders


def conduct_action_classification(path_ac_save, batch_processing_nights, enclosure_information, device):
    """
    Collect needed information to conduct action classification
    """
    ensure_directory(path_ac_save)

    for night_info in batch_processing_nights:
        ac_network_paths = night_info["ac_networkpaths"]
        date = night_info["date"]
        enclosure_id = night_info["enclosure_id"]
        # prepare information used for a list of image names which could be available in StLy mode
        potentially_available_intervals = get_required_timeintervals(
            enclosure_information, enclosure_id, night_info["eval_start"], night_info["eval_end"]
        )
        potential_start_frames = [
            SECONDS_PER_INTERVAL * time_interval
            for time_interval in potentially_available_intervals
        ]
        y, m, d = night_info["date"].split("-")
        start_time_video = datetime(
            year=int(y),
            month=int(m),
            day=int(d),
            hour=int(enclosure_information.loc[enclosure_id, "recording_start"]),
            minute=0,
            second=0,
        )
        potentially_available_image_names = {}

        # set mode that is detected for every individual
        for prediction_mode in AC_MODES:

            if prediction_mode == "Moving":
                # moving is processed later
                continue
            individuals = night_info["ac_predictions"][prediction_mode]
            ac_conduct_folders = []

            for individual in individuals:
                # potentially available image names by individual (not by enclosure)
                # images in this list which are not actually present need to be marked as "out of view"
                if not individual in potentially_available_image_names:
                    potentially_available_image_names[individual] = {}
                potentially_available_image_names[individual][
                    night_info["date"]
                ] = [
                    f'{(start_time_video + timedelta(seconds=float(start_frame))).strftime("%Y%m%d-%H%M%S")}_{individual}'
                    for start_frame in potential_start_frames
                ]

                # save path for every individual
                output_ac_prediction = f"{path_ac_save}{individual}/raw/{prediction_mode}/prediction/"
                output_stly_prediction = (
                    f"{path_ac_save}{individual}/raw/StLy/prediction/"
                )
                ensure_directory(output_ac_prediction)
                ensure_directory(output_stly_prediction)
                input_od_images = f'{night_info["temporal_storage"]}/predicted_images/{enclosure_id}/{night_info["date"]}/{individual}/images/'
                new_path = f'{night_info["temporal_storage"]}/predicted_images/{enclosure_id}/{night_info["date"]}/{individual}/images/0/'
                input_imagenames_use = get_images_ac_mode(
                    prediction_mode,
                    output_stly_prediction,
                    night_info["date"],
                    individual,
                    new_path,
                )
                path_ac_network = ac_network_paths[individual][prediction_mode]

                # list with variables to perform ac
                ac_conduct_folders.append(
                    [
                        input_od_images, # path to images without /0 (for pre-loading)
                        input_imagenames_use, # list of image names with .jpg
                        path_ac_network, # path to ac network prediction model file
                        prediction_mode, # AC mode: StLy, StFo or LHULHD
                        device, # device to perform ac on
                        output_ac_prediction, # path to save ac prediction results to
                        date, # date of the start of the night
                        individual, # individual id
                        output_stly_prediction, # path where stly prediction results have been saved (for StFo and LHULHD)
                        potentially_available_image_names, # dict by individual_id: list of image names without ".jpg" - only the individual that is passed is used.
                    ]
                )

            # performing ac for the current prediction mode
            if len(ac_conduct_folders) > 0:
                _ = process_map(predict_folder_ac, ac_conduct_folders,
                                max_workers=cpu_count() if device == "cpu" else 1,
                                desc=f"Conduct action classification {prediction_mode}")


def apply_postprocessing(batch_processing_nights, path_ac_save):
    """
    Collect needed information to apply post processing for StLy and if required subactions.
    """

    for night_info in batch_processing_nights:

        date = night_info["date"]
        enclosure_id = night_info["enclosure_id"]

        # set mode that is detected for every individual
        ac_modes_temp = AC_MODES.copy()
        if ac_modes_temp.__contains__("LHULHD") and ac_modes_temp.__contains__("StFo"):
            ac_modes_temp.remove("StFo")
        for mode in ac_modes_temp:

            # determines for which individual post processing is applied
            individuals = night_info["apply_postprocessing"][mode]

            for individual in individuals:

                if mode == "StLy":
                    stly_seq, stly_seq_path_pp, stly_info_path_pp = get_postprocessing_stly_paths(path_ac_save, individual, date)

                    # read ruleset for post processing rules
                    ruleset_df = night_info["postprocessing_rules"][individual][mode]

                    # create post processor object
                    post_processor = PostProcessor(
                        ruleset_df, "min_length", SECONDS_PER_INTERVAL, stly_seq
                    )

                    # filter out short phases in behavior sequence
                    pp_stly_seq, pp_stly_time = post_processor.filter_short_phases(
                        post_processor.original_behavior_sequence,
                        post_processor.original_time_sequence,
                        post_processor.rule_set_time,
                        post_processor.rule_set_behavior,
                    )

                    # save post processed sequence as csv file
                    post_processor.save_post_processed_sequence(
                        pp_stly_seq,
                        pp_stly_time,
                        stly_seq_path_pp,
                        stly_info_path_pp,
                        night_info["eval_start"]
                    )

                elif (
                        mode == "LHULHD" or mode == "StFo"
                ):
                    stfo_seq, lhulhd_seq, all_modes_seq_path_pp, all_modes_info_path_pp = get_postprocessing_subaction_paths(path_ac_save, individual, date)

                    # create post processor subaction object and incorporate sequence
                    pp_subactions = PostProcessorSubactions(
                        pp_stly_seq, pp_stly_time, SECONDS_PER_INTERVAL
                    )
                    pp_behav_seq = pp_subactions.incorporate_subactions_sequence(
                        pp_subactions.stly_behavior, stfo_seq, lhulhd_seq
                    )

                    # set ruleset for subaction StFo
                    ruleset_stfo_df = night_info["postprocessing_rules"][individual]["StFo"]
                    ruleset_lhulhd_df = night_info["postprocessing_rules"][individual]["LHULHD"]
                    ruleset_subactions = pd.concat(
                        [ruleset_lhulhd_df, ruleset_stfo_df], ignore_index=True
                    )

                    # create post processor object
                    post_processor_subaction = PostProcessor(
                        ruleset_subactions,
                        "min_length",
                        SECONDS_PER_INTERVAL,
                        pp_behav_seq,
                    )

                    # filter short phases for the subaction
                    pp_subaction_seq, pp_subaction_time = (
                        post_processor_subaction.filter_short_phases(
                            post_processor_subaction.original_behavior_sequence,
                            post_processor_subaction.original_time_sequence,
                            post_processor_subaction.rule_set_time,
                            post_processor_subaction.rule_set_behavior,
                        )
                    )

                    pp_subaction_seq = [int(action) for action in pp_subaction_seq]

                    # save post processed sequence with subactions as csv file
                    post_processor_subaction.save_post_processed_sequence(
                        pp_subaction_seq,
                        pp_subaction_time,
                        all_modes_seq_path_pp,
                        all_modes_info_path_pp,
                        night_info["eval_start"]

                    )





def get_postprocessing_stly_paths(path_ac_save, individual, date):
    """
    Create input and output path for StLy post processing
    """
    # input paths with behavior sequences
    stly_seq_path = f"{path_ac_save}{individual}/raw/StLy/prediction/{date}_{individual}_StLy_behavior_seq.csv"

    ensure_directory(
        f"{path_ac_save}{individual}/post_processed/StLy/"
    )

    # output paths for post processed behavior sequences and information
    stly_seq_path_pp = f"{path_ac_save}{individual}/post_processed/StLy/{date}_{individual}_StLy_behavior_seq_post_processed.csv"
    stly_info_path_pp = f"{path_ac_save}{individual}/post_processed/StLy/{date}_{individual}_StLy_behavior_info_post_processed.csv"

    # dataframe for behavior sequence
    stly_df = pd.read_csv(stly_seq_path, header=None)
    stly_seq = stly_df[0].tolist()

    return stly_seq, stly_seq_path_pp, stly_info_path_pp

def get_postprocessing_subaction_paths(path_ac_save, individual, date):
    """
    Create input and output path for subaction post processing
    """
    # input paths with behavior sequences
    stfo_seq_path = f"{path_ac_save}{individual}/raw/StFo/prediction/{date}_{individual}_StFo_behavior_seq.csv"
    lhulhd_seq_path = f"{path_ac_save}{individual}/raw/LHULHD/prediction/{date}_{individual}_LHULHD_behavior_seq.csv"

    ensure_directory(
     f"{path_ac_save}{individual}/post_processed/"
    )

    # output paths for post processed behavior sequences and information
    all_modes_seq_path_pp = f"{path_ac_save}{individual}/post_processed/{date}_{individual}_behavior_seq_post_processed.csv"
    all_modes_info_path_pp = f"{path_ac_save}{individual}/post_processed/{date}_{individual}_behavior_info_post_processed.csv"

    # data frame for available subaction sequence
    try:
        stfo_df = pd.read_csv(stfo_seq_path, header=None)
        stfo_seq = stfo_df[0].tolist()
    except:
        stfo_seq = []
        print("Warning: StFo not available.")
    try:
        lhulhd_df = pd.read_csv(lhulhd_seq_path, header=None)
        lhulhd_seq = lhulhd_df[0].tolist()
    except:
        lhulhd_seq = []
        print("Warning: LHULHD not available.")

    return stfo_seq, lhulhd_seq, all_modes_seq_path_pp, all_modes_info_path_pp


def detect_moving(batch_processing_nights, path_ac_save, server_od):
    """
    Detect if animal is moving from one timepoint to the next.
    #TODO: incorporate timepoints, where iou value is < thershold, into sequence as moving
    """

    # get information for moving detection from each night
    for night_info in batch_processing_nights:
        date = night_info["date"]
        enclosure = night_info["enclosure_id"]
        individuals = night_info["individual_moving"]
        for individual in individuals:
            # load the correct post processed behavior sequence
            stly_info_path_pp = f"{path_ac_save}{individual}/post_processed/StLy/{date}_{individual}_StLy_behavior_info_post_processed.csv"
            all_modes_info_path_pp = f"{path_ac_save}{individual}/post_processed/{date}_{individual}_behavior_info_post_processed.csv"
            if os.path.exists(all_modes_info_path_pp):
                pp_df = pd.read_csv(all_modes_info_path_pp)
            else:
                pp_df = pd.read_csv(stly_info_path_pp)
            #TODO: detect variable machen (das auch segment funktioniert)
            # load csv file with bounding box positions for each timepoint
            bbox_positions_path= f"{server_od}/detect/{enclosure}/{date}/{individual}/{date}_{individual}_boundingbox-positions.csv"
            bbox_df = pd.read_csv(bbox_positions_path)

            # access all timepoints
            for row in pp_df.index:
                behavior = int(pp_df.loc[row, 'seq_behavior'])
                # moving can only be detect if animal is standing (1)
                if behavior:
                    # time formats needed, for identical comparison
                    fmt = "%H:%M:%S"
                    fmt2 = "%H%M%S"

                    # read start and endtime and convert them into H:M:S-format
                    start = datetime.strptime(pp_df.loc[row, 'start'], fmt)
                    end = datetime.strptime(pp_df.loc[row, 'end'], fmt)

                    # get length of an interval
                    interval = timedelta(seconds=SECONDS_PER_INTERVAL)

                    # transform date from str to datetime format
                    date_fmt = datetime.strptime(date, "%Y-%m-%d")

                    # if end time is after midnight add one day to the date
                    if start > end:
                        end += timedelta(days=1)

                    timepoints_list = []
                    current = start

                    # iterate over each timepoint in standing timeframe up to the end time of evaluation
                    while current < end:
                        # to every timepoint after midnight a day is added to the date
                        if current.time() < time(7, 0, 0):
                            current_date = date_fmt + timedelta(days=1)
                        else:
                            current_date = date_fmt

                        # create timepoint string in the correct format corresponding to names of the bbox images
                        date_timepoints = current_date.strftime("%Y%m%d")
                        timepoint = date_timepoints + "-" + current.strftime(fmt2)

                        # save the timepoints in a list
                        timepoints_list.append(timepoint)

                        # go to next timepoint by adding the given interval
                        current += interval

                    # iterate over all timepoints to compare consecutive timepoints and calculate their iou
                    for i in range(len(timepoints_list)-1):
                        # get first image
                        image_1 = timepoints_list[i]
                        # calculate bounding box information to calculate iou value for the first image
                        xcenter_1 = bbox_df.loc[bbox_df["time"] == image_1, "xcenter"].values[0]
                        ycenter_1 = bbox_df.loc[bbox_df["time"] == image_1, "ycenter"].values[0]
                        width_1 = bbox_df.loc[bbox_df["time"] == image_1, "width"].values[0]
                        height_1 = bbox_df.loc[bbox_df["time"] == image_1, "height"].values[0]
                        print(i)
                        # get second image
                        image_2 = timepoints_list[i+1]
                        # calculate bounding box information to calculate iou value for the first image
                        xcenter_2 = bbox_df.loc[bbox_df["time"] == image_2, "xcenter"].values[0]
                        ycenter_2 = bbox_df.loc[bbox_df["time"] == image_2, "ycenter"].values[0]
                        width_2 = bbox_df.loc[bbox_df["time"] == image_2, "width"].values[0]
                        height_2 = bbox_df.loc[bbox_df["time"] == image_2, "height"].values[0]

                        # create tuples including all information for each bounding box
                        bbox1 = (xcenter_1, ycenter_1, width_1, height_1)
                        bbox2 = (xcenter_2, ycenter_2, width_2, height_2)

                        # call function to calculate iou values
                        iou = intersection_over_union(bbox1, bbox2)
                        print(f"IoU: {iou:.2f}")


def bbox_from_center(x_center, y_center, width, height):
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return x_min, y_min, x_max, y_max

def intersection_over_union(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox_from_center(*bbox1)
    x2_min, y2_min, x2_max, y2_max = bbox_from_center(*bbox2)

    # Berechnung der Schnittmenge
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Flächen der Bounding Boxes
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # IoU berechnen
    iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0
    return iou


def create_statistics(batch_processing_nights, path_stats, path_ac_save):
    """
        Collect needed information to create statistics for StLy and if required subactions.
    """
    # get information for statistics from each night
    for night_info in batch_processing_nights:
        date = night_info["date"]

        for individual, stats_type in night_info["statistic_tasks"].items():
            # select individual for which statistics should be applied

            if any(stats_type):
                df_pp, df_pp_stly, phases_stly_file, phases_subactions_file, output_stats_path = get_statistics_paths(path_stats, individual, date, path_ac_save, stats_type)


                # apply statistic evaluation for standing and lying phases
                df_phases_stly = statistics.create_phase_csv(
                    df_pp_stly, date, individual, phases_stly_file
                )
                # apply statistic evaluation for subaction phases
                if stats_type[1]:
                    df_phases_subactions = statistics.create_phase_csv(
                        df_pp, date, individual, phases_subactions_file
                    )

                # get all standing cycles
                standing_list = statistics.get_cycles_behavior(
                    df_pp_stly, date, individual, 1
                )
                # get all lying cycles
                lying_list = statistics.get_cycles_behavior(
                    df_pp_stly, date, individual, 2
                )

                # path to save cycles file
                cycle_file = f"{output_stats_path}{date}_{individual}_cycles.csv"

                # create cycle file
                df_cycles = statistics.create_cycle_csv(
                    standing_list,
                    lying_list,
                    df_pp_stly["seq_behavior"][0],
                    df_pp_stly["seq_behavior"][1],
                    cycle_file,
                )

                # path to save key values file
                key_values_file = f"{output_stats_path}{date}_{individual}_key_values.csv"

                # create key values file (no subactions)
                if not stats_type[1]:
                    statistics.create_key_values_csv(
                        df_cycles,
                        df_phases_stly,
                        False,
                        standing_list[5],
                        lying_list[5],
                        key_values_file,
                    )
                # create key values file (with subactions)
                if stats_type[1]:
                    statistics.create_key_values_csv(
                        df_cycles,
                        df_phases_subactions,
                        True,
                        standing_list[5],
                        lying_list[5],
                        key_values_file
                    )


def get_statistics_paths(path_stats, individual, date, path_ac_save, stats_type):
    """
    Create input and output paths for statistics
    """
    # set paths where statistic is saved
    output_stats_path = f"{path_stats}details/{individual}/"
    ensure_directory(output_stats_path)
    phases_stly_file = (
        f"{output_stats_path}{date}_{individual}_phases_stly.csv"
    )
    # input paths for StLy
    path_StLy = f"{path_ac_save}{individual}/post_processed/StLy/{date}_{individual}_StLy_behavior_info_post_processed.csv"
    # read input dataframe
    df_pp_stly = pd.read_csv(path_StLy)

    if stats_type[1]:
        phases_subactions_file = (
            f"{output_stats_path}{date}_{individual}_phases_subactions.csv"
        )
        # input paths for subactions
        path_pp = f"{path_ac_save}{individual}/post_processed/{date}_{individual}_behavior_info_post_processed.csv"
        # read input dataframe for subactions
        df_pp = pd.read_csv(path_pp)
    else:
        phases_subactions_file = ''
        df_pp = pd.DataFrame()

    return df_pp, df_pp_stly, phases_stly_file, phases_subactions_file, output_stats_path


def create_visualization(batch_processing_nights, path_stats, path_visual, individual_information):
    """
        Collect needed information to create visualization for StLy and if required subactions.
    """
    # lists to save behaviors for y axis
    behaviors_to_plot = []
    behaviors_to_plot_sub = []
    datetime_format = "%Y-%m-%d %H:%M:%S"
    for night_info in batch_processing_nights:
        date = night_info["date"]

        for individual, stats_type in night_info["statistic_tasks"].items():
            # create dicts to store stly, subactions and the combination of both
            stly_visualizer = {v: [] for v in BEHAVIOR_STATISTICS.values()}
            subactions_visualizer = {v: [] for v in BEHAVIOR_STATISTICS.values()}
            total_visualizer = {v: [] for v in BEHAVIOR_STATISTICS.values()}

            # path to get statistics
            output_stats_path = f"{path_stats}details/{individual}/"
            ensure_directory(output_stats_path)

            # check if StLy visualization should be done
            if stats_type[0]:
                phases_stly_file = (
                    f"{output_stats_path}{date}_{individual}_phases_stly.csv"
                )
                df_phases_stly = pd.read_csv(phases_stly_file)

                # append information to visualize StLy
                stly_visualizer, total_visualizer = get_stly_visualization_info(
                    df_phases_stly, datetime_format, stly_visualizer, total_visualizer)
                for key, value in total_visualizer.items():
                    if value:
                        behaviors_to_plot.append(key)
            else:
                continue

            # check if subactions visualization should be done
            if stats_type[1]:
                phases_subactions_file = (
                    f"{output_stats_path}{date}_{individual}_phases_subactions.csv"
                )

                df_phases_subactions = pd.read_csv(phases_subactions_file)

                # append information to visualize subactions and combination
                subactions_visualizer, total_visualizer = get_subactions_visualization_info(
                    df_phases_subactions, datetime_format, subactions_visualizer, total_visualizer)
                for key, value in subactions_visualizer.items():
                    if value:
                        behaviors_to_plot_sub.append(key)

            # visualization save paths
            output_visual_path = f"{path_visual}{individual}/"
            ensure_directory(output_visual_path)
            image_total = f'{output_visual_path}{date}_{individual}_visualization_total.png'
            image_binary = f'{output_visual_path}{date}_{individual}_visualization_binary.png'
            image_sub = f'{output_visual_path}{date}_{individual}_visualization_subactions.png'

            ind_name = individual_information.loc[individual, 'individual_name']

            # timepoints for x axis
            plot_start = date + ' ' + df_phases_stly["start"][0]
            plot_start_datetime = datetime.strptime(plot_start, datetime_format)
            plot_end = date + ' ' + df_phases_stly["end"].iloc[-1]
            # calculate time axis for plot, assumption: end time must be after date change
            plot_end_datetime = datetime.strptime(plot_end, datetime_format) + timedelta(days=1)

            # create visualization
            NightVisualizer.plot_nocturnal_sequence(
                stly_visualizer,
                subactions_visualizer,
                total_visualizer,
                behaviors_to_plot,
                behaviors_to_plot_sub,
                image_total,
                image_binary,
                image_sub,
                ind_name,
                plot_start_datetime,
                plot_end_datetime,
                stats_type[1]
            )


def get_stly_visualization_info(df_phases_stly, datetime_format, stly_visualizer, total_visualizer):
    """
        Get time information for StLy visualization.
    """
    next_day = False
    for row in df_phases_stly.index:
        # set start and end time
        datetime_start_str = df_phases_stly["date"][row] + ' ' + df_phases_stly["start"][row]
        datetime_start = datetime.strptime(datetime_start_str, datetime_format)
        datetime_end_str = df_phases_stly["date"][row] + ' ' + df_phases_stly["end"][row]
        datetime_end = datetime.strptime(datetime_end_str, datetime_format)

        # check  if end time is on next day
        if datetime_end < datetime_start or next_day:
            datetime_end += timedelta(days=1)
            if next_day:
                datetime_start += timedelta(days=1)
            next_day = True

        # append time intervals in which stly is observed
        time_tuple = (datetime_start, datetime_end)
        stly_visualizer[df_phases_stly["behavior"][row]].append(time_tuple)
        total_visualizer[df_phases_stly["behavior"][row]].append(time_tuple)

    return stly_visualizer, total_visualizer


def get_subactions_visualization_info(df_phases_subactions, datetime_format, subactions_visualizer, total_visualizer):
    """
        Get time information for subactions visualization.
    """
    next_day = False
    for row in df_phases_subactions.index:
        # set start and end time
        datetime_start_str = df_phases_subactions["date"][row] + ' ' + df_phases_subactions["start"][row]
        datetime_start = datetime.strptime(datetime_start_str, datetime_format)
        datetime_end_str = df_phases_subactions["date"][row] + ' ' + df_phases_subactions["end"][row]
        datetime_end = datetime.strptime(datetime_end_str, datetime_format)

        # check  if end time is on next day
        if datetime_end < datetime_start or next_day:
            datetime_end += timedelta(days=1)
            if next_day:
                datetime_start += timedelta(days=1)
            next_day = True

        # append time intervals in which subactions are observed
        time_tuple = (datetime_start, datetime_end)
        subactions_visualizer[df_phases_subactions["behavior"][row]].append(time_tuple)
        if df_phases_subactions["behavior"][row] != "Out of View":
            total_visualizer[df_phases_subactions["behavior"][row]].append(time_tuple)

    return subactions_visualizer, total_visualizer


def delete_local_data(batch_processing_nights, path_ac_save):
    """
    Delete local data after pipline run finished
    """
    # TODO: noch testen
    """
    for night_info in batch_processing_nights:
        enclosure_id = night_info["enclosure_id"]
        sh.rmtree(f'{night_info["temporal_storage"]}/predicted_images/{enclosure_id}')

        for mode in AC_MODES:

            if mode == 'Moving':
                # moving is processed later
                continue

            individuals = night_info['ac_predictions'][mode]

            for individual in individuals:
                sh.rmtree(f'{path_ac_save}{individual}')

    """

