__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"


from server.bovids_v2.lib.func import (
    ensure_directory, zip_od_images,
    zip_pp,
    _copy_file
)
from server.bovids_v2.config.get_config import AC_MODES
from server.bovids_v2.lib.availability_checks import _requires_od_images

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, cpu_count
import shutil as sh
import os

def copy_videos_to_local_storage(batch_processing_nights):
    videos_to_copy = []
    for night_info in batch_processing_nights:
        if night_info["perform_od"]:
            if not night_info["copy_videos"]:
                continue
            vid_names = [os.path.basename(x) for x in night_info["video_list"]]
            for j in range(len(vid_names)):
                videos_to_copy.append(
                    [
                        night_info["video_list"][j],
                        f'{night_info["temporal_storage"]}videofiles/{vid_names[j]}',
                    ]
                )
    if len(videos_to_copy) > 0:
        _ = process_map(_copy_file, videos_to_copy, max_workers=cpu_count(),
                        desc='Copying video files to local storage ')
    return len(videos_to_copy)


def copy_od_to_server(od_conduct_folders, batch_processing_nights):
    folders_to_zip = []
    csv_files_copy = []
    if len(od_conduct_folders) > 0:

        for night_info in batch_processing_nights:
            enclosure_id = night_info["enclosure_id"]
            od_output_folder_date = f'{night_info["temporal_storage"]}/predicted_images/{enclosure_id}/{night_info["date"]}/'
            date = night_info["date"]

            for individual_id in os.listdir(od_output_folder_date):
                if not os.path.exists(
                        f"{od_output_folder_date}{individual_id}/images/"
                ):
                    continue
                if not os.path.exists(
                        f"{od_output_folder_date}{individual_id}/{date}_{individual_id}_boundingbox-positions.csv"
                ):
                    print(
                        f"ERROR. {od_output_folder_date}{individual_id}/{date}_{individual_id}_boundingbox-positions.csv not found. Cannot copy od images to server storage."
                    )
                folders_to_zip.append(
                    [
                        f"{od_output_folder_date}{individual_id}/images/",
                        f"{od_output_folder_date}{individual_id}/{date}_{individual_id}_odimages",
                        f'{night_info["savepath_od"]}{individual_id}/{date}_{individual_id}_odimages',
                    ]
                )

                csv_files_copy.append(
                    [
                        f"{od_output_folder_date}{individual_id}/{date}_{individual_id}_boundingbox-positions.csv",
                        f'{night_info["savepath_od"]}{individual_id}/{date}_{individual_id}_boundingbox-positions.csv',
                    ]
                )
    if len(folders_to_zip) > 0:
        # zip files
        _ = process_map(
            zip_od_images,
            folders_to_zip,
            max_workers=cpu_count(),
            desc="Compress images to store on server ",
        )
        # copy to server
        for j_copy in tqdm(
                range(len(folders_to_zip)), desc="Copy archives to server "
        ):
            ensure_directory(csv_files_copy[j_copy][1])
            sh.copy2(
                src=folders_to_zip[j_copy][1] + ".zip",
                dst=folders_to_zip[j_copy][2] + ".zip",
            )
            sh.copy2(src=csv_files_copy[j_copy][0], dst=csv_files_copy[j_copy][1])


def copy_od_to_local_storage(batch_processing_nights):
    zip_files_copy = []
    csv_files_copy = []
    for night_info in batch_processing_nights:
        for individual_id in night_info["copy_od_server_to_tmp_images"]:
            if _requires_od_images(
                    [
                        ac_mode
                        for ac_mode in AC_MODES
                        if individual_id in night_info["ac_predictions"][ac_mode]
                    ]
            ):
                zip_files_copy.append(
                    [
                        f'{night_info["savepath_od"]}{individual_id}/{night_info["date"]}_{individual_id}_odimages.zip',
                        f'{night_info["temporal_storage"]}/predicted_images/{individual_id}/{night_info["date"]}/{night_info["date"]}_{individual_id}_odimages.zip',
                    ]
                )
            csv_files_copy.append(
                [
                    f'{night_info["savepath_od"]}{individual_id}/{night_info["date"]}_{individual_id}_boundingbox-positions.csv',
                    f'{night_info["temporal_storage"]}/predicted_images/{individual_id}/{night_info["date"]}/{night_info["date"]}_{individual_id}_boundingbox-positions.csv',
                ]
            )
    for csv_file_copy in csv_files_copy:
        ensure_directory(csv_file_copy[1])
        sh.copy2(src=csv_file_copy[0], dst=csv_file_copy[1])
    for zip_file in tqdm(
            zip_files_copy, desc="Copy od images from server to local storage "
    ):
        ensure_directory(zip_file[1])
        sh.copy2(src=zip_file[0], dst=zip_file[1])
        sh.unpack_archive(
            zip_file[1],
            f'{night_info["temporal_storage"]}/predicted_images/{night_info["enclosure_id"]}/{night_info["date"]}/{individual_id}/images/0/',
            "zip",
        )  # 0 hinzugefügt


def copy_ac_to_server(batch_processing_nights, path_ac_save):
    counter = 0
    csv_files_copy = []
    csv_files_seq = []

    for night_info in batch_processing_nights:

        date = night_info["date"]

        # set mode that is detected for every individual
        for mode in AC_MODES:

            if mode == "Moving":
                # moving is processed later
                continue
            individuals = night_info["ac_predictions"][mode]

            for individual_id in individuals:
                if not os.path.exists(
                        f"{path_ac_save}{individual_id}/raw/{mode}/prediction/"
                ):
                    continue
                ensure_directory(
                    f'{batch_processing_nights[counter]["savepath_ac"][individual_id]}/raw/{mode}/prediction/'
                )
                csv_files_copy.append(
                    [
                        f"{path_ac_save}{individual_id}/raw/{mode}/prediction/{date}_{individual_id}_{mode}.csv",
                        f'{batch_processing_nights[counter]["savepath_ac"][individual_id]}/raw/{mode}/prediction/{date}_{individual_id}_{mode}.csv',
                    ]
                )
                csv_files_seq.append(
                    [
                        f"{path_ac_save}{individual_id}/raw/{mode}/prediction/{date}_{individual_id}_{mode}_behavior_seq.csv",
                        f'{batch_processing_nights[counter]["savepath_ac"][individual_id]}/raw/{mode}/prediction/{date}_{individual_id}_{mode}_behavior_seq.csv',
                    ]
                )
        counter += 1
    if len(csv_files_copy) > 0:
        # copy to server
        for j_copy in tqdm(
                range(len(csv_files_copy)), desc="Copy archives to server "
        ):
            # ensure_directory(csv_files_copy[j_copy][1])
            sh.copy2(src=csv_files_copy[j_copy][0], dst=csv_files_copy[j_copy][1])
            sh.copy2(src=csv_files_seq[j_copy][0], dst=csv_files_seq[j_copy][1])


def copy_ac_to_local_storage(batch_processing_nights, path_ac_save):
    csv_files_copy = []
    csv_files_seq = []
    counter = 0

    for night_info in batch_processing_nights:
        # for individual_id in night_info['copy_od_server_to_tmp_images']:
        date = night_info["date"]

        for mode in AC_MODES:
            if mode == "Moving":
                # moving is processed later
                continue
            individuals = night_info["ac_predictions"][mode]

            for individual_id in individuals:
                # if _requires_od_images([ac_mode for ac_mode in AC_MODES if
                #                  individual_id in night_info['ac_predictions'][ac_mode]]):
                csv_files_copy.append(
                    [
                        f'{batch_processing_nights[counter]["savepath_ac"][individual_id]}/raw/{mode}/prediction/{date}_{individual_id}_{mode}.csv',
                        f"{path_ac_save}{individual_id}/raw/{mode}/prediction/{date}_{individual_id}_{mode}.csv",
                    ]
                )
                csv_files_seq.append(
                    [
                        f'{batch_processing_nights[counter]["savepath_ac"][individual_id]}/raw/{mode}/prediction/{date}_{individual_id}_{mode}_behavior_seq.csv',
                        f"{path_ac_save}{individual_id}/raw/{mode}/prediction/{date}_{individual_id}_{mode}_behavior_seq.csv",
                    ]
                )
        counter += 1
    for csv_file_copy in csv_files_copy:
        ensure_directory(csv_file_copy[1])
        sh.copy2(src=csv_file_copy[0], dst=csv_file_copy[1])
    for csv_files_seq in csv_files_seq:
        sh.copy2(src=csv_files_seq[0], dst=csv_files_seq[1])

def copy_postprocessing_to_server(batch_processing_nights, path_ac_save):
    folders_to_zip = []

    for night_info in batch_processing_nights:
        for individual in night_info["individual_postprocessing"].keys():
            if night_info["individual_postprocessing"][individual]:
                if not os.path.exists(
                        f"{batch_processing_nights}{individual}/post_processed/"
                ):
                    continue
                folders_to_zip.append(
                    [
                        f"{path_ac_save}{individual}/post_processed/",
                        f"{path_ac_save}{individual}/post_processed_zipped/",
                        f'{night_info["savepath_ac"][individual]}/post_processed',
                    ]
                )

    if len(folders_to_zip) > 0:
        # zip files
        _ = process_map(
            zip_pp,
            folders_to_zip,
            max_workers=cpu_count(),
            desc="Compress images to store on server ",
        )
        # copy to server
        for j_copy in tqdm(
                range(len(folders_to_zip)), desc="Copy archives to server "
        ):
            sh.copy2(
                src=folders_to_zip[j_copy][1] + ".zip",
                dst=folders_to_zip[j_copy][2] + ".zip",
            )

            sh.unpack_archive(
                folders_to_zip[j_copy][2] + ".zip", folders_to_zip[j_copy][2]
            )