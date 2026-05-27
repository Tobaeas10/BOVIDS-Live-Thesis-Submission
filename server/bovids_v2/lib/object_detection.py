#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains methods to 
	- detect or segment individuals on a folder of images
		- save the images and the differences image
		- save the bounding box / segment per given image
Methods:
	predict_folder_differences(folder_path, difference_folder_path, output_images, output_segment, enclosure_id, mode = 'detect'):
		- requires input folders that contain images name.jpg in both folders to merge the
		regions identified by the object detector
		- object detector is chosen by mode (detect, segment) and the enclosure_id
"""

__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

# make it possible to import own modules
import os, sys
import time

from image_manipulation import save_image_to_file_unit8
from local_pc.thesis_tests import timer, LapTimer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from server.bovids_v2.lib.func import (
	ensure_directory,
	write_yaml_file,
	get_polygons_from_json,
	polygon_to_bounding_box,
	write_json_file_boundingbox_as_polygon,
	write_json_file_polygon,
)
from server.bovids_v2.config.get_config import get_enclosure_information
from server.bovids_v2.lib.image_manipulation import glue_image_difference

# import yolov8
from ultralytics import YOLO

# import standard libraries
import glob, json, shutil, random
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from copy import copy
import pandas as pd

# set global path to networks
resources_folder = os.path.abspath(os.path.dirname(__file__) + "/../res/")

# Cache models to avoid duplicate loading
cached_od_models = {}


def clean_bounding_boxes(bboxes, possible_labels):
	"""
	Removes duplicate labels, assigns the left over label if only one individual is not found but one twice,
	differently just removes duplicate boxes
	"""
	if len(bboxes) == 1:
		return bboxes

	bboxes = copy(bboxes)
	given_labels = {}
	for j in range(len(bboxes)):
		bbox = bboxes[j]
		if not bbox[0] in given_labels.keys():
			given_labels[bbox[0]] = []
		given_labels[bbox[0]].append((j, bbox[2]))

	if len(given_labels) == len(possible_labels):
		return bboxes

	ret = []
	chosen_bbs = []
	# remove duplicates
	for label in given_labels.keys():
		if len(given_labels[label]) == 1:
			ret.append(bboxes[given_labels[label][0][0]])
			chosen_bbs.append(given_labels[label][0][0])
		else:
			tmp = sorted(given_labels[label], key=lambda a: a[1], reverse=True)
			ret.append(bboxes[tmp[0][0]])
			chosen_bbs.append(tmp[0][0])

	# check if only one label missing
	if len(chosen_bbs) == len(possible_labels) - 1 and len(bboxes) > len(chosen_bbs):
		missing_label = [j for j in possible_labels if not j in [k[0] for k in ret]]
		missing_box = [j for j in range(len(bboxes)) if not j in chosen_bbs]
		ret.append(bboxes[missing_box[0]])


	return ret


def is_strange_bbox(x, y, w, h):
	if h < 15:
		return True
	if w < 15:
		return True
	return False


def _load_model(df_enc_info, enclosure_id, device):
	model = None
	target_model_path = "None"
	try:
		# Optimize inference. ONNX and ENGINE export modes are not compatible to transfer between different CPUs/GPUs!
		# TODO: test if optimized version is stable and equivalent on GPU! Tobi confirmed this for CPU only!
		base_model_path = f"{resources_folder}/objectdetection/{df_enc_info.loc[enclosure_id, 'object_detector']}"
		model_path_pt = f"{base_model_path}.pt"
		model_path_onnx = f"{base_model_path}.onnx"
		model_path_engine = f"{base_model_path}.engine"

		# Use .engine export mode if gpu, else onnx. This improves runtime on both cpu and gpu by approx 2 to 3 times
		is_gpu = str(device).startswith("cuda")
		if is_gpu:
			target_model_path = model_path_engine
			export_format = "engine"
		else:
			target_model_path = model_path_onnx
			export_format = "onnx"

		# Export model if not exported yet
		if not os.path.exists(target_model_path):
			temp_model = YOLO(model_path_pt, task="detect")
			if export_format == "onnx":
				saved_model_path = temp_model.export(
					format="onnx",
					dynamic=True,
					simplify=True,
					opset=17,
					half=False,
					batch=1,
				)
			else:
				saved_model_path = temp_model.export(
					format="engine",
					dynamic=False,  # usually faster
					half=True,  # FP16 acceleration
					workspace=4,  # GB of TensorRT workspace
					batch=1,
					device=device,
				)
			if os.path.abspath(saved_model_path) != os.path.abspath(target_model_path):
				shutil.move(saved_model_path, target_model_path)

		# Load from disk and then save to cache OR load from cache
		global cached_od_models
		if target_model_path in cached_od_models:
			model = cached_od_models[target_model_path]
		else:
			model = YOLO(target_model_path, task="detect")
			cached_od_models[target_model_path] = model
	except Exception as e:
		print(f"ERROR. Could not load model weights ({target_model_path}): {e}")
	return model


def predict_folder_differences_par(args):
	(
		folder_path,
		difference_folder_path,
		output_images,
		output_segment,
		enclosure_id,
		mode,
		device,
		dismiss_individuals,
		date,
	) = args
	predict_folder_differences(
		folder_path,
		difference_folder_path,
		output_images,
		output_segment,
		enclosure_id,
		mode,
		device,
		dismiss_individuals,
		date,
	)

@timer
def predict_folder_differences(
	folder_path,
	difference_folder_path,
	output_images,
	output_segment,
	enclosure_id,
	mode,
	device,
	dismiss_individuals=[],
	date="",
):
	"""
	folder_path: folder with imagefiles enclosure-id_time.jpg
	difference_folder_path: corresponding difference images (same name)
	output_segment: csv_file which will be created containing image_name and the bounding boxes per class, contains image_name, individual_name, xcenter, ycenter, width, height, polygon_points
	enclosure_id is the enclosure_id used to infer the correct detector
	mode: segment or detect
	device: cuda:0, cuda:1 or cpu
	dismiss_individuals: boxes corresponding to those individuals will be ignored
	"""

	df_enc_info = get_enclosure_information()
	if not enclosure_id in df_enc_info.index:
		print(
			f"ERROR: enclosure id has no configuration entry {enclosure_id}. Known enclosures are {df_enc_info.index}"
		)
		return
	if not mode in ["detect", "segment"]:
		print("ERROR: Invalid task chosen for {}".format(enclosure_id))
		return
	elif (
		mode == "detect"
		and not len(df_enc_info.loc[enclosure_id, "object_detector"]) > 0
	):
		print("ERROR: no object detector given for {}".format(enclosure_id))
		return
	elif (
		mode == "segment"
		and not len(df_enc_info.loc[enclosure_id, "image_segmentor"]) > 0
	):
		print("ERROR: no image segmentor given for {}".format(enclosure_id))
		return
	elif mode == "detect" and not (
		os.path.exists(
			resources_folder
			+ "/objectdetection/"
			+ df_enc_info.loc[enclosure_id, "object_detector"]
			+ ".pt"
		)
	):
		print("ERROR: object detector does not exist ({})".format(enclosure_id))
		return
	elif mode == "segment" and not (
		os.path.exists(
			resources_folder
			+ "/imagesegmentation/"
			+ df_enc_info.loc[enclosure_id, "image_segmentor"]
			+ ".pt"
		)
	):
		print("ERROR: image segmentor does not exist ({})".format(enclosure_id))
		return

	ensure_directory(output_images)
	if output_segment != None:
		ensure_directory(output_segment)

	bounding_information = {
		"IndividualID": [],
		"time": [],
		"xcenter": [],
		"ycenter": [],
		"width": [],
		"height": [],
		"polygon_points": [],
	}
	existing_individuals = []

	if mode == "detect":
		model = _load_model(df_enc_info, enclosure_id, device)

		# do the actual inference per interval
		results = model(
			folder_path,
			stream=True,
			max_det=len(model.names),
			device=device,
			conf=float(df_enc_info.loc[enclosure_id, "od_confidence"]),
			iou=float(df_enc_info.loc[enclosure_id, "od_iou"]),
			verbose=False,
		)

		if len(model.names) == 1:
			# only one individual in this enclosure
			individual_base_name = df_enc_info.loc[enclosure_id, "individual_ids"]
		else:
			individual_base_name = ""

		stats = LapTimer("Object Detection Loop - 1 image each")
		for res in results:
			with stats:
				res = res.cpu()
				# get bounding boxes with respective classnames
				bboxes = []
				for j in range(len(res.boxes.data)):
					bbox = res.boxes.xywh[j].numpy().astype("uint16")
					if is_strange_bbox(*bbox):
						continue
					classid = int(res.boxes.cls[j])
					classname = res.names[classid]
					certainty = res.boxes.conf[j]
					bboxes.append([classname, bbox, certainty])
				img_name = os.path.basename(res.path)  # enc-id_yyyymmdd-mmhhss.jpg
				difference_image = difference_folder_path + img_name
				bboxes = clean_bounding_boxes(bboxes, model.names)


				if not os.path.exists(difference_image):
					continue


				img_dict, bb_coordinates, polygon_coordinates = glue_image_difference(
					res.path, difference_image, bboxes, None
				)
				# save the images
				time_val = img_name.split("_")[-1][:-4]
				for individualid, img_arr in img_dict.items():

					used_individual_id = individualid
					if individualid == "Ungulate":
						used_individual_id = individual_base_name

					if not used_individual_id in existing_individuals:
						existing_individuals.append(used_individual_id)
					# 0 added to path so that ac works properly
					ensure_directory(f"{output_images}{used_individual_id}/images/0/")
					save_image_to_file_unit8(img_arr, f"{output_images}{used_individual_id}/images/0/{time_val}_{used_individual_id}.jpg")

					bounding_information["IndividualID"].append(used_individual_id)
					bounding_information["time"].append(time_val)
					bounding_information["xcenter"].append(bb_coordinates[classname][0])
					bounding_information["ycenter"].append(bb_coordinates[classname][1])
					bounding_information["width"].append(bb_coordinates[classname][2])
					bounding_information["height"].append(bb_coordinates[classname][3])
					bounding_information["polygon_points"].append(None)

	if mode == "segment":
		model = YOLO(
			resources_folder
			+ "/imagesegmentation/"
			+ df_enc_info.loc[enclosure_id, "image_segmentor"]
			+ ".pt",
			task="segment",
		)
		# detect bounding boxes and masks
		results = model(
			folder_path,
			stream=True,
			max_det=len(model.names),
			device=device,
			conf=float(df_enc_info.loc[enclosure_id, "seg_confidence"]),
			iou=float(df_enc_info.loc[enclosure_id, "seg_iou"]),
			verbose=False,
		)

		if len(model.names) == 1:
			# only one individual in this enclosure
			individual_base_name = df_enc_info.loc[enclosure_id, "individual_ids"]
		else:
			individual_base_name = ""

		for res in results:
			res = res.cpu()
			# get segmented areas with respective classnames
			segments = []
			bboxes = []
			for j in range(len(res.boxes.data)):
				if res.masks == None:
					continue
				bbox = res.boxes.xywh[j].numpy().astype("uint16")
				if is_strange_bbox(*bbox):
					continue

				segment = res.masks.xy[j]
				classid = int(res.boxes.cls[j])
				classname = res.names[classid]
				certainty = res.boxes.conf[j]
				segments.append(segment)
				bboxes.append([classname, bbox, certainty])

			img_name = os.path.basename(res.path)  # enc-id_yyyymmdd-mmhhss.jpg
			difference_image = difference_folder_path + img_name
			bboxes = clean_bounding_boxes(bboxes, model.names)

			if not os.path.exists(difference_image):
				continue

			img_dict, bb_coordinates, polygon_coordinates = glue_image_difference(
				res.path, difference_image, bboxes, segments
			)

			# save the images
			time_val = img_name.split("_")[-1][:-4]
			for individualid, img_arr in img_dict.items():

				used_individual_id = individualid
				if individualid == "Ungulate":
					used_individual_id = individual_base_name

				if not used_individual_id in existing_individuals:
					existing_individuals.append(used_individual_id)

				ensure_directory(f"{output_images}{used_individual_id}/images/0/")
				imsave(
					f"{output_images}{used_individual_id}/images/0/{time_val}_{used_individual_id}.jpg",
					img_arr,
				)

				bounding_information["IndividualID"].append(used_individual_id)
				bounding_information["time"].append(time_val)
				bounding_information["xcenter"].append(bb_coordinates[classname][0])
				bounding_information["ycenter"].append(bb_coordinates[classname][1])
				bounding_information["width"].append(bb_coordinates[classname][2])
				bounding_information["height"].append(bb_coordinates[classname][3])
				bounding_information["polygon_points"].append(
					polygon_coordinates[classname]
				)

	if output_segment != None:
		df_output = pd.DataFrame(bounding_information)
		for individual_id in existing_individuals:
			df_output[df_output["IndividualID"] == individual_id].to_csv(
				f"{output_segment}{individual_id}/{date}_{individual_id}_boundingbox-positions.csv",
				index=False,
			)


def predict_folder_raw(
	task,
	input_images_to_detect,
	images_to_predict,
	enclosure_id,
	device,
	do_clean_boxes,
):
	"""
	Method used in create_annotate_images_from_video.
	Given a base folder and a list of images that need to be predicted, outputs json files with bounding boxes / segments
	task: detect or segment, depends which network is chosen
	input_images_to_detect: path that contains subfolders of the form enclosure_id/images/
	images_to_predict: list of images (paths) that will be predicted
	enclosure_id: enclosure id used to identify networks
	device: cuda:0, cuda:1, or cpu (may vary from system to system)
	do_clean_boxes: decide whether postprocessing should be applied to the boxes
	"""

	df_enc_info = get_enclosure_information([enclosure_id])

	# load model and do inference
	if task == "segment":
		model = YOLO(
			resources_folder
			+ "/imagesegmentation/"
			+ df_enc_info.loc[enclosure_id, "image_segmentor"]
			+ ".pt",
			task="segment",
		)
		# detect bounding boxes and masks
		for img_path in tqdm(images_to_predict):
			res = model(
				img_path,
				max_det=len(model.names),
				device=device,
				conf=float(df_enc_info.loc[enclosure_id, "seg_confidence"]),
				iou=float(df_enc_info.loc[enclosure_id, "seg_iou"]),
				verbose=False,
			).cpu()

			segments = []
			for j in range(len(res[0].boxes.data)):

				if res[0].masks == None:
					continue
				bbox = res[0].boxes.xywh[j].numpy().astype("uint16")
				if is_strange_bbox(*bbox):
					continue
				segment = res[0].masks.xy[j]
				classid = int(res[0].boxes.cls[j])
				classname = res[0].names[classid]
				certainty = res[0].boxes.conf[j]
				segments.append([classname, segment, certainty])

			# write json file
			if len(segments) == 0:
				continue

			if do_clean_boxes:
				segments = clean_bounding_boxes(segments, model.names)

			segments = [seq[:2] for seq in segments]
			json_path = (
				input_images_to_detect
				+ enclosure_id
				+ "/{}/{}.json".format(
					df_enc_info.loc[enclosure_id, "image_segmentor"],
					os.path.basename(res[0].path).split(".")[0],
				)
			)
			relative_img_path = "../images/{}".format(os.path.basename(res[0].path))
			write_json_file_polygon(
				json_path, segments, relative_img_path, res[0].orig_shape
			)

	elif task == "detect":
		model = YOLO(
			resources_folder
			+ "/objectdetection/"
			+ df_enc_info.loc[enclosure_id, "object_detector"]
			+ ".pt",
			task="detect",
		)
		# model.names is dictionary {class_id : 'classname'}
		# if complete folder of images, it is possible to stream results: results = model(inputs, stream=True) where inputs is a folder consisting of image files (need later in prediction pipeline)
		# res[0].boxes.xywh list of bounding boxes xcenter, ycenter, widht, height (not normalized)
		# res[0].boxes.cls list of class ids
		# res[0].names dictionary of classnames
		# res[0].masks.masks list of polygons if segmentation
		# res[0].orig_shape = (w,h) of original image
		for img_path in tqdm(images_to_predict):

			# detect bounding boxes
			res = model(
				img_path,
				max_det=len(model.names),
				device=device,
				conf=float(df_enc_info.loc[enclosure_id, "od_confidence"]),
				iou=float(df_enc_info.loc[enclosure_id, "od_iou"]),
				verbose=False,
			).cpu()

			bboxes = []
			for j in range(len(res[0].boxes.data)):
				bbox = res[0].boxes.xywh[j].numpy().astype("uint16")
				if is_strange_bbox(*bbox):
					continue
				classid = int(res[0].boxes.cls[j])
				classname = res[0].names[classid]
				certainty = res[0].boxes.conf[j]
				bboxes.append([classname, bbox, certainty])

			# write json file
			if len(bboxes) == 0:
				continue

			if do_clean_boxes:
				bboxes = clean_bounding_boxes(bboxes, model.names)

			bboxes = [box[:2] for box in bboxes]
			json_path = (
				input_images_to_detect
				+ enclosure_id
				+ "/{}/{}.json".format(
					df_enc_info.loc[enclosure_id, "object_detector"],
					os.path.basename(res[0].path).split(".")[0],
				)
			)
			relative_img_path = "../images/{}".format(os.path.basename(res[0].path))
			write_json_file_boundingbox_as_polygon(
				json_path, bboxes, relative_img_path, res[0].orig_shape
			)
