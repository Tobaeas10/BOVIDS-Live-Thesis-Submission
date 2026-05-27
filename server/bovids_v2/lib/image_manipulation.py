#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains methods to merge video streams, add blacked out regions, ...
"""

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

# make it possible to import own modules
import sys, os
import time

import imageio

from local_pc.thesis_tests import timer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from server.bovids_v2.config.get_config import (
	get_black_polygons,
	IMAGE_HEIGHT_ACTION_CLASSIFICATION,
	IMAGE_WIDTH_ACTION_CLASSIFICATION,
	use_frames_per_interval,
)
from server.bovids_v2.lib.func import ensure_directory

import numpy as np
from skimage.transform import resize  # scikit-image
from skimage.draw import polygon
from skimage.io import imread

from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta
from skimage.io import imsave
from skimage.color import rgb2gray, gray2rgb
from skimage.exposure import adjust_gamma
from skimage.util import img_as_float
import cv2


cached_polygons = {}


def fetch_images_for_frame(frame, video_files, enc_id, start_date):

	try:
		images = [clip.get_frame(frame) for clip in video_files]
		images = [adjust_gamma(gray2rgb(rgb2gray(img)), gamma=0.75) for img in images]

		return images, True
	except:
		print(
			"ERROR: frame number {} does not exist in {} {}".format(
				frame, enc_id, start_date.strftime("%Y-%m-%d")
			)
		)
		return [], False


def fetch_interval_images(framelist, video_files, enc_id, start_date):
	"""
	for each considered frame in the framelist, the frame is grabbed from each video in the video_files and merged
	returns a list of merged images
	"""
	interval_images = []
	for frame in framelist:
		images, success = fetch_images_for_frame(frame, video_files, enc_id, start_date)
		if not success:
			continue
		interval_images.append(merge_images(images, enc_id))

	return interval_images


def save_image_to_file_unit8(img, path):
	# Use this function to save uint8 numpy arrays.

	ensure_directory(path)
	# can't use cv2 here because it doesn't support unicode characters like ü and ß. imageio is still faster than skimage.
	imageio.imwrite(path, img)


def save_image_to_file_float64(img, path):
	# Use this function to save float64 numpy arrays.

	ensure_directory(path)
	imsave(path, (255 * img).astype(np.uint8))


def IO_write_only_video_frames(args):
	"""
	Parameters
	----------
	args : List of lists:
		required frames: list of integers
		start_date: datetime
		save_path: path where to save images of this video file
		enclosure_id: enclosure id
		video_list: list of video paths to be used / concatenated
		subfolder: images/tmp to savepath
	Returns
	-------
	Nothing, but writes jpg images to save_path.
	Uses the black polygons etc.
	"""

	try:
		framelist, start_date, save_path, enc_id, vid_list, subfolder = args
	except:
		print("ERROR: Invalid arguments passed.")
		print(args)
		return None

	try:
		video_files = [VideoFileClip(x) for x in vid_list]
	except:
		print("ERROR: video files do not exist ({})".format(enc_id))
		return None

	for frame in framelist:
		images, success = fetch_images_for_frame(frame, video_files, enc_id, start_date)
		if not success:
			continue
		res = merge_images(images, enc_id)
		curr_time = (start_date + timedelta(seconds=frame)).strftime("%Y%m%d-%H%M%S")
		img_path = save_path + f"{subfolder}/" + enc_id + "_" + curr_time + ".jpg"

		save_image_to_file_float64(res, img_path)


def IO_video_to_img_prediction(args):
	"""
	Parameters
	----------
	args : List of lists:
		required timeintervals: list of integers
		start_date: datetime
		save_path: path where to save images of this video file
		enclosure_id: enclosure id
		video_list: list of video paths to be used / concatenated
	Returns
	-------
	Nothing, but writes jpg images to save_path.
	Uses the black polygons etc.

	For every time interval, the difference image is calculated as
	given in get_config
	"""
	try:
		intervallist, start_date, save_path, enc_id, vid_list = args
	except:
		print("ERROR: unknown error in IO_write_video_frames")
		return None

	# video_files = [VideoFileClip(x) for x in vid_list]

	try:
		video_files = [VideoFileClip(x) for x in vid_list]
	except:
		print("ERROR: video files do not exist ({})".format(enc_id))
		return None

	for interval in intervallist:
		framelist = use_frames_per_interval(interval)

		# returns list of all merged, gray, gamma-corrected images for this interval
		interval_images = fetch_interval_images(
			framelist, video_files, enc_id, start_date
		)

		img, diff = get_difference_images(interval_images)
		curr_time = (start_date + timedelta(seconds=int(framelist[0]))).strftime(
			"%Y%m%d-%H%M%S"
		)

		save_image_to_file_float64(img, save_path + f"images/{enc_id}_{curr_time}.jpg")
		save_image_to_file_float64(diff, save_path + f"differences/{enc_id}_{curr_time}.jpg")


def IO_video_timeinterval(args):
	"""
	Parameters
	----------
	args : List of lists:
		required timeintervals: list of integers
		start_date: datetime
		save_path: path where to save images of this video file
		action_label: list of same length as required timeintervals
		enclosure_id: enclosure id
		video_list: list of video paths to be used / concatenated
	Returns
	-------
	Nothing, but writes jpg images to save_path.
	Uses the black polygons etc.

	For every time interval, the difference image is calculated as
	given in get_config
	"""

	try:
		intervallist, start_date, save_path, action_label, enc_id, vid_list = args
	except:
		print("ERROR: invalid arguments passed to IO_video_timeinterval")
		return None
	try:
		video_files = [VideoFileClip(x) for x in vid_list]
	except:
		print("ERROR: video files do not exist ({})".format(enc_id))
		return None

	for interval in intervallist:
		framelist = use_frames_per_interval(interval)
		interval_images = fetch_interval_images(
			framelist, video_files, enc_id, start_date
		)

		img, diff = get_difference_images(interval_images)
		curr_time = (start_date + timedelta(seconds=int(framelist[0]))).strftime(
			"%Y%m%d-%H%M%S"
		)

		save_image_to_file_float64(
			img, save_path + f"{action_label}/{enc_id}/images/{enc_id}_{curr_time}.jpg"
		)
		save_image_to_file_float64(
			diff,
			save_path + f"{action_label}/{enc_id}/differences/{enc_id}_{curr_time}.jpg",
		)


def glue_image_difference(img_path, diff_path, bbs, polygons):
	"""
	Takes the image, the difference image and the bounding boxes as an input.
		bbs = [ [clsname, bb, certainty] ], polygons = None or [ [coordinates] ]
	For every bounding box label, it returns the cut out area, glues it together with the same area of the difference image
	If polygons are given, then the polygons are taken instead of the bounding boxes
	"""

	def gluing_operation(img, diff_img):
		"""
		Input are two images (rectangular, black background) of variable size
		Returns joint image of measurements IMAGE_HEIGHT_ACTION_CLASSIFICATION, IMAGE_WIDTH_ACTION_CLASSIFICATION
		with the input images centralised on each half

		Performance notice: Tobi has tried replacing the resize calls with opencv resize calls and found no cpu time
		improvement at all.
		"""

		def _rescale_image(image, h, w):
			img_h, img_w, _ = image.shape
			ret = np.zeros((h, w, 3))

			if img_h / h > img_w / w:
				img_r = resize(image, (h, int(img_w * h / img_h)), anti_aliasing=True)
				missing_width = w - int(img_w * h / img_h)
				if missing_width > 0:
					ret[
						:,
						int(missing_width / 2) : int(missing_width / 2)
						+ img_r.shape[1],
						:,
					] = img_r
			else:
				img_r = resize(image, (int(img_h * w / img_w), w), anti_aliasing=True)
				missing_height = h - int(img_h * w / img_w)
				if missing_height > 0:
					ret[
						int(missing_height / 2) : int(missing_height / 2)
						+ img_r.shape[0],
						:,
						:,
					] = img_r

			return ret

		left = _rescale_image(
			img,
			int(IMAGE_HEIGHT_ACTION_CLASSIFICATION),
			int(IMAGE_WIDTH_ACTION_CLASSIFICATION / 2),
		)
		right = _rescale_image(
			diff_img,
			int(IMAGE_HEIGHT_ACTION_CLASSIFICATION),
			int(IMAGE_WIDTH_ACTION_CLASSIFICATION / 2),
		)

		ret = (np.concatenate((left, right), axis=1) * 254).astype(
			np.uint8
		)  # resize of skimage uses [0,1] range
		return ret

	img = imread(img_path)
	diff = imread(diff_path)

	ret = {}
	bb_coordinates = {}
	polygon_coordinates = {}

	for j in range(len(bbs)):
		classname, bb, certainty = bbs[j]

		if polygons == None:
			x_center, y_center, w, h = bb
			img_part = img[
				y_center - h // 2 : y_center + h // 2,
				x_center - w // 2 : x_center + w // 2,
				:,
			]
			diff_part = diff[
				y_center - h // 2 : y_center + h // 2,
				x_center - w // 2 : x_center + w // 2,
				:,
			]

			# Commented out to reduce console output clutter
			# print(classname)
			ret[classname] = gluing_operation(img_part, diff_part)
			bb_coordinates[classname] = bb
			polygon_coordinates[classname] = []

		else:
			poly = np.array(polygons[j])
			rr, cc = polygon(poly[:, 1], poly[:, 0], img.shape)

			# create black rectangle with bounding box dimensions
			background_img = np.zeros((bbs[j][3], bbs[j][2], img.shape[2])).astype(
				np.uint8
			)
			background_diff = np.zeros((bbs[j][3], bbs[j][2], img.shape[2])).astype(
				np.uint8
			)

			# input the segment
			background_img[rr, cc, :] = img[rr, cc, :]
			background_diff[rr, cc, :] = diff[rr, cc, :]

			# rescale images and put them together
			ret[classname] = gluing_operation(background_img, background_diff)
			bb_coordinates[classname] = bb
			polygon_coordinates[classname] = poly

	return ret, bb_coordinates, polygon_coordinates


def augment_action_classifier_image(img):
	"""
	Gets an imagepath as an input and return an augmented variant (as a numpy array)
	"""
	from skimage.util import random_noise
	from skimage import exposure

	try:
		im = img_as_float(imread(img))
	except:
		print("ERROR. Imagefile not found.", img)
		return None

	x = np.random.random()
	if x <= 0.3:
		im = random_noise(im)
	if 0.2 <= x <= 0.4:
		if x <= 0.3:
			sign = -1
		else:
			sign = 1
		im = exposure.adjust_gamma(
			im, gamma=1 + sign * 0.5 * np.random.random(), gain=0.9
		)
	if 0.3 <= x <= 0.7:
		im = im[:, ::-1]
	if 0.6 <= x <= 0.9:
		v_min, v_max = np.percentile(im, (0.2, 99.8))
		im = exposure.rescale_intensity(im, in_range=(v_min, v_max))

	return im


def get_difference_images(frames):
	"""
	Given a list of frames (this means np.arrays), calculates the difference image.
	Returns the image in the middle of the list and the difference image. New: Checks for datatype of image
	and chooses optimized implementation or old one based on that. Return type is the same as input type.
	"""
	dtype = frames[0].dtype
	# BOVIDS-Live case
	if dtype == np.uint8:
		frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
		average_image = np.mean(frames_np, axis=0)
		designated_frame = frames_np[len(frames) // 2]
		differences = np.abs(average_image - designated_frame)
		differences = np.minimum(4 * differences, 1.0)
		designated_frame_uint8 = (designated_frame * 255).astype(np.uint8)
		differences_uint8 = (differences * 255).astype(np.uint8)
		return designated_frame_uint8, differences_uint8

	# BOVIDS2 case or Compatibility mode
	elif dtype == np.float64:
		average_image = np.zeros(frames[0].shape)
		for f in frames:
			average_image += f

		# print("average_image:", average_image)
		average_image /= len(frames)

		designated_frame = len(frames) // 2
		differences = np.abs(average_image - frames[designated_frame])
		differences = np.minimum(
			4 * differences, np.ones(differences.shape, dtype=np.float32)
		)
		return frames[designated_frame], differences

	else:
		print(f"ERROR: Unsupported dtype {dtype}. Expected uint8 or float64.")
		return None


def merge_images_optimized(l_in, enclosure_id=""):
	dtype = l_in[0].dtype

	if dtype not in (np.uint8, np.float64):
		print(f"ERROR: Unsupported dtype {dtype}. Expected uint8 or float64.")
		return None

	if len(l_in) > 9:
		print("IMPLEMENTATION ERROR: not more than 9 streams allowed.")
		return None

	l = list(l_in)

	if len(l) == 1:
		ret = l[0]

	if len(l) >= 2:
		res_l = [im.shape for im in l]
		aspect_ratio_169 = [np.abs(x[0] / x[1] - 4 / 3) > 0.2 for x in res_l]

		for j, ar in enumerate(aspect_ratio_169):
			if not ar:
				l[j] = _map_43_into_169(l[j])

		min_h = min(im.shape[0] for im in l)
		min_w = min(im.shape[1] for im in l)

		l = [cv2.resize(im, (min_w, min_h), interpolation=cv2.INTER_AREA) for im in l]

		if len(l) == 2:
			ret = np.concatenate(l, axis=0)

		elif len(l) == 3:
			b = np.zeros((min_h, min_w, 3), dtype=dtype)
			ret = np.concatenate([
				np.concatenate([l[0], l[1]], axis=0),
				np.concatenate([l[2], b], axis=0)
			], axis=1)

		elif len(l) == 4:
			ret = np.concatenate([
				np.concatenate([l[0], l[1]], axis=0),
				np.concatenate([l[2], l[3]], axis=0)
			], axis=1)

		elif len(l) == 5:
			b = np.zeros((min_h, min_w, 3), dtype=dtype)
			ret = np.concatenate([
				np.concatenate([l[0], l[1], l[2]], axis=0),
				np.concatenate([l[3], l[4], b], axis=0)
			], axis=1)

		elif len(l) == 6:
			ret = np.concatenate([
				np.concatenate([l[0], l[1], l[2]], axis=0),
				np.concatenate([l[3], l[4], l[5]], axis=0)
			], axis=1)

		elif len(l) == 7:
			b = np.zeros((min_h, min_w, 3), dtype=dtype)
			ret = np.concatenate([
				np.concatenate([l[0], l[1], l[2]], axis=0),
				np.concatenate([l[3], l[4], l[5]], axis=0),
				np.concatenate([l[6], b, b], axis=0)
			], axis=1)

		elif len(l) == 8:
			b = np.zeros((min_h, min_w, 3), dtype=dtype)
			ret = np.concatenate([
				np.concatenate([l[0], l[1], l[2]], axis=0),
				np.concatenate([l[3], l[4], l[5]], axis=0),
				np.concatenate([l[6], l[7], b], axis=0)
			], axis=1)

		elif len(l) == 9:
			ret = np.concatenate([
				np.concatenate([l[0], l[1], l[2]], axis=0),
				np.concatenate([l[3], l[4], l[5]], axis=0),
				np.concatenate([l[6], l[7], l[8]], axis=0)
			], axis=1)

	h, w, _ = ret.shape

	if h >= w:
		new_width = int(640 / h * w)
		new_height = 640
	else:
		new_height = int(640 / w * h)
		new_width = 640

	tmp = cv2.resize(ret, (new_width, new_height), interpolation=cv2.INTER_AREA)

	black_embedding = np.zeros((640, 640, 3), dtype=dtype)
	y_off = (640 - new_height) // 2
	x_off = (640 - new_width) // 2
	black_embedding[y_off:y_off + tmp.shape[0], x_off:x_off + tmp.shape[1]] = tmp
	ret = black_embedding

	if enclosure_id not in cached_polygons:
		cached_polygons[enclosure_id] = get_black_polygons(enclosure_id)

	for poly in cached_polygons[enclosure_id]:
		rr, cc = polygon(poly[:, 1], poly[:, 0], ret.shape)
		ret[rr, cc, :] = 0

	return ret


# Remove this together with compatibility mode on BOVIDS-Live.
def merge_images(l_in, enclosure_id=""):
	"""
	DEPRECATED! Use optimized version (25 times faster) to train new models and do predictions with them.
	Use this function only for compatibility. The difference in resizing functions yields slightly different outputs,
	leading to expected prediction quality decreases when the model was not trained using the same merge_images function
	version as is used for prediction. It should be noted that the difference in resizing artifacting actually makes the
	optimized version look sharper, losing fewer details, at least to the human eye.

	Parameters
	----------
	l_in : list of images
		Merges images as 2x1, 3x1, 2x2, 3x2 (blacked)

	Returns
	-------
	one joined image with resolution depending on the input
	the maximum width/height is kept - Tobias asks: are you sure it's not the minimum instead?
	"""

	l = l_in.copy()

	if len(l) > 9:
		print("IMPLEMENTATION ERROR: not more than 9 streams allowed.")
		return None

	if len(l) == 1:
		ret = l[0]

	if len(l) >= 2:
		res_l = [im.shape for im in l]
		aspect_ratio_169 = [np.abs(x[0] / x[1] - 4 / 3) > 0.2 for x in res_l]

		# convert to 16:9 format
		j = 0
		for ar in aspect_ratio_169:
			if not ar:
				l[j] = _map_43_into_169(l[j])
			j += 1

		# resize all images to minimum resolution
		res = (
			min([res_l[j][0] for j in range(len(res_l))]),
			min([res_l[j][1] for j in range(len(res_l))]),
		)
		l = [resize(im, (res[0], res[1])) for im in l]

		# concatenate images in grid
		if len(l) == 2:
			ret = np.concatenate(l, axis=0)
		if len(l) == 3:
			b = np.zeros((res[0], res[1], 3), np.uint8)
			ret1 = np.concatenate([l[0], l[1]], axis=0)
			ret2 = np.concatenate([l[2], b], axis=0)
			ret = np.concatenate([ret1, ret2], axis=1)
		elif len(l) == 4:
			ret1 = np.concatenate([l[0], l[1]], axis=0)
			ret2 = np.concatenate([l[2], l[3]], axis=0)
			ret = np.concatenate([ret1, ret2], axis=1)
		elif len(l) == 5:
			b = np.zeros((res[0], res[1], 3), np.uint8)
			ret1 = np.concatenate([l[0], l[1], l[2]], axis=0)
			ret2 = np.concatenate([l[3], l[4], b], axis=0)
			ret = np.concatenate([ret1, ret2], axis=1)
		elif len(l) == 6:
			ret1 = np.concatenate([l[0], l[1], l[2]], axis=0)
			ret2 = np.concatenate([l[3], l[4], l[5]], axis=0)
			ret = np.concatenate([ret1, ret2], axis=1)
		elif len(l) == 7:
			b = np.zeros((res[0], res[1], 3), np.uint8)
			ret1 = np.concatenate([l[0], l[1], l[2]], axis=0)
			ret2 = np.concatenate([l[3], l[4], l[5]], axis=0)
			ret3 = np.concatenate([l[6], b, b], axis=0)
			ret = np.concatenate([ret1, ret2, ret3], axis=1)
		elif len(l) == 8:
			b = np.zeros((res[0], res[1], 3), np.uint8)
			ret1 = np.concatenate([l[0], l[1], l[2]], axis=0)
			ret2 = np.concatenate([l[3], l[4], l[5]], axis=0)
			ret3 = np.concatenate([l[6], l[7], b], axis=0)
			ret = np.concatenate([ret1, ret2, ret3], axis=1)
		elif len(l) == 9:
			ret1 = np.concatenate([l[0], l[1], l[2]], axis=0)
			ret2 = np.concatenate([l[3], l[4], l[5]], axis=0)
			ret3 = np.concatenate([l[6], l[8], l[9]], axis=0)
			ret = np.concatenate([ret1, ret2, ret3], axis=1)

	# rescale combined images to fit into 640x640 px as this is used for object detection anyways and fill the rest with black
	h, w, c = ret.shape
	if h >= w:
		# height is larger
		new_width = int(640 / h * w)
		new_height = 640
	else:
		# width is larger
		new_height = int(640 / w * h)
		new_width = 640

	tmp = resize(ret, (new_height, new_width))
	missing_height = int(640 - new_height)
	missing_width = int(640 - new_width)
	black_embedding = np.zeros((640, 640, 3))
	black_embedding[
		missing_height // 2 : missing_height // 2 + tmp.shape[0],
		missing_width // 2 : missing_width // 2 + tmp.shape[1],
	] = tmp
	ret = black_embedding

	# apply black regions to remove overlap of cameras' FOVs in enclosure, cache per enclosure to reduce IO operations
	if not enclosure_id in cached_polygons.keys():
		cached_polygons[enclosure_id] = get_black_polygons(enclosure_id)

	black_regions = cached_polygons[enclosure_id]
	for poly in black_regions:
		rr, cc = polygon(poly[:, 1], poly[:, 0], ret.shape)
		ret[rr, cc, :] = (0, 0, 0)

	return ret


def _map_43_into_169(im):
	"""
	Maps a 4:3 image (640x480px) into a 1280x720 format by adding black borders
	"""

	res = np.zeros((720, 1280, 3), np.uint8)
	res[0 : im.shape[0], 0 : im.shape[1]] = im
	return res
