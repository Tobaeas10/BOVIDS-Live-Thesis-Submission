__author__ = ["Tobias Weiß"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"


try:
	# Enable this to fix import errors when trying to run from console without passing environment variables.
	if __name__ == "__main__":
		import sys, os
		sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

	# Standard library
	import datetime
	import time
	from multiprocessing import Array, Event, Value
	from concurrent.futures import ProcessPoolExecutor
	import os
	from threading import Thread, Lock
	import logging
	from signal import SIGINT, signal
	import av

	# Third-Party packages - Use in commercial products allowed, tho
	import numpy
	import pandas as pd
	import cv2
	from ataraxis_time import PrecisionTimer
	from skimage.color import gray2rgb, rgb2gray
	from skimage.exposure import adjust_gamma

	# Local project packages
	from server.bovids_v2.lib.statistics import create_key_values_csv, create_phase_csv, create_cycle_csv, get_cycles_behavior
	from server.bovids_v2.lib.action_classification import predict_folder_ac
	from server.bovids_v2.lib.availability_checks import get_ac_networkpath
	from server.bovids_v2.lib.func import ensure_directory, get_images_ac_mode
	from server.bovids_v2.lib.image_manipulation import merge_images_optimized, \
		save_image_to_file_unit8, merge_images, get_difference_images, save_image_to_file_float64
	from server.bovids_v2.lib.object_detection import predict_folder_differences
	from server.bovids_v2.config.get_config import get_enclosure_information, get_individual_information, get_postprocessing_rules, BEHAVIOR_STATISTICS
	from server.bovids_v2.lib.pipeline_functions import get_statistics_paths, get_postprocessing_subaction_paths, get_postprocessing_stly_paths, \
		get_subactions_visualization_info, get_stly_visualization_info
	from server.bovids_v2.lib.post_processor import PostProcessor, PostProcessorSubactions
	from local_pc.run_prediction import TEMPORAL_LOCAL_STORAGE, DEVICE, BOVIDS_AC, BOVIDS_STATS, BOVIDS_VISUAL
	from server.bovids_v2.lib.visualizer import NightVisualizer
	from status_monitor_for_BOVIDS_Live import StatusMonitor
	from local_pc import thesis_tests


except Exception as e:
	import traceback
	traceback.print_exc()
	input("\nPress Enter to continue...")


"""
TL;DR: This is a (close to) real-time, RTSP-Stream-based prediction pipeline script.
It can also be used to collect images for dataset building.
It mimics bovids_prediction.predict_excel_file() but for streams instead of videos,
keeping equivalence under normal conditions. When using old models for prediction,
use compatibility mode for equivalence.
It is recommended to keep total CPU usage below 90% most of the time. Otherwise,
timing errors will be so bad that hey might worsen prediction performance.

If there are any questions, email tobias.weiss16@gmail.com
"""


# FOLDER PATHS - Either choose a relative or absolute path to the stream prediction file.
STREAM_PREDICTION_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/predict_stream_example.xlsx" # which enclosure(s) to monitor
# STREAM_PREDICTION_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/minimal.xlsx"
# STREAM_PREDICTION_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/extended.xlsx"
# STREAM_PREDICTION_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/predict_stream_10C_2E-5_4.xlsx"
# STREAM_PREDICTION_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/predict_stream_10C_4E-1_2_3_4.xlsx"
# STREAM_PREDICTION_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/predict_stream_10C_4E-2_4_6_8.xlsx"
# STREAM_PREDICTION_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/predict_stream_30C_12E-3_6_6_6_9.xlsx"
STREAM_CONFIG_XLSX = os.path.dirname(os.path.abspath(__file__)) + "/stream_config.xlsx" # configure stream settings as below.


# Default settings. WILL BE OVERWRITTEN! Only change these if you know what you're doing. Otherwise please just change the values in STREAM_CONFIG_XLSX
CAMERAS_FPS = 1 # Frames per second (framerate/FPS) delivered by the RTSP camera streams per second. It is assumed that all connected cameras share the same value. If not, choose highest FPS of all streams.
SECONDS_PER_INTERVAL = 7  # How long each interval lasts (in seconds).
IMAGES_PER_INTERVAL = 3  # How many images to capture per interval (must be at least 2 for diffs to work).
IMAGE_SPACING_WITHIN_INTERVAL = CAMERAS_FPS # How many frames apart the images within the same interval shall be chosen (for moving detection). When the setting_value is 1 then frames 1, 2 and 3 are chosen. If it's 2 --> frames 1, 3 and 5 and so on.
SHOW_WINDOWS = False  # If true, windows showing the camera streams will be opened. - Default: False
SHOW_RESOURCE_MONITOR = True  # If true, a window showing the CPU and RAM usage as well as number of exhausted batch workers and number of queued batches. - Default: True
INTERVALS_PER_BATCH = 50  # How many intervals to collect in a batch before starting OD&AC processing on it. If chosen very high, final results will be available later. If chosen too low, technical overhead will slow BOVIDS down. Recommended range: Between 10 and 100
MAX_NUM_PARALLEL_BATCH_PROCESSES = 1  # Max number of workers to spawn in parallel for OD&AC processing. If chosen too low, the system might start lagging behind and final results will be available later. Always keep between 1 and {number of (physical) CPU cores} - 1. For 8GB RAM (Memory): up to 3. For 16GB RAM: up to 12. For 32GB RAM: up to 30.
MAX_NUM_PARALLEL_MERGE_PROCESSES = 1  # Same as MAX_NUM_PARALLEL_BATCH_PROCESSES but for pre-processing. Recommended to keep at 1 unless Compatibility mode is used and there are lots of streams or enclosures.
COMPATIBILITY_MODE = False  # If enabled, changes pipeline to use older, 25x slower pre-processing functions. Only use this if you want to use models that were trained on the old set of pre-processing functions (not [...]_optimzed).

# Try to read settings configuration from stream_config.xlsx
try:
	stream_config_df = pd.read_excel(STREAM_CONFIG_XLSX, index_col="setting_name")
	CAMERAS_FPS = int(stream_config_df.loc["CAMERAS_FPS", "setting_value"])
	SECONDS_PER_INTERVAL = int(stream_config_df.loc["SECONDS_PER_INTERVAL", "setting_value"])
	IMAGES_PER_INTERVAL = int(stream_config_df.loc["IMAGES_PER_INTERVAL", "setting_value"])
	IMAGE_SPACING_WITHIN_INTERVAL = int(stream_config_df.loc["IMAGE_SPACING_WITHIN_INTERVAL", "setting_value"])
	SHOW_WINDOWS = bool(stream_config_df.loc["SHOW_WINDOWS", "setting_value"])
	SHOW_RESOURCE_MONITOR = bool(stream_config_df.loc["SHOW_RESOURCE_MONITOR", "setting_value"])
	INTERVALS_PER_BATCH = int(stream_config_df.loc["INTERVALS_PER_BATCH", "setting_value"])
	MAX_NUM_PARALLEL_BATCH_PROCESSES = int(stream_config_df.loc["MAX_NUM_PARALLEL_BATCH_PROCESSES", "setting_value"])
	MAX_NUM_PARALLEL_MERGE_PROCESSES = int(stream_config_df.loc["MAX_NUM_PARALLEL_MERGE_PROCESSES", "setting_value"])
	COMPATIBILITY_MODE = bool(stream_config_df.loc["COMPATIBILITY_MODE", "setting_value"])
except Exception as e:
	logging.error(f"While trying to read the stream config from {STREAM_CONFIG_XLSX} an error occurred: {e}")
	logging.warning("Using default settings instead.")

# Additional Debugging Settings - not user-exposed.
STREAM_CONNECT_TIMEOUT = 5  # Not only timeout (in seconds) for stream connection establishment, but also seconds to wait for a reconnect attempt before retrying again.
DETAILLED_TIMING_DEBUGGING_MESSAGES = False  # Prints details and stats for slow ticks, Metric 1 (inter-stream sampling time error) and Metric 2 (per-stream sampling spacing time error).
WARMUP_TICKS = 1000  # number of ticks excluded from all metric recording at startup. Only has effect if DETAILLED_TIMING_DEBUGGING_MESSAGES = True
BENCHMARK_SHUTDOWN_TICK = -1  # post-warmup ticks before auto-shutdown (only used for shutdown testing). Set to -1 to disable premature shutdown.

if DETAILLED_TIMING_DEBUGGING_MESSAGES:
	import matplotlib
	matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot is ever imported
	from matplotlib import pyplot as plt

# Automatically configured constants. Do not touch!
FRAMES_PER_INTERVAL = SECONDS_PER_INTERVAL * CAMERAS_FPS
FRAMES_PER_BATCH = FRAMES_PER_INTERVAL * INTERVALS_PER_BATCH
REQUIRED_FRAME_INDICES_IN_INTERVAL = []
for i in range(0, FRAMES_PER_INTERVAL, IMAGE_SPACING_WITHIN_INTERVAL):
	REQUIRED_FRAME_INDICES_IN_INTERVAL.append(i)
	if len(REQUIRED_FRAME_INDICES_IN_INTERVAL) == IMAGES_PER_INTERVAL:
		break
SMALLEST_SAMPLE_SPACING = 69420
BIGGEST_SAMPLE_SPACING = 0
for i in range(len(REQUIRED_FRAME_INDICES_IN_INTERVAL)):
	# If last index, loop around to first index (should always be 0)
	if i + 1 == len(REQUIRED_FRAME_INDICES_IN_INTERVAL):
		spacing = FRAMES_PER_INTERVAL - REQUIRED_FRAME_INDICES_IN_INTERVAL[i]
	else:
		spacing = REQUIRED_FRAME_INDICES_IN_INTERVAL[i+1] - REQUIRED_FRAME_INDICES_IN_INTERVAL[i]
	SMALLEST_SAMPLE_SPACING = min(SMALLEST_SAMPLE_SPACING, spacing)
	BIGGEST_SAMPLE_SPACING = max(BIGGEST_SAMPLE_SPACING, spacing)

NO_FRESH_FRAME_TIMEOUT = SECONDS_PER_INTERVAL + 1
START_DATE_Y_M_D = datetime.date.today().strftime("%Y-%m-%d")
START_DATE_YMD = START_DATE_Y_M_D.replace("-", "")


# Global variables. Mostly initialized in launch_stream_capture_service. Do not touch!
streams_ready = None
streams_active = None
num_cameras = None
shutdown_flag = Event() if __name__ == "__main__" else None
timer = PrecisionTimer('us')
_shared_noo_batches_ready_to_process = None # Only gets initialized on batch worker processes
any_valid_shape = None
decode_yield_times = []


# Set up the console logging format: timestamp + threadname + message
if not logging.getLogger().hasHandlers():
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s",
		datefmt="%H:%M:%S"
	)
else:
	# Logging is already configured. Using existing logger.
	logging.getLogger().setLevel(logging.INFO)
# Disable annoying log messages from plotly
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)


class SingleFrameBuffer:
	"""
	Thread-safe single-slot frame buffer. The capture thread pushes raw av.VideoFrames
	cheaply via put(); the main loop calls get() to retrieve the latest frame as a
	numpy array. Conversion to numpy array happens in get(), under the lock, only when needed.
	"""

	def __init__(self):
		self._lock = Lock()
		self._av_frame = None
		self._timestamp = _get_time_s()
		self._last_valid_shape = None
		self.abs_delays = []
		self.counter = 0


	# ── called by the main loop ───────────────────────────────────────────────

	def get(self, tick_delay):
		"""
		Returns (frame, is_real, age_seconds) for the latest stored frame.
		Falls back to a black frame if no frame has been received or the last one
		is older than SECONDS_PER_INTERVAL. is_real=False signals a fallback frame.
		tick_delay is only relevant for debugging. Just pass a 0 if you don't care.
		"""
		global any_valid_shape

		with self._lock:
			age = _get_time_s() - self._timestamp
			if self._av_frame is not None and age < SECONDS_PER_INTERVAL:
				np_frame = self._av_frame.to_ndarray(format="rgb24") if COMPATIBILITY_MODE else self._av_frame.to_ndarray(format="gray")
				any_valid_shape = self._last_valid_shape = np_frame.shape
				if tick_delay is not None: self.abs_delays.append(tick_delay - age)
				return np_frame, True, age
			elif any_valid_shape is not None:
				logging.error("Inserting black frame to keep system running.")
				return numpy.zeros(self._last_valid_shape if self._last_valid_shape is not None else any_valid_shape, dtype=numpy.uint8), False, age
			else:
				return None, False, 0.0

	def has_ever_received(self):
		with self._lock:
			return self._av_frame is not None or self._last_valid_shape is not None

	# ── called by the stream thread ───────────────────────────────────────────
	def put(self, av_frame, counter):
		"""Stores the latest av.VideoFrame, overwriting whatever was there. Always keeps the freshest frame."""
		with self._lock:
			self._av_frame = av_frame
			self.counter = counter
			self._timestamp = _get_time_s()
			return self._timestamp


def open_stream(enclosure, camera_index, rtsp_url, thread_index):
	"""
	Capture thread for a single RTSP camera. Continuously decodes frames and pushes
	them into the enclosure's SingleFrameBuffer. Invalid frames are silently dropped;
	the main loop handles stale-frame fallback via SingleFrameBuffer.get().

	Reconnects automatically on stream failure, looping indefinitely until
	shutdown_flag is set.
	"""
	enclosure_id = enclosure["enclosure_id"]
	stream_name = f"Camera {camera_index + 1} of {enclosure_id} on {rtsp_url}"
	buf = enclosure["stream_buffers"][camera_index]

	def _try_open():
		"""
		Opens the RTSP container with a timeout and flushes the initial jitter buffer.
		Returns a ready (container, video_stream) pair, or (None, None) on failure or shutdown.
		"""
		while not shutdown_flag.is_set():
			logging.info(f"Opening {stream_name}")
			try:
				container = av.open(rtsp_url, timeout=(STREAM_CONNECT_TIMEOUT, NO_FRESH_FRAME_TIMEOUT))
				break
			except Exception as e:
				logging.error(f"Connection attempt for {stream_name} failed due to {e}. Trying again in {STREAM_CONNECT_TIMEOUT} seconds.")
				time.sleep(STREAM_CONNECT_TIMEOUT)

		try:
			video_stream = container.streams.video[0]
			# print(video_stream.codec, video_stream.codec_tag, video_stream.codec_context)
		except Exception as exc:
			logging.warning(f"⚠️  {stream_name}: no video stream found – {exc}")
			_cleanup_capture_device(container, stream_name)
			return None, None

		return container

	# ── first open ────────────────────────────────────────────────────────────

	container = _try_open()
	if container is None:  # shutdown was set during the retry loop
		streams_ready[thread_index] = True
		streams_active[thread_index] = False
		return

	streams_ready[thread_index] = True
	streams_active[thread_index] = True

	logging.info(f"{stream_name} starting continuous capture…")

	# ── main decode loop ──────────────────────────────────────────────────────
	counter = 0
	while not shutdown_flag.is_set():
		try:
			t0 = _get_time_s()
			for av_frame in container.decode(video=0):
				tmp = _get_time_s()
				if counter > WARMUP_TICKS:
					decode_yield_times.append(tmp - t0)
					t0 = tmp
				if shutdown_flag.is_set():
					break

				buf.put(av_frame, counter)

				# Optional cv2 preview window
				# We only pay the to_ndarray cost when we already stored (or always_sample).
				if SHOW_WINDOWS:
					with buf._lock:
						frame_to_show = buf._frame  # already converted, no re-copy needed
					if frame_to_show is not None:
						try:
							cv2.imshow(stream_name, frame_to_show)
							if cv2.waitKey(1) & 0xFF == ord("q"):
								shutdown_flag.set()
								break
						except cv2.error:
							logging.warning(f"{stream_name} display error.")
				t0 = _get_time_s()
				counter += 1


		except Exception as exc:
			_cleanup_capture_device(container, window_name=stream_name)
			logging.warning(
				f"⚠️  {stream_name}: stream error – {exc}. "
				f"Closing and attempting reconnect…"
			)

		finally:
			streams_active[thread_index] = False
			if container is not None:
				try:
					_cleanup_capture_device(container, window_name=stream_name)
				except Exception:
					pass
				container = None

		if shutdown_flag.is_set():
			break

		# ── reconnect loop ────────────────────────────────────────────────────
		while not shutdown_flag.is_set():
			container = _try_open()
			if container is not None:
				logging.info(f"✅ {stream_name}: reconnect successful. Resuming capture.")
				streams_active[thread_index] = True
				break
			logging.warning(f"❌ {stream_name}: reconnect failed.")

	_cleanup_capture_device(container, window_name=stream_name)


def run_stream_capture_service(enclosures):
	"""
	Main orchestration loop for the live pipeline.

	Spawns one capture thread per camera (via open_stream), then ticks at 1/CAMERAS_FPS.
	Each sampling tick calls all SingleFrameBuffer.get(), accumulates IMAGES_PER_INTERVAL frames per
	enclosure, computes merged + difference images (in parallel), and dispatches OD+AC batches to a
	worker process pool whenever a full batch is ready.

	After shutdown, submits any partial batch, waits for all OD+AC workers to finish,
	then runs post-processing, statistics, and visualizations.

	Shutdown triggers (any one is sufficient):
	  1. Ctrl-C
	  2. Observation-period end reached for all enclosures
	  3. No camera delivers a fresh frame for NO_FRESH_FRAME_TIMEOUT seconds
	"""

	signal(SIGINT, _signal_handler)
	logging.info("Starting stream capture service. Press Ctrl-C to terminate.")
	_set_process_priority_high()

	# ── Build per-enclosure runtime state ────────────────────────────────────
	for enclosure in enclosures:
		enclosure["stream_buffers"] = [SingleFrameBuffer() for _ in range(len(enclosure["rtsp_urls"]))]
		enclosure["frames_to_merge"] = []        # accumulates within one interval
		enclosure["num_cameras"] = len(enclosure["rtsp_urls"])
		enclosure["interval_index"] = 0        # how many intervals captured so far

	# Enable status monitoring to notify users about OD&AC falling behind, RAM usage and worker utilization.
	monitor = StatusMonitor(max_workers=MAX_NUM_PARALLEL_BATCH_PROCESSES)
	if SHOW_RESOURCE_MONITOR:
		monitor.start()
	# ── Launch one thread per camera ─────────────────────────────────────────
	threads = []
	thread_index = 0
	for enclosure in enclosures:
		for camera_index, rtsp_url in enumerate(enclosure["rtsp_urls"]):
			t = Thread(
				target=open_stream,
				args=(enclosure, camera_index, rtsp_url, thread_index),
				name=f"Enc{enclosure['enclosure_id']}_Cam{camera_index}",
				daemon=False,
			)
			threads.append(t)
			thread_index += 1

	for t in threads:
		t.start()

	# ── OD+AC batch pool ─────────────────────────────────────────────────────
	num_submitted_batches = 0
	num_completed_batches = 0

	# Track running futures so the monitor can show "busy" count accurately
	def _batch_done(_):
		nonlocal num_completed_batches
		num_completed_batches += 1
		monitor.update(completed=num_completed_batches)

	noo_batches_ready_to_process = Value('i', 0)
	batch_pool = ProcessPoolExecutor(max_workers=MAX_NUM_PARALLEL_BATCH_PROCESSES, initializer=_init_batch_processes, initargs=(noo_batches_ready_to_process,))
	def _submit_batch(args):
		nonlocal num_submitted_batches
		future = batch_pool.submit(_do_OD_and_AC_on_batch, args)
		future.add_done_callback(_batch_done)
		num_submitted_batches += 1
		monitor.update(submitted=num_submitted_batches)

	merge_pool = ProcessPoolExecutor(max_workers=MAX_NUM_PARALLEL_MERGE_PROCESSES)
	def _last_merge_of_batch_done_callback(_):
		with noo_batches_ready_to_process.get_lock():
			noo_batches_ready_to_process.value += 1

	# ── Main sampling loop ───────────────────────────────────────────────────
	tick_interval = 1.0 / CAMERAS_FPS
	# tick_interval = 1.0 / (30 * CAMERAS_FPS)
	loop_start_time = _get_time_s()  # absolute reference point — never changes
	tick_number = 0  # monotonically increasing tick counter
	last_fresh_frame_time = loop_start_time  # for global "all streams dead" watchdog

	# ── Diagnostic timing state ───────────────────────────────────────────────────
	_slow_tick_threshold_ms = 1000 * tick_interval * 1.1 + 1  # log any tick slower than this
	print(f"Slow ticks are anything over {_slow_tick_threshold_ms}ms, where the expected tick length is {1000 * tick_interval}ms.")
	_section_times = {}  # rolling totals for summary (post-warmup only)
	_tick_count = 0
	_post_warmup_tick_count = 0
	_slow_tick_count = 0
	_slow_tick_log = []  # (post_warmup_tick_number, section_durations_dict)
	_snapshot_taken = False
	spreads = []


	# Wait until every stream thread has finished its flush and signalled ready
	logging.info("Waiting for all cameras to initialise…")
	t0 = _get_time_s()
	while True:
		if shutdown_flag.is_set():
			break
		if all(streams_active):
			logging.info("All cameras ready. Starting image collection loop")
			break
		if _get_time_s() - t0 > 60 and any(streams_active):
			logging.warning(f"Only {sum(streams_active)} cameras are ready. Starting image collection loop anyways, but black frames will be inserted for dead cameras.")
			break
		time.sleep(0.05)

	try:
		while not shutdown_flag.is_set():
			tick_start = _get_time_s()
			expected_tick_start = loop_start_time + tick_number * tick_interval
			tick_delay = tick_start - expected_tick_start  # positive = main loop is late
			_tick_count += 1
			is_warmup = _tick_count <= WARMUP_TICKS
			if not is_warmup:
				_post_warmup_tick_count += 1
			section = {}
			submit_list = []

			# ── SECTION: enclosure loop ───────────────────────────────────────────
			t = _get_time_s()
			for enclosure in enclosures:
				enc_t = _get_time_s()
				sample_idx = enclosure["_sample_index_in_interval"]
				need_frame = sample_idx in REQUIRED_FRAME_INDICES_IN_INTERVAL

				# -- buffer.get --
				t_bg = _get_time_s()
				if need_frame:
					frames_for_merge = []
					ages = []
					got_any_real = False
					for buffer in enclosure["stream_buffers"]:
						frame, is_real, age = buffer.get(None if is_warmup else tick_delay)
						if is_real:
							frames_for_merge.append(frame)
							ages.append(age)
							got_any_real = True
					if got_any_real:
						last_fresh_frame_time = _get_time_s()
						submit_list.append("sample")
						spreads.append(max(ages) - min(ages))
					section.setdefault("buffer_lock_wait_ms", 0)
					section.setdefault("buffer_conversion_ms", 0)
				section.setdefault("buffer_get_ms", 0)
				section["buffer_get_ms"] += (_get_time_s() - t_bg) * 1000

				# -- frame_infos build + frames_to_merge append --
				t_fi = _get_time_s()
				if need_frame and len(frames_for_merge) == enclosure["num_cameras"]:
					frame_infos = [
						{
							"stream_name": f"Cam{i}",
							"frame_index": enclosure["interval_index"] * FRAMES_PER_INTERVAL + sample_idx,
							"assumed_time": _get_assumed_time(
								enclosure["interval_index"] * FRAMES_PER_INTERVAL + sample_idx,
								enclosure["evaluation_start"]
							),
							"frame_index_in_interval": sample_idx,
							"frame": frames_for_merge[i],
						}
						for i in range(enclosure["num_cameras"])
					]
					merge_tuple = (frame_infos, enclosure["minimal version"])
					enclosure["frames_to_merge"].append(merge_tuple)
				section.setdefault("frame_info_build_ms", 0)
				section["frame_info_build_ms"] += (_get_time_s() - t_fi) * 1000

				# advance interval pointer
				enclosure["_sample_index_in_interval"] = (sample_idx + 1) % FRAMES_PER_INTERVAL

				# -- merge_pool.submit + batch_pool.submit --
				t_sub = _get_time_s()
				if len(enclosure["frames_to_merge"]) == IMAGES_PER_INTERVAL:
					enclosure_id = enclosure["enclosure_id"]
					global_frame_start = enclosure["interval_index"] * FRAMES_PER_INTERVAL
					batch_index = enclosure["interval_index"] // INTERVALS_PER_BATCH
					assumed_time = _get_assumed_time(global_frame_start, enclosure["evaluation_start"])

					containing_folder = (
						f"{TEMPORAL_LOCAL_STORAGE}single_images/{enclosure_id}"
						f"/{START_DATE_Y_M_D}/batch_{batch_index}"
					)
					file_name = f"{enclosure_id}_{START_DATE_YMD}-{assumed_time}.jpg"
					middle_file_path = f"{containing_folder}/images/{file_name}"
					differences_file_path = f"{containing_folder}/differences/{file_name}"

					future = merge_pool.submit(_merge_and_diffs, enclosure["frames_to_merge"], middle_file_path,
											   differences_file_path)
					submit_list.append(f"merge")
					enclosure["frames_to_merge"] = []

					is_last_interval_in_batch = (enclosure["interval_index"] + 1) % INTERVALS_PER_BATCH == 0

					if is_last_interval_in_batch:
						_submit_batch(
							[batch_index, enclosure["minimal version"], containing_folder, global_frame_start])
						submit_list.append(f"batch")
						if enclosure == enclosures[-1]:
							future.add_done_callback(_last_merge_of_batch_done_callback)
							with noo_batches_ready_to_process.get_lock():
								if batch_index > noo_batches_ready_to_process.value + 2:
									logging.warning("Merging is falling behind. Consider increasing number of workers.")

					enclosure["interval_index"] += 1

				section.setdefault("submit_ms", 0)
				section["submit_ms"] += (_get_time_s() - t_sub) * 1000

			section["enclosure_loop_ms"] = (_get_time_s() - t) * 1000

			# ── SECTION: watchdog ─────────────────────────────────────────────────
			t = _get_time_s()
			if NO_FRESH_FRAME_TIMEOUT < _get_time_s() - last_fresh_frame_time < NO_FRESH_FRAME_TIMEOUT * 2:
				logging.warning(
					f"No fresh frame received from any camera for "
					f"{_get_time_s() - last_fresh_frame_time:.1f}s. This is bad."
				)
			section["watchdog_ms"] = (_get_time_s() - t) * 1000

			# ── SECTION: observation period check ─────────────────────────────────
			t = _get_time_s()
			now_hour = datetime.datetime.now().hour
			all_periods_ended = all(_observation_period_ended(enc, now_hour) for enc in enclosures)
			section["obs_check_ms"] = (_get_time_s() - t) * 1000
			if all_periods_ended:
				logging.info("All observation periods have ended. Shutting down.")
				shutdown_flag.set()
				break

			# ── SECTION: tick sleep ───────────────────────────────────────────────
			t = _get_time_s()
			tick_number += 1
			next_tick_time = loop_start_time + tick_number * tick_interval
			sleep_for = next_tick_time - _get_time_s()
			if sleep_for > 0:
				timer.delay_noblock(int(sleep_for * 1000000), allow_sleep=True)
			section["sleep_ms"] = (_get_time_s() - t) * 1000

			# ── Diagnostic: log slow ticks ────────────────────────────────────────
			total_tick_ms = (_get_time_s() - tick_start) * 1000
			section["total_tick_ms"] = total_tick_ms

			if not is_warmup:
				# ── Rolling totals (post-warmup only) ─────────────────────────────────
				for k, v in section.items():
					_section_times[k] = _section_times.get(k, 0.0) + v

				if total_tick_ms > _slow_tick_threshold_ms:
					_slow_tick_count += 1
					_slow_tick_log.append((_post_warmup_tick_count, dict(section)))
					if DETAILLED_TIMING_DEBUGGING_MESSAGES:
						logging.warning(
							f"🐢 SLOW TICK #{_post_warmup_tick_count} (post-warmup) | "
							f"actions: {submit_list}"
							f"total={total_tick_ms:.1f}ms | "
							f"enclosure_loop={section['enclosure_loop_ms']:.1f}ms ("
							f"buffer_get={section['buffer_get_ms']:.1f} "
							f"frame_info_build={section['frame_info_build_ms']:.1f} "
							f"submit={section['submit_ms']:.1f}) | "
							f"watchdog={section['watchdog_ms']:.1f}ms | "
							f"obs_check={section['obs_check_ms']:.1f}ms | "
							f"sleep={section['sleep_ms']:.1f}ms"
						)

				# ── Periodic summary every 1000 post-warmup ticks ─────────────────────
				if _post_warmup_tick_count % 1000 == 0:
					if DETAILLED_TIMING_DEBUGGING_MESSAGES:
						logging.info(
							f"📊 TICK SUMMARY after {_post_warmup_tick_count} post-warmup ticks | "
							f"slow_ticks={_slow_tick_count} "
							f"({100 * _slow_tick_count / _post_warmup_tick_count:.1f}%) | "
						)

						print("Decode yield times (Age variation) stats:")
						_print_stats(decode_yield_times)

						print("\nMetric 1 (cam-spread error) stats:")
						_print_stats(spreads)

						diffs = []
						for enclosure in enclosures:
							for buffer in enclosure["stream_buffers"]:
								diffs += _get_differences(buffer.abs_delays)
						print("\nMetric 2 (interval error) stats:")
						_print_stats(diffs)

						# merge_pool.submit(save_distribution, diffs) # saving plot takes like 700ms.

				# ── Snapshot + optional benchmark shutdown at BENCHMARK_SHUTDOWN_TICK ─
				if not _snapshot_taken and _post_warmup_tick_count == BENCHMARK_SHUTDOWN_TICK:
					_snapshot_taken = True
					shutdown_flag.set()

	except Exception as e:
		print(e)
		shutdown_flag.set()
	t0 = _get_time_s()
	# ── Drain remaining partial batch ─────────────────────────────────────────
	# For each enclosure, if there are any complete intervals in the current
	# (not-yet-dispatched) batch, dispatch it now.
	for enclosure in enclosures:
		intervals_in_partial_batch = enclosure["interval_index"] % INTERVALS_PER_BATCH
		if intervals_in_partial_batch > 0:
			batch_index = enclosure["interval_index"] // INTERVALS_PER_BATCH
			global_frame_start = (enclosure["interval_index"] - intervals_in_partial_batch) * FRAMES_PER_INTERVAL
			containing_folder = (
				f"{TEMPORAL_LOCAL_STORAGE}single_images/{enclosure['enclosure_id']}"
				f"/{START_DATE_Y_M_D}/batch_{batch_index}"
			)
			_submit_batch([batch_index, enclosure["minimal version"], containing_folder, global_frame_start])
			logging.info(
				f"Dispatched partial batch {batch_index} for {enclosure['enclosure_id']} "
				f"({intervals_in_partial_batch} interval(s))."
			)

	# ── Wait for OD+AC to finish, then post-process ───────────────────────────
	if num_submitted_batches > 0:
		logging.info("Waiting for all OD&AC batches to complete…")
		time.sleep(1)
		noo_batches_ready_to_process.value += 2 # make sure batches actually process, even if keyboard interrupt was called between submit and increment.
		while num_completed_batches < num_submitted_batches:
			time.sleep(0.1)
		monitor.stop()
		batch_pool.shutdown(wait=False, cancel_futures=False)

		logging.info("Starting post-processing…")
		_merge_batches(num_completed_batches, enclosures)
		try:
			_apply_postprocessing_v2(enclosures)
		except Exception as e:
			print(_get_time_s(), e)
		try:
			_create_statistics_v2(enclosures)
		except Exception as e:
			print(_get_time_s(), e)
		try:
			_create_visualizations_v2(enclosures)
		except Exception as e:
			print(_get_time_s(), e)
		print("Total time taken for pp:", _get_time_s() - t0)
		logging.info("Finished post-processing. Thanks for using BOVIDS-Live!\n- Tobi")

	_shutdown_threads_and_windows(threads, enclosures)
	batch_pool.shutdown(wait=True, cancel_futures=True)
	merge_pool.shutdown(wait=True, cancel_futures=True)
	logging.info("Exiting…")


def launch_stream_capture_service():
	"""
	Entry point. Reads enclosure, individual, and prediction config from Excel,
	builds the list of per-enclosure dicts (RTSP URLs, evaluation window, AC tasks,
	post-processing rules, per-individual model paths, etc.), initialises shared
	globals (num_cameras, streams_ready), and calls run_stream_capture_service.
	"""

	# Load Excel files containing information about enclosures, individuals and what the stream service should do
	try:
		print(f"Configuration:  Stream xlsx: {STREAM_PREDICTION_XLSX},  "
			  f"FPS: {CAMERAS_FPS},  Interval length: {SECONDS_PER_INTERVAL}s,  Frames per Interval: {IMAGES_PER_INTERVAL},  "
			  f"Frame spacing: {IMAGE_SPACING_WITHIN_INTERVAL} frames,  Resource Monitor: {SHOW_RESOURCE_MONITOR},  "
			  f"Batch Size: {INTERVALS_PER_BATCH},  Batch Workers: {MAX_NUM_PARALLEL_BATCH_PROCESSES},  Merge Workers: "
			  f"{MAX_NUM_PARALLEL_MERGE_PROCESSES},  Smallest sample spacing: {SMALLEST_SAMPLE_SPACING}.")
		prediction_df = pd.read_excel(STREAM_PREDICTION_XLSX, index_col="enclosure_id")
	except Exception as e:
		logging.error(f"While trying to read the stream prediction file {STREAM_PREDICTION_XLSX} this error occurred: {e}.\nPlease check if it exists and is readable. Exiting...")
		return
	enclosure_df = get_enclosure_information(filtering=False)
	individual_df = get_individual_information()

	# Extract all necessary info into enclosure dicts
	enclosures = []
	for enclosure_id in prediction_df.index:
		enclosure = {
			"enclosure_id": enclosure_id,
			"rtsp_urls": prediction_df.loc[enclosure_id, "rtsp_stream_addresses"].split(";"),
			"evaluation_start": int(prediction_df.loc[enclosure_id, "evaluation_start"]),
			"evaluation_end": int(prediction_df.loc[enclosure_id, "evaluation_end"]),
			"dismiss_individuals": prediction_df.loc[enclosure_id, "dismiss_individuals"].split(";") if isinstance(
				prediction_df.loc[enclosure_id, "dismiss_individuals"], str) else [],
			"do_od": bool(prediction_df.loc[enclosure_id, "od"]),
			"od_task": enclosure_df.loc[enclosure_id, "task"],
			"ac_todos": [ac_mode for ac_mode in ["StLy", "StFo", "LHULHD"] if prediction_df.loc[enclosure_id, ac_mode]],
			"apply_postprocessing": bool(prediction_df.loc[enclosure_id, "apply_postprocessing"]),
			"stats_StLy": bool(prediction_df.loc[enclosure_id, "stats_StLy"]),
			"stats_subactions": bool(prediction_df.loc[enclosure_id, "stats_subactions"]),
			"individuals": {},
			"_sample_index_in_interval": 0,
		}

		individual_ids = set(enclosure_df.loc[enclosure_id, "individual_ids"].split(";")) - set(enclosure["dismiss_individuals"])
		for individual_id in individual_ids:
			enclosure["individuals"][individual_id] = {
				"name": individual_df.loc[individual_id, "individual_name"],
				"ac_network_paths": {},
				"post_processing_rulesets": {}
			}
			if enclosure["do_od"]:
				for ac_mode in enclosure["ac_todos"]:
					enclosure["individuals"][individual_id]["ac_network_paths"][ac_mode] = get_ac_networkpath(individual_df, individual_id, ac_mode)
					if enclosure["apply_postprocessing"]:
						pp_ruleset_name = individual_df.loc[individual_id, f"postproc_{ac_mode}"]
						enclosure["individuals"][individual_id]["post_processing_rulesets"][ac_mode] = get_postprocessing_rules(pp_ruleset_name, ac_mode)

		if not enclosure["apply_postprocessing"] and any([enclosure["stats_StLy"], enclosure["stats_subactions"]]):
			logging.warning(f"Enclosure {enclosure_id} has post-processing disabled but statistics are requested. This is not supported.")
			enclosure["stats_StLy"], enclosure["stats_subactions"] = False, False

		# For OD and AC
		enclosure["minimal version"] = {
			"enclosure_id": enclosure["enclosure_id"],
			"dismiss_individuals": enclosure.get("dismiss_individuals", []),
			"do_od": bool(enclosure.get("do_od", False)),
			"od_task": enclosure["od_task"],
			"ac_todos": list(enclosure.get("ac_todos", [])),
			"individuals": enclosure.get("individuals", {}).copy(),
			"evaluation_start": enclosure.get("evaluation_start")
		}

		enclosures.append(enclosure)

	# Set up some global variables
	global num_cameras, streams_ready, streams_active, previously_reported_timing_errors
	num_cameras = sum(len(enclosure["rtsp_urls"]) for enclosure in enclosures)
	streams_ready = Array('b', [False] * num_cameras)
	streams_active = Array('b', [False] * num_cameras)
	previously_reported_timing_errors = [0.0] * num_cameras

	# Finally, run the stream capture service main function
	run_stream_capture_service(enclosures)


# ____________________ MAIN CODE END ____________________
















# ____________________ HELPERS ____________________


# Statistics function: returns a list that contains the difference to the next value.
def _get_differences(values):
	diffs = []
	for i in range(len(values) - 1):
		diff = values[i + 1] - values[i]
		diffs.append(diff)
	return diffs


# Statistics function
def _print_stats(values_s):
	values_ms = [1000 * value for value in values_s]
	print(f"n: {len(values_ms)}. All following times in milliseconds.")
	print(f"avg: {float(numpy.mean(values_ms)):.2f}")
	print(f"std: {float(numpy.std(values_ms))}:.2f")
	print(f"min: {float(numpy.min(values_ms))}:.2f")
	print(f"max: {float(numpy.max(values_ms))}:.2f")
	print(f"p5: {float(numpy.percentile(values_ms, 5))}:.2f")
	print(f"p95: {float(numpy.percentile(values_ms, 95))}:.2f")
	print(f"p1: {float(numpy.percentile(values_ms, 1))}:.2f")
	print(f"p99: {float(numpy.percentile(values_ms, 99))}:.2f")


# Statistics function. Creates and saves distribution bar plot with 50 bins for values.
def _save_distribution(values_s):
	values_ms = [1000 * value for value in values_s]
	values_ms = values_ms * 10

	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	fig.suptitle(
		f"BOVIDS Sampling Distributions  —  {len(values_ms):,} post-warmup ticks",
		fontsize=13,
	)

	def _plot_hist(ax, data_ms: list[float], title: str, stats_ms: dict,
				   threshold_lines: list | None = None) -> None:
		if data_ms:
			ax.hist(data_ms, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
			if threshold_lines:
				seen_labels = set()
				for x_val, lbl, col in threshold_lines:
					display_lbl = lbl if lbl not in seen_labels else "_nolegend_"
					seen_labels.add(lbl)
					ax.axvline(x_val, color=col, ls="-", lw=1.0,
							   alpha=0.6, label=display_lbl)
			for val, lbl, col, ls in [
				(stats_ms["avg"], f"avg = {stats_ms['avg']:.2f} ms", "red", "--"),
				(stats_ms["p1"], f"1P  = {stats_ms['p1']:.2f} ms", "orange", ":"),
				(stats_ms["p99"], f"99P = {stats_ms['p99']:.2f} ms", "green", ":"),
			]:
				if not numpy.isnan(val):
					ax.axvline(val, color=col, ls=ls, lw=1.2, label=lbl)
			ax.legend(fontsize=8, loc="upper right")
			ax.margins(y=0.30)
		else:
			ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
		ax.set_title(title)
		ax.set_xlabel("ms")
		ax.set_ylabel("Count")


	_tick_ms_single = (1 / CAMERAS_FPS) * 1000
	_plot_hist(axes[1], values_ms, "Metric 2 · Sampling Spacing Error (pooled)",
			   {"avg": float(numpy.mean(values_ms)), "p1": float(numpy.percentile(values_ms, 1)), "p99": float(numpy.percentile(values_ms, 99))},
			   threshold_lines=[(_tick_ms_single / 2, "±½ tick", "orange"), (-_tick_ms_single / 2, "±½ tick", "orange")]
			   )

	plt.tight_layout()

	path = "C:\\outputs\\distribution.png"
	ensure_directory(path)
	plt.savefig(path, dpi=300)
	plt.close(fig)
	logging.info(f"Saved plots to {path}")


# Python 3.7+: More accurate version of time.time() that has better accuracy than 16ms
def _get_time_s():
	return time.perf_counter_ns() / (10 ** 9)


# Handles SIGINT (Ctrl+C) to trigger a controlled shutdown of the streaming service by setting the shutdown flag.
def _signal_handler(_, _2):
	logging.info("🛑 Ctrl+C detected. Attempting controlled shutdown...")
	shutdown_flag.set()


# Cleanup to safely release rtsp stream resources and close windows
def _cleanup_capture_device(container, window_name):
	try:
		if container is not None:
			container.close()
			logging.info(f"{window_name} closed successfully")
	except Exception as e:
		logging.warning(f"{window_name} potentially didn't release resources because: {e}")

	if SHOW_WINDOWS:
		try:
			cv2.destroyWindow(window_name)
		except cv2.error:
			pass


# Calculates at what time a frame was captured based on its global frame_index and the enclosure’s evaluation start hour
def _get_assumed_time(frame_index, evaluation_start):
	total_seconds = frame_index // CAMERAS_FPS
	assumed_second = total_seconds % 60
	assumed_minute = (total_seconds // 60) % 60
	assumed_hour = ((total_seconds // 3600) + evaluation_start) % 24
	return datetime.time(hour=assumed_hour, minute=assumed_minute, second=assumed_second).strftime("%H%M%S")


# TODO: Known limitation: BOVIDS-Live will only exit when observation periods of all enclosures have ended.
# Returns True when the wall-clock hour has passed the enclosure's evaluation_end.
def _observation_period_ended(enclosure: dict, now_hour: int) -> bool:
	start = enclosure["evaluation_start"]
	end = enclosure["evaluation_end"]
	if start < end:          # same-day window  e.g. 8 → 18
		return now_hour >= end
	elif start > end:        # overnight window e.g. 20 → 6
		return start > now_hour and now_hour >= end
	else:
		logging.warning("Observation period's end time is the same as the start time. Invalid parameters. Exiting.")
		return True


# Join all camera threads and clean up OpenCV windows.
def _shutdown_threads_and_windows(threads: list[Thread], enclosures: list[dict]) -> None:
	timeout = (8.0 * (num_cameras or 1)) / CAMERAS_FPS
	for t in threads:
		t.join(timeout=timeout)
	if SHOW_WINDOWS:
		cv2.destroyAllWindows()
	if not any(streams_ready):
		logging.warning(
			"No stream ever delivered a frame. "
			"Please check your RTSP URLs and camera connections."
		)


# Mimics server.bovids_v2.lib.image_manipulation.IO_video_timeinterval but much quicker.
def _merge_frames_optimized(frame_infos, enclosure):
	frames = [frame_info["frame"] for frame_info in frame_infos]

	# Adjust gamma and convert to grayscale but encode back into RGB (don't ask why, I have no clue)
	# At least there's no need to worry about opencv2's BGR format anymore, since all channels are equal now
	if COMPATIBILITY_MODE:
		frames = [adjust_gamma(gray2rgb(rgb2gray(frame)), gamma=0.75) for frame in frames]
	else:
		frames = [
		cv2.cvtColor(
			cv2.LUT(
				frame.astype(numpy.uint8),
				numpy.array([((i / 255.0) ** 0.75) * 255 for i in range(256)], dtype="uint8")
			),
			cv2.COLOR_GRAY2RGB
		) for frame in frames
	]

	# Merge frames into one image and then black out any overlapping parts using the .json file provided by the user
	merged_image = merge_images(frames, enclosure["enclosure_id"]) if COMPATIBILITY_MODE else merge_images_optimized(frames, enclosure["enclosure_id"])
	return merged_image


ac_times_per_enclosure = []
ac_times_per_mode = {}
total_inferece_times = []
def _do_OD_and_AC_on_batch(args):
	"""
	Performs object detection (OD) and action classification (AC) on a single batch of merged images for an enclosure.
	Runs OD if enabled, then conducts AC for each requested mode and individual using the corresponding network paths.
	Saves results to batch-specific folders and logs success or errors.

	Args:
	args (tuple): Contains the following elements:
	batch_index (int): The batch index being processed.
	enclosure (dict): Dictionary containing enclosure and individual configuration, including AC modes and paths.
	merged_images_path (str): Path to the folder containing merged interval images for the batch.
	index_of_first_frame_in_last_interval (int): Global frame index of the first frame in the last interval of the batch.

	Returns:
	tuple: (success (bool), batch_index (int)) indicating whether processing succeeded and which batch was processed.
	"""
	# return False, 0
	(batch_index, enclosure, merged_images_path, index_of_first_frame_in_last_interval) = args
	while True:
		with _shared_noo_batches_ready_to_process.get_lock():
			if _shared_noo_batches_ready_to_process.value > batch_index:
				break
		time.sleep(1)
	logging.info(f"Starting OD&AC on batch {batch_index} for enclosure {enclosure['enclosure_id']}...")

	start_time = _get_time_s()
	success = True
	try:
		# Do object detection if requested in predict_stream_example.xlsx
		enclosure_id = enclosure["enclosure_id"]
		od_output_folder = f"{TEMPORAL_LOCAL_STORAGE}/predicted_images/{enclosure_id}/{START_DATE_Y_M_D}/batch_{batch_index}/"
		if enclosure["do_od"]:
			predict_folder_differences(
				f"{merged_images_path}/images/",
				f"{merged_images_path}/differences/",
				od_output_folder,
				od_output_folder,
				enclosure_id,
				enclosure["od_task"],
				DEVICE,
				enclosure["dismiss_individuals"],
				START_DATE_Y_M_D
			)

		t0_ac = _get_time_s()
		# Conduct AC for each mode requested in predict_stream_example.xlsx, for each individual separately
		for ac_mode in enclosure["ac_todos"]:
			t0_mode = _get_time_s()
			for individual in enclosure["individuals"]:
				potentially_available_image_names = {individual: {}}
				index_of_first_frame_in_batch = index_of_first_frame_in_last_interval - (INTERVALS_PER_BATCH - 1) * FRAMES_PER_INTERVAL
				potentially_available_image_names[individual][START_DATE_Y_M_D] \
					= ([f"{START_DATE_YMD}-{assumed_time}_{individual}" for assumed_time in
						[_get_assumed_time(index_of_first_frame_in_batch + i * FRAMES_PER_INTERVAL,
						                   enclosure['evaluation_start']) for i in range(INTERVALS_PER_BATCH)]
						])
				path_ac_save = f"{TEMPORAL_LOCAL_STORAGE}/ac_batch_results/batch_{batch_index}/{individual}/raw/{ac_mode}/prediction/"
				ac_input_folder = f"{od_output_folder}/{individual}/images/"
				input_imagenames_use = get_images_ac_mode(
					ac_mode,
					f"{TEMPORAL_LOCAL_STORAGE}/ac_batch_results/batch_{batch_index}/{individual}/raw/StLy/prediction/",
					START_DATE_Y_M_D,
					individual,
					f"{ac_input_folder}0/",
				)
				ensure_directory(path_ac_save)
				ensure_directory(ac_input_folder)  # usually done within OD, but if OD does not detect individual, folder won't exist
				predict_folder_ac([
					ac_input_folder,
					input_imagenames_use,
					enclosure["individuals"][individual]["ac_network_paths"][ac_mode],
					ac_mode,
					DEVICE,
					path_ac_save,
					START_DATE_Y_M_D,
					individual,
					f"{TEMPORAL_LOCAL_STORAGE}/ac_batch_results/batch_{batch_index}/{individual}/raw/StLy/prediction/",
					potentially_available_image_names,
				])
			if not ac_times_per_mode.keys().__contains__(ac_mode): ac_times_per_mode[ac_mode] = []
			if not batch_index == 0: ac_times_per_mode[ac_mode].append(_get_time_s() - t0_mode)
		if not batch_index == 0: ac_times_per_enclosure.append(_get_time_s() - t0_ac)
		if not batch_index == 0 and len(ac_times_per_enclosure) % 1 == 0:
			print("\nAC time per enclosure (just like predict_folder_differences):")
			_print_stats(ac_times_per_enclosure)
			print("\nAC time by mode per enclosure:")
			for ac_mode, times in ac_times_per_mode.items():
				print(f" - {ac_mode}:")
				_print_stats(times)
	except Exception as e:
		logging.error(f"Error during OD&AC on batch {batch_index}: {e}")
		success = False
	logging.info(f"OD&AC on batch {batch_index} {'completed processing in' if success else 'errored out after processing for'} {_get_time_s() - start_time:.2f} seconds.")
	return success, batch_index



def _merge_batches(num_batches, enclosures):
	"""
	Merges AC batch results and behavior sequences for each individual and mode into single CSV files per individual/mode.
	Iterates over all enclosures, individuals, and ac_todos, reads each batch file if it exists, concatenates them,
	and writes the merged results to the designated output folder. Logs warnings for any batch files that cannot be read.

	Args:
	num_batches (int): Total number of AC batches to merge.
	enclosures (list[dict]): List of enclosure dicts containing individuals and their AC modes.
	"""
	for enclosure in enclosures:
		for individual in enclosure["individuals"]:
			for ac_mode in enclosure["ac_todos"]:
				# Files to save merged results to
				output_folder = f"{BOVIDS_AC}/{individual}/raw/{ac_mode}/prediction"
				ac_results_file = ensure_directory(f"{output_folder}/{START_DATE_Y_M_D}_{individual}_{ac_mode}.csv")
				ac_sequence_file = f"{output_folder}/{START_DATE_Y_M_D}_{individual}_{ac_mode}_behavior_seq.csv"

				ac_results_df = pd.DataFrame()
				ac_sequence_df = pd.DataFrame()

				for batch_index in range(num_batches):
					batch_ac_results_file = f"{TEMPORAL_LOCAL_STORAGE}/ac_batch_results/batch_{batch_index}/{individual}/raw/{ac_mode}/prediction/{START_DATE_Y_M_D}_{individual}_{ac_mode}.csv"
					batch_ac_sequence_file = f"{TEMPORAL_LOCAL_STORAGE}/ac_batch_results/batch_{batch_index}/{individual}/raw/{ac_mode}/prediction/{START_DATE_Y_M_D}_{individual}_{ac_mode}_behavior_seq.csv"
					if os.path.exists(batch_ac_results_file):
						try:
							batch_ac_results_df = pd.read_csv(batch_ac_results_file)
							if batch_ac_results_df.dropna().empty:
								continue
							ac_results_df = pd.concat([ac_results_df, batch_ac_results_df], ignore_index=True)
						except Exception as e:
							logging.warning(f"Non-critical error: Could not read {batch_ac_results_file}: {e}")

					if os.path.exists(batch_ac_sequence_file):
						try:
							batch_ac_sequence_df = pd.read_csv(batch_ac_sequence_file, header=None)
							ac_sequence_df = pd.concat([ac_sequence_df, batch_ac_sequence_df], ignore_index=True)
						except Exception as e:
							logging.warning(f"Non-critical error: Could not read {batch_ac_sequence_file}: {e}")

				if not ac_results_df.empty:
					ensure_directory(output_folder)
					ac_results_df.to_csv(ac_results_file, index=False)

				if not ac_sequence_df.empty:
					ensure_directory(output_folder)
					ac_sequence_df.to_csv(ac_sequence_file, index=False)


# Adjusted version of apply_postprocessing(...) in pipeline_functions.py
def _apply_postprocessing_v2(enclosures):
	for enclosure in enclosures:
		if enclosure["apply_postprocessing"]:
			for individual in enclosure["individuals"]:
				if "StLy" in enclosure["ac_todos"]:
					# Gather data
					stly_seq, stly_seq_path_pp, stly_info_path_pp = get_postprocessing_stly_paths(BOVIDS_AC, individual,
																								  START_DATE_Y_M_D)
					ruleset_df = enclosure["individuals"][individual]["post_processing_rulesets"]["StLy"]
					post_processor = PostProcessor(ruleset_df, "min_length", SECONDS_PER_INTERVAL, stly_seq)

					# Filter out short phases in the behavior sequence
					pp_stly_seq, pp_stly_time = post_processor.filter_short_phases(
						post_processor.original_behavior_sequence,
						post_processor.original_time_sequence,
						post_processor.rule_set_time,
						post_processor.rule_set_behavior
					)

					# save post processed sequence as csv file
					post_processor.save_post_processed_sequence(
						pp_stly_seq,
						pp_stly_time,
						stly_seq_path_pp,
						stly_info_path_pp,
						enclosure["evaluation_start"]
					)

					if "StFo" in enclosure["ac_todos"] or "LHULHD" in enclosure["ac_todos"]:
						# Gather data
						stfo_seq, lhulhd_seq, all_modes_seq_path_pp, all_modes_info_path_pp \
							= get_postprocessing_subaction_paths(BOVIDS_AC, individual, START_DATE_Y_M_D)
						ruleset_stfo_df = enclosure["individuals"][individual]["post_processing_rulesets"]["StFo"]
						ruleset_lhulhd_df = enclosure["individuals"][individual]["post_processing_rulesets"]["LHULHD"]
						ruleset_subactions = pd.concat([ruleset_lhulhd_df, ruleset_stfo_df], ignore_index=True)
						pp_subactions = PostProcessorSubactions(pp_stly_seq, pp_stly_time, SECONDS_PER_INTERVAL)

						# Merge all three sequences
						pp_behav_seq = pp_subactions.incorporate_subactions_sequence(
							pp_subactions.stly_behavior, stfo_seq, lhulhd_seq
						)

						# filter out short phases
						post_processor_subaction = PostProcessor(
							ruleset_subactions,
							"min_length",
							SECONDS_PER_INTERVAL,
							pp_behav_seq,
						)
						pp_subaction_seq, pp_subaction_time = (
							post_processor_subaction.filter_short_phases(
								post_processor_subaction.original_behavior_sequence,
								post_processor_subaction.original_time_sequence,
								post_processor_subaction.rule_set_time,
								post_processor_subaction.rule_set_behavior,
							)
						)

						# save post processed sequence with subactions as csv file
						pp_subaction_seq = [int(action) for action in pp_subaction_seq]
						post_processor_subaction.save_post_processed_sequence(
							pp_subaction_seq,
							pp_subaction_time,
							all_modes_seq_path_pp,
							all_modes_info_path_pp,
							enclosure["evaluation_start"]
						)


# Adjusted version of create_statistics(...) in pipeline_functions.py
def _create_statistics_v2(enclosures):
	for enclosure in enclosures:
		for individual in enclosure["individuals"]:
			if enclosure["stats_StLy"] or enclosure["stats_subactions"]:
				# Gather data
				df_pp, df_pp_stly, phases_stly_file, phases_subactions_file, output_stats_path = get_statistics_paths(
					BOVIDS_STATS, individual, START_DATE_Y_M_D, BOVIDS_AC,
					[enclosure["stats_StLy"], enclosure["stats_subactions"]])

				# get all standing & lying cycles
				standing_list = get_cycles_behavior(df_pp_stly, START_DATE_Y_M_D, individual, 1)
				lying_list = get_cycles_behavior(df_pp_stly, START_DATE_Y_M_D, individual, 2)

				# create cycle file
				cycle_file_path = f"{output_stats_path}{START_DATE_Y_M_D}_{individual}_cycles.csv"
				df_cycles = create_cycle_csv(
					standing_list,
					lying_list,
					df_pp_stly["seq_behavior"][0],
					df_pp_stly["seq_behavior"][1],
					cycle_file_path,
				)

				# create and save statistic evaluation
				if enclosure["stats_StLy"]:
					df_phases_stly = create_phase_csv(df_pp_stly, START_DATE_Y_M_D, individual,
																 phases_stly_file)
				if enclosure["stats_subactions"]:
					df_phases_subactions = create_phase_csv(
						df_pp, START_DATE_Y_M_D, individual, phases_subactions_file
					)

				# Create and save key values files
				key_values_file_path = f"{output_stats_path}{START_DATE_Y_M_D}_{individual}_key_values.csv"
				if enclosure["stats_StLy"]:
					create_key_values_csv(
						df_cycles,
						df_phases_stly,
						False,
						standing_list[5],
						lying_list[5],
						key_values_file_path,
					)
				if enclosure["stats_subactions"]:
					create_key_values_csv(
						df_cycles,
						df_phases_subactions,
						True,
						standing_list[5],
						lying_list[5],
						key_values_file_path
					)


# Adjusted version of create_visualizations(...) in pipeline_functions.py
def _create_visualizations_v2(enclosures):
	# lists to save behaviors for y axis
	behaviors_to_plot = []
	behaviors_to_plot_sub = []
	datetime_format = "%Y-%m-%d %H:%M:%S"
	for enclosure in enclosures:
		for individual in enclosure["individuals"]:
			if enclosure["stats_StLy"] or enclosure["stats_subactions"]:
				# create dicts to store stly, subactions and the combination of both
				stly_visualizer = {v: [] for v in BEHAVIOR_STATISTICS.values()}
				subactions_visualizer = {v: [] for v in BEHAVIOR_STATISTICS.values()}
				total_visualizer = {v: [] for v in BEHAVIOR_STATISTICS.values()}

				# path to get statistics
				output_stats_path = f"{BOVIDS_STATS}details/{individual}/"
				ensure_directory(output_stats_path)

				# check if StLy visualization should be done
				if enclosure["stats_StLy"]:
					phases_stly_file = (f"{output_stats_path}{START_DATE_Y_M_D}_{individual}_phases_stly.csv")
					df_phases_stly = pd.read_csv(phases_stly_file)

					# append information to visualize StLy
					stly_visualizer, total_visualizer = get_stly_visualization_info(
						df_phases_stly, datetime_format, stly_visualizer, total_visualizer
					)
					for key, value in total_visualizer.items():
						if value:
							behaviors_to_plot.append(key)
				else:
					if enclosure["stats_subactions"]:
						logging.warning(f"Subactions visuals can only be made if StLy visuals are also requested. Skipping {individual}.")
					continue

				# check if subactions visualization should be done
				if enclosure["stats_subactions"]:
					phases_subactions_file = (f"{output_stats_path}{START_DATE_Y_M_D}_{individual}_phases_subactions.csv")
					df_phases_subactions = pd.read_csv(phases_subactions_file)

					# append information to visualize subactions and combination
					subactions_visualizer, total_visualizer = get_subactions_visualization_info(
						df_phases_subactions, datetime_format, subactions_visualizer, total_visualizer
					)
					for key, value in subactions_visualizer.items():
						if value:
							behaviors_to_plot_sub.append(key)

				# visualization save paths
				output_visual_path = f"{BOVIDS_VISUAL}{individual}/"
				ensure_directory(output_visual_path)
				image_total = f'{output_visual_path}{START_DATE_Y_M_D}_{individual}_visualization_total.png'
				image_binary = f'{output_visual_path}{START_DATE_Y_M_D}_{individual}_visualization_binary.png'
				image_sub = f'{output_visual_path}{START_DATE_Y_M_D}_{individual}_visualization_subactions.png'

				# timepoints for x axis
				plot_start = START_DATE_Y_M_D + ' ' + df_phases_stly["start"][0]
				plot_start_datetime = datetime.datetime.strptime(plot_start, datetime_format)
				plot_end = START_DATE_Y_M_D + ' ' + df_phases_stly["end"].iloc[-1]
				# calculate time axis for plot, assumption: end time must be after date change
				plot_end_datetime = datetime.datetime.strptime(plot_end, datetime_format) + datetime.timedelta(days=1)

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
					enclosure["individuals"][individual]["name"],
					plot_start_datetime,
					plot_end_datetime,
					enclosure["stats_subactions"]
				)


# Only helps when PC runs other stuff next to BOVIDS-Live
def _set_process_priority_high():
	import psutil
	import os
	p = psutil.Process(os.getpid())
	p.nice(psutil.HIGH_PRIORITY_CLASS)


# Sets priority to lowest and transfers over the mergeing done counter from the main process.
def _init_batch_processes(counter):
	import psutil
	import os
	p = psutil.Process(os.getpid())
	p.nice(psutil.IDLE_PRIORITY_CLASS)
	global _shared_noo_batches_ready_to_process
	_shared_noo_batches_ready_to_process = counter


# Called once for each interval for each enclosure
def _merge_and_diffs(merge_tuples, path_middle, path_diffs):
	# return
	try:
		merged_frames = [_merge_frames_optimized(frames, enclosure) for frames, enclosure in merge_tuples]
		middle_image, differences_image = get_difference_images(merged_frames)
		if COMPATIBILITY_MODE:
			save_image_to_file_float64(middle_image, path_middle)
			save_image_to_file_float64(differences_image, path_diffs)
		else:
			save_image_to_file_unit8(middle_image, path_middle)
			save_image_to_file_unit8(differences_image, path_diffs)
	except Exception as e:
		print(f"Error during frame merging/difference image creation: {e}")



if __name__ == "__main__":
	try:
		launch_stream_capture_service()
	except Exception as e:
		print(e)
		import traceback
		traceback.print_exc()
	finally:
		input("\nPress Enter to exit...")