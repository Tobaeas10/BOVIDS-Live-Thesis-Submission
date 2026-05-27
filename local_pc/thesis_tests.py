# TODO: Remove before merging the fork. This is only for the thesis submition. Also remove all testing xlxs's

from functools import wraps
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor

PRINT_SPACING_ITERATION = 1


"""
compare_frames.py

Compares a frame decoded from an RTSP stream against frames decoded from disk
using MoviePy, to verify pixel-level consistency for ML pipelines.
"""
if __name__ == "__main__":
	import os
	import sys
	import cv2
	import av
	from moviepy.editor import VideoFileClip
	from skimage.metrics import structural_similarity as ssim

	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	RTSP_URL = "rtsp://127.0.0.1:8554/"
	VIDEO_PATH = "D:\\Programs\\RTSP Emulation\\deployment pkg\\videos\\2.avi"  # adjust if needed
	FLUSH_FRAMES = 20  # number of frames to discard before capturing


	# ── 1-3. Open stream, flush N frames, capture frame N+1 ──────────────────────

	print(f"[stream] Connecting to {RTSP_URL} ...")
	container = av.open(RTSP_URL)

	stream_frame_bgr = None
	captured_index = 0

	i = 0
	for av_frame in container.decode(video=0):
		if i < FLUSH_FRAMES:
			print(f"[stream] Flushing frame {i + 1}/{FLUSH_FRAMES}")
			i += 1
			continue
		# Frame 21 (0-based index 20)
		stream_frame_bgr = av_frame.to_ndarray(format="bgr24")
		captured_index = i
		print(f"\n[stream] Captured frame index {captured_index} from stream.")
		break

	container.close()
	print("[stream] Stream closed.")

	if stream_frame_bgr is None:
		sys.exit("[error] Failed to capture a frame from the stream.")


	# ── 4. Save stream frame to disk ──────────────────────────────────────────────

	stream_out_path = os.path.join(SCRIPT_DIR, "stream_frame.png")
	cv2.imwrite(stream_out_path, stream_frame_bgr)
	print(f"[saved] Stream frame → {stream_out_path}")


	# ── 5. Ask for frame index range ──────────────────────────────────────────────

	print("\nEnter a range of frame indices to decode from the video file.")
	start_idx = int(input("  Start index (inclusive): ").strip())
	end_idx   = int(input("  End index   (inclusive): ").strip())

	if start_idx > end_idx:
		sys.exit("[error] Start index must be <= end index.")


	# ── 6-7. Decode frames from disk with MoviePy and save ────────────────────────

	print(f"\n[disk] Loading {VIDEO_PATH} ...")
	clip = VideoFileClip(VIDEO_PATH)
	fps = clip.fps
	print(f"[disk] FPS = {fps}")

	disk_frames = {}   # index → numpy RGB array

	for idx in range(start_idx, end_idx + 1):
		t = idx / fps  # convert frame index to timestamp in seconds
		frame_rgb = clip.get_frame(t)  # returns H×W×3 uint8 RGB
		disk_frames[idx] = frame_rgb

		out_path = os.path.join(SCRIPT_DIR, f"disk_frame_{idx:05d}.png")
		# MoviePy returns RGB; save as BGR for OpenCV
		cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
		print(f"[saved] Disk frame {idx} → {out_path}")

	clip.reader.close()
	clip.audio = None
	print(f"[disk] {len(disk_frames)} frame(s) saved.")


	# ── 8. Ask which disk frame corresponds to the captured stream frame ──────────

	match_idx = int(input(
		f"\nWhich disk frame index corresponds to the stream frame? "
		f"[{start_idx}–{end_idx}]: "
	).strip())

	if match_idx not in disk_frames:
		sys.exit(f"[error] Index {match_idx} was not in the decoded range.")


	# ── 9. Compare the two frames ─────────────────────────────────────────────────

	disk_frame_rgb = disk_frames[match_idx]
	disk_frame_bgr = cv2.cvtColor(disk_frame_rgb, cv2.COLOR_RGB2BGR)

	stream = stream_frame_bgr.astype(np.float64)
	disk   = disk_frame_bgr.astype(np.float64)

	# — Exact equality —
	exact_match = np.array_equal(stream_frame_bgr, disk_frame_bgr)

	# — Per-pixel MSE (mean over all pixels AND channels) —
	mse_total = np.mean((stream - disk) ** 2)

	# — Per-channel MSE —
	mse_b = np.mean((stream[:, :, 0] - disk[:, :, 0]) ** 2)
	mse_g = np.mean((stream[:, :, 1] - disk[:, :, 1]) ** 2)
	mse_r = np.mean((stream[:, :, 2] - disk[:, :, 2]) ** 2)

	# — SSIM (computed on grayscale) —
	stream_gray = cv2.cvtColor(stream_frame_bgr, cv2.COLOR_BGR2GRAY)
	disk_gray   = cv2.cvtColor(disk_frame_bgr,   cv2.COLOR_BGR2GRAY)
	ssim_score, ssim_map = ssim(stream_gray, disk_gray, full=True)

	# Save the SSIM difference map for visual inspection
	ssim_vis = (ssim_map * 255).astype(np.uint8)
	ssim_path = os.path.join(SCRIPT_DIR, "ssim_diff_map.png")
	cv2.imwrite(ssim_path, ssim_vis)

	# — Print results —
	separator = "─" * 52
	print(f"\n{separator}")
	print("  COMPARISON RESULTS")
	print(separator)
	print(f"  Stream frame (captured)  : stream_frame.png")
	print(f"  Disk frame   (index {match_idx:>5}): disk_frame_{match_idx:05d}.png")
	print(separator)
	print(f"  Exact pixel match        : {exact_match}")
	print(f"  MSE (overall)            : {mse_total:.6f}")
	print(f"  MSE  B channel           : {mse_b:.6f}")
	print(f"  MSE  G channel           : {mse_g:.6f}")
	print(f"  MSE  R channel           : {mse_r:.6f}")
	print(f"  SSIM score               : {ssim_score:.6f}  (1.0 = identical)")
	print(f"  SSIM diff map saved      : ssim_diff_map.png")
	print(separator)

	if exact_match:
		print("  ✓ Frames are IDENTICAL at the pixel level.")
	else:
		diff = np.abs(stream - disk)
		print(f"  ✗ Frames differ.")
		print(f"    Max abs difference : {diff.max():.0f}  (out of 255)")
		print(f"    Mean abs difference: {diff.mean():.4f}")
	print(separator)



BASE_PORT = 8554
BASE_URL = "rtsp://127.0.0.1:"

def generate_rtsp_addresses(counts: list[int]) -> list[str]:
	# Assign ports sorted by stream count ascending (fewest streams = lowest ports)
	# but preserve original order in output
	indexed = sorted(enumerate(counts), key=lambda x: x[1])

	port = BASE_PORT
	port_map = {}  # original index -> list of ports
	for original_idx, count in indexed:
		port_map[original_idx] = list(range(port, port + count))
		port += count

	results = []
	for i, count in enumerate(counts):
		urls = [f"{BASE_URL}{p}/" for p in port_map[i]]
		results.append(";".join(urls))

	return results


def main():
	while True:
		try:
			raw = input("Enter stream counts (space-separated integers): ").strip()
			if not raw:
				continue
			counts = list(map(int, raw.split()))
			if any(c < 0 for c in counts):
				print("All counts must be non-negative.\n")
				continue
		except ValueError:
			print("Invalid input — please enter integers only.\n")
			continue

		addresses = generate_rtsp_addresses(counts)
		print()
		for line in addresses:
			print(line)
		print()


# if __name__ == "__main__":
# 	main()


# ── worker must be defined at module level (picklable) ──────────────────────
def dummy(frames):
	return len(frames)


def bench_pickle(frames):
	"""Isolated pickle round-trip benchmark (no IPC overhead)."""
	t0 = time.perf_counter()
	data = pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL)
	t1 = time.perf_counter()
	frames_back = pickle.loads(data)
	t2 = time.perf_counter()

	ser_ms   = (t1 - t0) * 1000
	deser_ms = (t2 - t1) * 1000
	size_mb  = len(data) / (1024 ** 2)
	print(f"  Pickle size   : {size_mb:.1f} MB")
	print(f"  Serialise     : {ser_ms:.1f} ms")
	print(f"  Deserialise   : {deser_ms:.1f} ms")
	print(f"  Round-trip    : {ser_ms + deser_ms:.1f} ms")
	return ser_ms, deser_ms


def bench_ipc(frames, n_warmup=1, n_measured=1000):
	with ProcessPoolExecutor(max_workers=1) as executor:
		for _ in range(n_warmup):
			executor.submit(dummy, frames).result()

		times = []
		for _ in range(n_measured):
			t0 = time.perf_counter()
			executor.submit(dummy, frames).result()
			t1 = time.perf_counter()
			times.append((t1 - t0) * 1000)

	times_sorted = sorted(times)
	n = len(times_sorted)
	pct1_count = max(1, int(n * 0.01))   # at least 1 sample

	bottom_1pct = times_sorted[:pct1_count]
	top_1pct    = times_sorted[-pct1_count:]

	avg   = sum(times) / n
	p50   = times_sorted[int(n * 0.50)]
	p95   = times_sorted[int(n * 0.95)]
	p99   = times_sorted[int(n * 0.99)]

	print(f"  Runs          : {n}")
	print(f"  Avg           : {avg:.1f} ms")
	print(f"  Median (p50)  : {p50:.1f} ms")
	print(f"  p95           : {p95:.1f} ms")
	print(f"  p99           : {p99:.1f} ms")
	print(f"  1% fastest ({pct1_count} samples) — avg {sum(bottom_1pct)/pct1_count:.1f} ms, "
		  f"min {bottom_1pct[0]:.1f} ms, max {bottom_1pct[-1]:.1f} ms")
	print(f"  1% slowest ({pct1_count} samples) — avg {sum(top_1pct)/pct1_count:.1f} ms, "
		  f"min {top_1pct[0]:.1f} ms,  max {top_1pct[-1]:.1f} ms")

	return avg, times_sorted


# if __name__ == '__main__':
# 	N_FRAMES = 6
# 	frames = [np.random.randint(0, 256, (1080, 1920, 1), dtype=np.uint8)
# 			  for _ in range(N_FRAMES)]
#
# 	print(f"\n=== Pickle benchmark ({N_FRAMES}x FHD frames) ===")
# 	ser_ms, deser_ms = bench_pickle(frames)
#
# 	print(f"\n=== IPC benchmark (ProcessPoolExecutor, 1 worker) ===")
# 	bench_ipc(frames)

class _RunStats:
	"""Reusable stats accumulator — shared by the timer decorator and LapTimer."""

	def __init__(self, name):
		self.name       = name
		self.times      = []
		self.call_count = 0   # includes the ignored warm-up slot

	def record(self, elapsed):
		"""Add a measurement. Returns True if a print should occur."""
		self.call_count += 1
		if self.call_count == 1:          # skip first (warm-up)
			print("warmup run for", self.name, "took:", elapsed, "seconds.")
			return False
		self.times.append(elapsed)
		n = len(self.times)
		return n % PRINT_SPACING_ITERATION == 0 and n <= 200

	def print_stats(self, label=None):
		times = self.times
		n     = len(times)
		if n == 0:
			return
		mean = sum(times) / n
		mn   = min(times)
		mx   = max(times)
		std  = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
		tag  = label or self.name

		print(
			f"\n{'─' * 44}\n"
			f"  {tag} — stats after {n} runs\n"
			f"{'─' * 44}\n"
			f"  mean : {mean * 1000:>10.4f} ms\n"
			f"  min  : {mn   * 1000:>10.4f} ms\n"
			f"  max  : {mx   * 1000:>10.4f} ms\n"
			f"  std  : {std  * 1000:>10.4f} ms\n"
			f"  p5  : {np.percentile(times, 5)  * 1000:>10.4f} ms\n"
			f"  p95  : {np.percentile(times, 95)  * 1000:>10.4f} ms\n"
			f"  p1  : {np.percentile(times, 1)  * 1000:>10.4f} ms\n"
			f"  p99  : {np.percentile(times, 99)  * 1000:>10.4f} ms\n"
			f"{'─' * 44}"
		)


def timer(func):
	"""Decorator: tracks total execution time per call."""
	stats = _RunStats(f"{func.__name__}()")

	@wraps(func)
	def wrapper(*args, **kwargs):
		start  = time.perf_counter()
		result = func(*args, **kwargs)
		elapsed = time.perf_counter() - start

		if stats.record(elapsed):
			stats.print_stats()

		return result
	return wrapper


class LapTimer:
	"""
	Context-manager lap timer for loops — reuses _RunStats.

	Usage:
		lap = LapTimer("my_loop")
		for item in data:
			with lap:
				process(item)
	"""

	def __init__(self, name):
		# self.stats = _RunStats(name)
		pass

	def __enter__(self):
		# self._start = time.perf_counter()
		pass

	def __exit__(self, *_):
		# elapsed = time.perf_counter() - self._start
		# if self.stats.record(elapsed):
		#     self.stats.print_stats(label=f"{self.stats.name} [lap]")
		pass


