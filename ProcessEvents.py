import argparse
import cv2
import glob
import os
import shutil
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from tqdm import tqdm

import ffmpeg
import numpy as np
import RMS.ConfigReader as cr
from RMS.Formats.FFfits import write as write_ff_fits
from RMS.Formats.FFfits import filenameToDatetimeStr as filename_to_datetime_str
from RMS.Formats.FFfile import filenameToDatetime as filename_to_datetime
from RMS.Formats.FFStruct import FFStruct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EventDetection:
	def __init__(self, dt: datetime, source_file: str, line: Optional[str] = None):
		self.dt = dt
		self.source_file = source_file
		self.line = line

def find_nearest_datetime(dt_list: List[datetime], target_dt: datetime) -> Tuple[int, datetime]:
	"""Finds the nearest datetime in a list to the target datetime."""
	if not dt_list:
		raise ValueError("The list of datetimes is empty.")
	
	diffs = [abs(dt - target_dt).total_seconds() for dt in dt_list]
	idx = np.argmin(diffs)
	return idx, dt_list[idx]

def video_to_datetime(video_name: str) -> datetime:
	"""Extracts the datetime from an MKV filename."""
	parts = video_name.split("_")
	if len(parts) < 3:
		raise ValueError(f"Invalid video filename format: {video_name}")
	date_str = parts[1]
	time_str = parts[2]
	dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S").replace(tzinfo=timezone.utc)
	# Handle potential milliseconds (e.g., STATION_YYYYMMDD_HHMMSS_mmm_video.mkv)
	if len(parts) > 3 and parts[3].isdigit():
		dt += timedelta(milliseconds=int(parts[3]))
	return dt

def parse_ev_filename(file_path: Path) -> Optional[EventDetection]:
	"""Parses the datetime from an ev_*.txt filename."""
	try:
		parts = file_path.name.split("_")
		if len(parts) < 3:
			return None
		date_str = parts[1]
		time_str = parts[2]
		dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S").replace(tzinfo=timezone.utc)
		return EventDetection(dt, str(file_path))
	except ValueError:
		return None

def parse_corr_file(file_path: Path) -> List[EventDetection]:
	"""Parses events from a corr.txt file."""
	events = []
	logger.info(f"Parsing corr file: {file_path}")
	try:
		with open(file_path, 'r') as f:
			for line in f:
				line = line.strip()
				if line.startswith("+") or line.startswith("%"):
					parts = line.split()
					if len(parts) < 3:
						continue
					date_str = parts[1]
					time_str = parts[2]
					try:
						# Expecting YYYYMMDD HH:MM:SS
						dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H:%M:%S").replace(tzinfo=timezone.utc)
						events.append(EventDetection(dt, str(file_path), line))
					except ValueError:
						continue
	except Exception as e:
		logger.error(f"Error reading {file_path}: {e}")
	return events

def video_to_vid(video_path: Path, config, output_path: Optional[Path] = None):
	"""Converts a video file to the RMS .vid format."""
	
	def write_frame(f, data, time, exptime, seqnum):
		magic = 809789782
		seqlen = config.width * config.height
		headlen = 116
		flags = 999
		ts = int(time)
		tu = int(round((time - ts) * 1000000))
		num = 1 # station number
		
		f.write(data.tobytes())
		
		bkstp = -1 * data.size
		f.seek(bkstp, 2)
		f.write(magic.to_bytes(4, byteorder="little"))
		f.write(seqlen.to_bytes(4, byteorder="little"))
		f.write(headlen.to_bytes(4, byteorder="little"))
		f.write(flags.to_bytes(4, byteorder="little"))
		f.write(seqnum.to_bytes(4, byteorder="little"))
		f.write(ts.to_bytes(4, byteorder="little"))
		f.write(tu.to_bytes(4, byteorder="little"))
		f.write(num.to_bytes(2, byteorder="little"))
		f.write(config.width.to_bytes(2, byteorder="little"))
		f.write(config.height.to_bytes(2, byteorder="little"))
		f.write((8).to_bytes(2, byteorder="little")) # depth
		f.write((0).to_bytes(4, byteorder="little")) # hxt
		f.write((1).to_bytes(2, byteorder="little")) # strid
		f.write((0).to_bytes(2, byteorder="little")) # res0
		f.write(exptime.to_bytes(4, byteorder="little"))
		f.write((0).to_bytes(4, byteorder="little")) # res2
		
		desc = b'event cutout'
		f.write(desc.ljust(64, b'\0'))
		f.seek(0, 2)

	vid_path = (output_path or video_path.parent) / (video_path.stem + ".vid")
	try:
		start_time = video_to_datetime(video_path.name)
	except ValueError:
		start_time = datetime.now(timezone.utc)
	
	frame_time = start_time.timestamp()
	exposure_time = int(round((1 / config.fps) * 1000))

	logger.info(f"Converting {video_path} to {vid_path}")
	with open(vid_path, "wb") as f:
		cap = cv2.VideoCapture(str(video_path))
		frame_index = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			write_frame(f, gray, frame_time, exposure_time, frame_index)
			frame_index += 1
			frame_time += exposure_time / 1000
		cap.release()

def cutout_from_mkv(video_path: Path, start_time: datetime, end_time: datetime, camera_id: str, output_dir: Path) -> Optional[Path]:
	"""Cuts out a portion of an MKV file using ffmpeg."""
	try:
		file_time = video_to_datetime(video_path.name)
	except ValueError as e:
		logger.error(f"Could not parse datetime from {video_path.name}: {e}")
		return None
	
	ss = max(0, (start_time - file_time).total_seconds())
	to = (end_time - file_time).total_seconds()
	
	new_file_time_str = start_time.strftime("%Y%m%d_%H%M%S_%f")[:-3] + "A"
	output_file = output_dir / f"ev_{new_file_time_str}_{camera_id}.mp4"
	
	output_file.parent.mkdir(parents=True, exist_ok=True)

	if output_file.exists():
		logger.info(f"Cutout already exists: {output_file}")
		return output_file

	logger.info(f"Cutting {video_path} from {ss:.2f}s to {to:.2f}s -> {output_file}")
	try:
		(
			ffmpeg
			.input(str(video_path), ss=ss, to=to)
			.output(str(output_file))
			.run(quiet=True, overwrite_output=True)
		)
		return output_file
	except ffmpeg.Error as e:
		logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
		return None

def sync_to_remote(local_dir: Path, remote_path: str):
	"""Syncs the local directory to a remote path using rsync."""
	logger.info(f"Syncing {local_dir} to {remote_path}...")
	try:
		# -a: archive mode, -v: verbose, -z: compress, --progress: show progress
		cmd = ["rsync", "-avz", str(local_dir) + "/", remote_path]
		result = subprocess.run(cmd, capture_output=True, text=True)
		if result.returncode == 0:
			logger.info("Rsync completed successfully.")
		else:
			logger.error(f"Rsync failed with return code {result.returncode}: {result.stderr}")
	except Exception as e:
		logger.error(f"Error during rsync: {e}")

def main():
	arg_parser = argparse.ArgumentParser(description="Process meteor events and create MKV cutouts.")
	arg_parser.add_argument('config', help='Path to the RMS config file.')
	arg_parser.add_argument('-e', '--events', help='Path to events directory, ev_*.txt, or corr.txt file.')
	arg_parser.add_argument('--mkv-paths', nargs='+', help='List of directories containing MKV files.')
	arg_parser.add_argument('--output-dir', help='Directory to save cutouts (default: /mnt/RMS_data/dump.vid/Test).')
	arg_parser.add_argument('--buffer', type=float, default=2.0, help='Buffer before/after event in seconds (default: 2.0).')
	arg_parser.add_argument('--length', type=float, default=3.0, help='Assumed meteor duration in seconds (default: 3.0).')
	arg_parser.add_argument('--camera-id', help='Override camera ID. If not provided, will try to infer.')
	arg_parser.add_argument('--convert-vid', action='store_true', help='Convert the resulting MP4 cutouts to .vid format.')
	arg_parser.add_argument('--rsync-path', help='Remote path to rsync the cutouts to (e.g., user@host:/path/to/dest).')

	args = arg_parser.parse_args()

	# Load configuration
	print(args.config)
	config = cr.loadConfigFromDirectory([args.config], os.path.abspath("."))
	logger.info("Loaded configuration successfully!")
	print(f"  Station ID : {getattr(config, 'stationID', 'N/A')}")
	print(f"  Resolution : {getattr(config, 'width', '?')}x{getattr(config, 'height', '?')} @ {getattr(config, 'fps', '?')} fps")

	# Resolve event path
	event_root = Path(args.events) if args.events else Path("/srv/meteor/klingon/events")
	all_events = []

	if event_root.is_file() and event_root.suffix == ".txt":
		if "corr" in event_root.name.lower():
			all_events.extend(parse_corr_file(event_root))
		else:
			ev = parse_ev_filename(event_root)
			if ev: all_events.append(ev)
	elif event_root.is_dir():
		all_txt = sorted([f for f in event_root.rglob("*") if f.is_file() and f.suffix == ".txt"])
		for f in tqdm(all_txt, desc="Scanning event files", unit="file"):
			if "corr.txt" in f.name.lower():
				all_events.extend(parse_corr_file(f))
			elif f.name.startswith("ev"):
				ev = parse_ev_filename(f)
				if ev: all_events.append(ev)

	if not all_events:
		logger.warning(f"No events found in {event_root}")
		return

	logger.info(f"Unique events to process: {len(all_events)}")

	# Resolve MKV paths
	mkv_roots = [Path(p) for p in args.mkv_paths] if args.mkv_paths else [
		Path("/mnt/RMS_data/CAWES1/VideoFiles"),
		Path("/mnt/RMS_data/CAWES2/VideoFiles")
	]

	def _index_mkv_root(root: Path) -> Optional[Dict]:
		if not root.exists():
			logger.warning(f"MKV root does not exist: {root}")
			return None
		files = sorted(list(root.rglob("*.mkv")))
		dts, valid_files = [], []
		for f in tqdm(files, desc=f"Indexing {root.name}", unit="file", leave=False):
			try:
				dts.append(video_to_datetime(f.name))
				valid_files.append(f)
			except ValueError:
				continue
		return {'root': root, 'files': valid_files, 'datetimes': dts} if valid_files else None

	mkv_indices = []
	with ThreadPoolExecutor(max_workers=len(mkv_roots) or 1) as pool:
		futures = {pool.submit(_index_mkv_root, root): root for root in mkv_roots}
		for fut in tqdm(as_completed(futures), total=len(futures), desc="Indexing MKV roots", unit="root"):
			result = fut.result()
			if result:
				mkv_indices.append(result)
				logger.info(f"Indexed {len(result['files'])} MKV files from {result['root']}")

	if not mkv_indices:
		logger.error("No valid MKV files found in any of the specified paths.")
		return

	output_dir_base = Path(args.output_dir) if args.output_dir else Path("/mnt/RMS_data/dump.vid/Test")

	processed_count = 0
	count_lock = threading.Lock()

	def _process_event(event: EventDetection) -> int:
		"""Processes a single event across all MKV indices. Returns number of cutouts created."""
		local_count = 0
		for index in mkv_indices:
			try:
				idx, mkv_dt = find_nearest_datetime(index['datetimes'], event.dt)
				if event.dt < mkv_dt and idx > 0:
					idx -= 1
					mkv_dt = index['datetimes'][idx]

				m_video = index['files'][idx]
				time_diff = (event.dt - mkv_dt).total_seconds()

				if 0 <= time_diff < 120:
					start_cut = max(mkv_dt, event.dt - timedelta(seconds=args.buffer))
					end_cut = event.dt + timedelta(seconds=args.buffer + args.length)

					current_camera_id = args.camera_id
					if not current_camera_id:
						current_camera_id = "02L" if "CAWES1" in str(index['root']) else "02M" if "CAWES2" in str(index['root']) else "UNK"

					out_path = output_dir_base / current_camera_id
					mp4_path = cutout_from_mkv(m_video, start_cut, end_cut, current_camera_id, out_path)

					if mp4_path:
						local_count += 1
						if args.convert_vid:
							video_to_vid(mp4_path, config, out_path)
			except Exception as e:
				logger.error(f"Error matching event {event.dt} in {index['root']}: {e}")
		return local_count

	max_workers = min(8, len(all_events))
	with ThreadPoolExecutor(max_workers=max_workers) as pool:
		futures = {pool.submit(_process_event, ev): ev for ev in all_events}
		with tqdm(total=len(all_events), desc="Processing events", unit="event") as pbar:
			for fut in as_completed(futures):
				count = fut.result()
				with count_lock:
					processed_count += count
				pbar.update(1)

	logger.info(f"Processing complete. {processed_count} cutouts created.")

	# Sync to remote if requested
	if args.rsync_path:
		sync_to_remote(output_dir_base, args.rsync_path)

if __name__ == "__main__":
	main()