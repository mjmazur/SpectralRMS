import argparse
import cv2
import glob
import os
import shutil
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import ffmpeg
import numpy as np
import RMS.ConfigReader as cr
from RMS.Formats.FFfits import write as write_ff_fits
from RMS.Formats.FFfits import filenameToDatetimeStr as filename_to_datetime_str
from RMS.Formats.FFfile import filenameToDatetime as filename_to_datetime
from RMS.Formats.FFStruct import FFStruct
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo, findFTPdetectinfoFile, readFTPdetectinfo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeteorDetection:
	def __init__(self, ff_filename: str, station_id: str, meteor_num: int, fps: float, segments: List):
		self.ff_filename = ff_filename
		self.station_id = station_id
		self.meteor_num = meteor_num
		self.fps = fps
		self.segments = segments
		self.dt = filename_to_datetime(ff_filename).replace(tzinfo=timezone.utc)
		
		# Calculate start and end offsets
		if segments:
			# segments format: [Frame#, Col, Row, RA, Dec, Azim, Elev, Inten, Mag]
			self.start_frame = segments[0][0]
			self.end_frame = segments[-1][0]
			self.start_offset = self.start_frame / self.fps
			self.end_offset = self.end_frame / self.fps
		else:
			self.start_frame = self.end_frame = 0
			self.start_offset = self.end_offset = 0

def find_nearest_datetime(dt_list: List[datetime], target_dt: datetime) -> Tuple[int, datetime]:
	"""Finds the nearest datetime in a list to the target datetime."""
	if not dt_list:
		raise ValueError("The list of datetimes is empty.")
	
	diffs = [abs(dt - target_dt).total_seconds() for dt in dt_list]
	idx = np.argmin(diffs)
	return idx, dt_list[idx]

def get_ftp_detectinfo_files(path: Path) -> List[Path]:
	"""Finds the FTPdetectinfo files in directory if path is a directory, otherwise returns the path."""
	if path.is_file():
		return [path]

	if not path.is_dir():
		logger.error(f"Path {path} is not a valid file or directory.")
		return []

	ftp_files = []
	for f in sorted(path.iterdir()):
		if f.is_file() and 'FTPdetectinfo_' in f.name and validDefaultFTPdetectinfo(f.name):
			ftp_files.append(f)
	
	return ftp_files

def read_detections_from_dir(ff_path_str: str, config) -> List[MeteorDetection]:
	"""Gets the detections from FTPdetectinfo files in the given directory."""
	ff_path = Path(ff_path_str)
	ftp_files = get_ftp_detectinfo_files(ff_path)

	detections = []
	for ftp_file in ftp_files:
		logger.info(f"Reading detection file: {ftp_file}")
		# readFTPdetectinfo returns a list of detections
		# Each detection is a list: [ff_name, station_id, meteor_num, ..., segments]
		file_data = readFTPdetectinfo(str(ftp_file.parent), ftp_file.name)
		for entry in file_data:
			det = MeteorDetection(
				ff_filename=entry[0],
				station_id=entry[1],
				meteor_num=entry[2],
				fps=entry[4] if entry[4] > 0 else config.fps,
				segments=entry[11]
			)
			detections.append(det)

	logger.info(f"Total detections found: {len(detections)}")
	return detections

def video_to_datetime(video_name: str) -> datetime:
	"""Extracts the datetime from an MKV filename."""
	# Example filename: CANUCK_20250321_055952_video.mkv
	parts = video_name.split("_")
	if len(parts) < 3:
		raise ValueError(f"Invalid video filename format: {video_name}")
	date_str = parts[1]
	time_str = parts[2]
	return datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S").replace(tzinfo=timezone.utc)

def video_to_vid(video_path: Path, config, output_path: Optional[Path] = None):
	"""Converts a video file to the RMS .vid format."""
	
	def write_frame(f, data, time, exptime, seqnum):
		magic = 809789782
		seqlen = config.width * config.height
		headlen = 116
		flags = 999
		ts = int(time)
		tu = int(round((time - ts) * 1000000))
		num = 1  # station number
		
		# Write image data
		data.tofile(f)
		
		# Write header at the end of the frame (RMS format)
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
		f.write((8).to_bytes(2, byteorder="little"))  # depth
		f.write((0).to_bytes(4, byteorder="little"))  # hxt
		f.write((1).to_bytes(2, byteorder="little"))  # strid
		f.write((0).to_bytes(2, byteorder="little"))  # res0
		f.write(exptime.to_bytes(4, byteorder="little"))
		f.write((0).to_bytes(4, byteorder="little"))  # res2
		
		desc = b'short description'
		f.write(desc.ljust(64, b'\0'))
		f.seek(0, 2)

	vid_path = (output_path or video_path.parent) / (video_path.stem + ".vid")
	start_time = video_to_datetime(video_path.name)
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

def video_to_ff_fits(video_path: Path, config, output_path: Optional[Path] = None):
	"""Generates an FF FITS file from a video."""
	cap = cv2.VideoCapture(str(video_path))
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
	cap.release()

	if not frames:
		logger.warning(f"No frames found in {video_path}")
		return

	video = np.stack(frames, axis=0)
	ff = FFStruct()
	ff.ncols = config.width
	ff.nrows = config.height
	ff.starttime = filename_to_datetime_str(video_path.name)
	ff.array = np.stack([
		video.max(axis=0),
		video.argmax(axis=0),
		video.mean(axis=0),
		video.std(axis=0)
	], axis=0)

	out_dir = output_path or (video_path.parent / "FFfiles")
	out_dir.mkdir(parents=True, exist_ok=True)
	
	write_ff_fits(ff, str(out_dir), video_path.stem[:-5] + "0000000")
	logger.info(f"Wrote FF file to {out_dir}")

def cutout_from_mkv(video_path: Path, start_time: datetime, end_time: datetime, camera_id: str, output_dir: Path) -> Optional[Path]:
	"""Cuts out a portion of an MKV file using ffmpeg."""
	file_time = video_to_datetime(video_path.name)
	
	ss = max(0, (start_time - file_time).total_seconds())
	to = (end_time - file_time).total_seconds()
	
	new_file_time_str = start_time.strftime("%Y%m%d_%H%M%S_%f")[:-3] + "A"
	output_file = output_dir / f"ev_{new_file_time_str}_{camera_id}.mp4"
	
	output_file.parent.mkdir(parents=True, exist_ok=True)

	if output_file.exists():
		logger.info(f"Cutout already exists: {output_file}")
		print(f"Cutout already exists: {output_file}")
		return output_file

	logger.info(f"Cutting {video_path} from {ss:.2f}s to {to:.2f}s -> {output_file}")
	print(f"Cutting {video_path} from {ss:.2f}s to {to:.2f}s -> {output_file}")
	try:
		(
			ffmpeg
			.input(str(video_path), ss=ss, to=to)
			.output(str(output_file))
			.run(quiet=True, overwrite_output=True)
		)
		print(f"Successfully created cutout: {output_file}")
		return output_file
	except ffmpeg.Error as e:
		logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
		print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
		return None

def sync_to_remote(local_dir: Path, remote_path: str):
	"""Syncs the local directory to a remote path using rsync."""
	logger.info(f"Syncing {local_dir} to {remote_path}...")
	try:
		# -a: archive mode, -v: verbose, -z: compress, --progress: show progress
		# We use shlex.split or just a list for safety
		cmd = ["rsync", "-avz", str(local_dir) + "/", remote_path]
		result = subprocess.run(cmd, capture_output=True, text=True)
		if result.returncode == 0:
			logger.info("Rsync completed successfully.")
		else:
			logger.error(f"Rsync failed with return code {result.returncode}: {result.stderr}")
	except Exception as e:
		logger.error(f"Error during rsync: {e}")

def main():
	arg_parser = argparse.ArgumentParser(description="Convert MKV meteor detections to cutout videos.")
	arg_parser.add_argument('config', help='Path to the RMS config file or directory.')
	arg_parser.add_argument('-d', '--detections', help='Path to a detections file or directory containing FTPdetectinfo files.')
	arg_parser.add_argument('--vid-path', help='Path to the directory containing MKV files.')
	arg_parser.add_argument('--output-dir', help='Directory to save cutouts (default: /mnt/RMS_data/dump.vid).')
	arg_parser.add_argument('--camera-id', help='Camera ID for filename (e.g., 02L). If not provided, will try to infer from station ID.')
	arg_parser.add_argument('--convert-vid', action='store_true', help='Convert the resulting MP4 cutouts to .vid format.')
	arg_parser.add_argument('--rsync-path', help='Remote path to rsync the cutouts to (e.g., user@host:/path/to/dest).')

	args = arg_parser.parse_args()

	# Load configuration
	config = cr.loadConfigFromDirectory([args.config], os.path.abspath("."))
	logger.info("Loaded configuration successfully!")

	# Resolve detection path
	det_path = Path(args.detections) if args.detections else Path(config.data_dir) / (config.video_dir if hasattr(config, "video_dir") else config.raw_video_dir)
	detections = read_detections_from_dir(str(det_path), config)

	# Resolve MKV path
	if args.vid_path:
		mkv_root = Path(args.vid_path)
	else:
		# Use some heuristic or default based on det_path
		if "CAWES1" in str(det_path):
			mkv_root = Path("/mnt/RMS_data/CAWES1/VideoFiles")
		elif "CAWES2" in str(det_path):
			mkv_root = Path("/mnt/RMS_data/CAWES2/VideoFiles")
		else:
			mkv_root = det_path # Fallback

	logger.info(f"Searching for MKV files in {mkv_root}...")
	mkv_files = sorted(list(mkv_root.rglob("*.mkv")))
	if not mkv_files:
		logger.error(f"No MKV files found in {mkv_root}")
		return

	mkv_datetimes = []
	valid_mkv_files = []
	for f in mkv_files:
		try:
			mkv_datetimes.append(video_to_datetime(f.name))
			valid_mkv_files.append(f)
		except ValueError:
			continue
	
	mkv_files = valid_mkv_files

	# Handle Camera ID
	camera_id = args.camera_id
	if not camera_id:
		if config.stationID == "CAWES1":
			camera_id = "02L"
		elif config.stationID == "CAWES2":
			camera_id = "02M"
		else:
			camera_id = config.stationID # Fallback to station ID

	# Resolve Output Directory
	output_dir_base = Path(args.output_dir) if args.output_dir else Path("/mnt/RMS_data/dump.vid")
	output_dir = output_dir_base / camera_id

	# Matching and cutting loop
	processed_count = 0
	for det in detections:
		try:
			# Find the nearest MKV
			idx, mkv_dt = find_nearest_datetime(mkv_datetimes, det.dt)
			
			# If the detection started before the MKV, check the previous one
			if det.dt < mkv_dt and idx > 0:
				idx -= 1
				mkv_dt = mkv_datetimes[idx]
			
			m_video = mkv_files[idx]
			
			# Check if detection is likely within this video (typically 30s-60s long)
			time_diff = (det.dt - mkv_dt).total_seconds()
			if 0 <= time_diff < 120: # Allow up to 2m for longer MKVs if any
				# Calculate cutout times with 2-second buffer
				start_cut = mkv_dt + timedelta(seconds=(det.start_offset - 2))
				end_cut = mkv_dt + timedelta(seconds=(det.end_offset + 2))
				
				# Ensure we don't start before the video beginning
				start_cut = max(start_cut, mkv_dt)
				
				mp4_path = cutout_from_mkv(m_video, start_cut, end_cut, camera_id, output_dir)
				
				if mp4_path and args.convert_vid:
					video_to_vid(mp4_path, config, output_dir)
				
				if mp4_path:
					processed_count += 1
			else:
				logger.warning(f"No matching MKV found for detection at {det.dt} (nearest was {mkv_dt}, diff {time_diff}s)")

		except Exception as e:
			logger.error(f"Error processing detection {det.ff_filename}: {e}", exc_info=True)

	logger.info(f"Processing complete. {processed_count} cutouts created in {output_dir}")

	# Sync to remote if requested
	if args.rsync_path:
		sync_to_remote(output_dir, args.rsync_path)

if __name__ == "__main__":
	main()
