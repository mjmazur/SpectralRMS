import argparse
import cv2
import glob
import os
import shutil
import logging
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
        return output_file

    logger.info(f"Cutting {video_path} from {ss}s to {to}s -> {output_file}")
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


if __name__=="__main__":

	arg_parser = argparse.ArgumentParser(description="")
	arg_parser.add_argument('config', nargs=1, metavar='CONFIG_PATH', type=str, \
		help='Path to a config file which will be used instead of the default config file.')
	arg_parser.add_argument('--vid_path', nargs=1, metavar='VIDEO_PATH', type=str, \
		help='Path to the video directory.')
	arg_parser.add_argument('-d', '--detections', nargs=1, metavar='DETECT_PATH', type=str, \
		help='Path to a detections file which will be used.')
	# arg_parser.add_argument('--stash', nargs=1, metavar='STASH_PATH', type=str, \
	# 	help='Path to a stash file which will be used.')

	cml_args = arg_parser.parse_args()

	# Read the configuration file
	config_path = cml_args.config
	config = cr.loadConfigFromDirectory(config_path, os.path.abspath("."))
	print("Loaded configuration successfully!")

	if cml_args.vid_path is not None:
		vid_path = cml_args.vid_path[0]
		input_file = vid_path[0]
	else:
		print("Trying to get details from config file...")
		if hasattr(config, "video_dir"):
			vid_path = os.path.join(config.data_dir, config.video_dir)
			print("You have a 'new' config file. Congratulations!")
			print(config.data_dir)
		elif hasattr(config, "raw_video_dir"):
			vid_path = os.path.join(config.data_dir, config.raw_video_dir)
			print("You have a 'old' config file. Boo, boo, boo!")

	# Read the detections file
	if cml_args.detections is not None:
		print("Trying to get detectinfo file list...")
		ftp_dir = cml_args.detections[0]
		ftp_data = getData(ftp_dir)
	else:
		#ftp_dir = os.path.dirname(vid_path)
		ftp_dir = vid_path
		print("***")
		print(ftp_dir)
		ftp_data = getData(ftp_dir)
		print(vid_path)

	if "CAWES1" in vid_path:
		print("CAWES1 is in path")
		mkv_path = "/mnt/RMS_data/CAWES1/VideoFiles"
	elif "CAWES2" in vid_path:
		mkv_path = "/mnt/RMS_data/CAWES2/VideoFiles"

	video_list = glob.glob(os.path.join(mkv_path, "**/*.mkv"), recursive=True)

	video_list.sort()
	video_datetime_list = []

	for i in range(len(video_list)):
		video_name = os.path.basename(video_list[i])
		video_datetime_list.append(videoNameToDatetime(video_name))

	# FTPdetectinfo... fields: [0] FF filename, [1] Station ID, [2] Meteor #, [3] # of segments 
	# [4] fps, [5] hnr, [6] mle, [7] bin, [8] Pix/fm, [9] Rho, [10] Phi, [11] Segment data
	# [11][i][1] Frame #, [11][i][2] Column, [11][i][3] Row, [11][i][4] RA, [11][i][5] Dec
	# [11][i][6] Azim, [11][i][7] Elev, [11][i][8] Intensity, [11][i][9] Mag

	ff_filenames = []
	ff_datetime = []
	for i in range(len(ftp_data)):
		ff_filenames.append(ftp_data[i][0])
		ff_datetime.append((filenameToDatetime(ftp_data[i][0])).replace(tzinfo=timezone.utc))

	videos_to_archive = []
	cut_start = []
	cut_end = []
	cut_mp4_list = []

	for i in range(len(ff_datetime)):
		n, nt = nearestDatetime(video_datetime_list, ff_datetime[i])
		
		td = (ff_datetime[i]-nt).total_seconds()

		if td < 0:
			n -= 1

		if td > 0 and td < 30:
			videos_to_archive.append(video_list[n])

			start_frame = ftp_data[i][11][0][1]
			end_frame = ftp_data[i][11][-1][1]

			start_offset = start_frame/config.fps
			end_offset = end_frame/config.fps

			cut_frame_start = video_datetime_list[n] + timedelta(seconds=(start_offset-2))
			cut_frame_end = video_datetime_list[n] + timedelta(seconds=(end_offset+2))
			cut_start.append(cut_frame_start)
			cut_end.append(cut_frame_end)
			print(video_list[n])
			cut_mp4_list.append(cutoutFromMKV(video_list[n], [cut_start[-1], cut_end[-1]]))


