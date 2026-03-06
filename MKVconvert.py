import argparse
import cv2
import imageio.v3
import glob, os, shutil
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timezone, timedelta

import RMS.ConfigReader as cr
from RMS.Formats.FFfits import write as writeFF
from RMS.Formats.FFfits import filenameToDatetimeStr
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.FFStruct import FFStruct
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo, findFTPdetectinfoFile, readFTPdetectinfo

from astropy.io import fits

def nearestDatetime(dt_list, ntime):
	time_diff = np.abs([date - ntime for date in dt_list])
	nt = min(dt_list, key=lambda x: abs(x - ntime))
	n = time_diff.argmin(0)

	return n, nt

def getData(ff_path):
	''' Gets the detections from an FTPdetectinfo file.

	Arguments: Takes the path to the FF files which should also contain the FTPdetectinfo file.

	Returns: A list containing the detection data.
	'''
	
	ftp_dir = findFTPdetectinfoFile(ff_path)
	# print("ftp:", ftp_dir)
	ftp_file_list = getFTPdetectinfoFileList(ff_path)
	# print("ftp list: ", ftp_file_list)

	event_data = []

	if isinstance(ftp_file_list, list):
		for i in range(len(ftp_file_list)):
			event_data += readFTPdetectinfo(*os.path.split(ftp_file_list[i]))

	# print("event:", event_data)
	# event_data += readFTPdetectinfo(*os.path.split(ftp_dir))

	ff_list = [entry[0] for entry in event_data]

	print('No. events: %s' % len(event_data))

	return event_data

def getFTPdetectinfoFileList(path):
    """ Finds the FTPdetectinfo files in directory if path is a directory, otherwise will return the path """

    if os.path.isfile(path):
        return path

    ftpdetectinfo_files = [filename for filename in sorted(os.listdir(path)) if 'FTPdetectinfo_' in filename]

    # Remove backup files from list
    filtered_ftpdetectinfo_files = []
    for filename in ftpdetectinfo_files:
        if validDefaultFTPdetectinfo(filename):
            filtered_ftpdetectinfo_files.append(filename)

    ftpdetectinfo_files = list(filtered_ftpdetectinfo_files)
    
    for i in range(len(ftpdetectinfo_files)):
	    ftpdetectinfo_files[i] = os.path.join(path, ftpdetectinfo_files[i])
    
    return ftpdetectinfo_files

    raise FileNotFoundError("FTPdetectinfo file not found")

def videoToVID(video_path, output_path=None):
	
	def writeFrame(f, data, time, exptime, seqnum):
		magic = 809789782
		seqlen = config.width * config.height * 1
		headlen = 116
		flags = 999
		seq = seqnum
		ts = int(time) # seconds since epoch
		tu = int(round((time - ts)*1000000)) # micros since last second
		num = 1 # station number
		wid = config.width
		ht = config.height
		depth = 8
		hxt = 0
		strid = 1
		res0 = 0
		expose = exptime
		res2 = 0
		desc = 'short description'

		filler = 0
		arr = data
		# arr = arr.astype(np.uint16)

		framenum = seq
		bkstp = -1 * arr.size * 1
		# bkstp = -4147200
		arr.tofile(f)

		f.seek(bkstp+0, 2)
		f.write(magic.to_bytes(4,byteorder="little"))
		f.seek(bkstp+4, 2)
		f.write(seqlen.to_bytes(4,byteorder="little"))
		f.seek(bkstp+8, 2)
		f.write(headlen.to_bytes(4,byteorder="little"))
		f.seek(bkstp+12, 2)
		f.write(flags.to_bytes(4,byteorder="little"))
		f.seek(bkstp+16, 2)
		f.write(seq.to_bytes(4,byteorder="little"))
		f.seek(bkstp+20, 2)
		f.write(ts.to_bytes(4,byteorder="little"))
		f.seek(bkstp+24, 2)
		f.write(tu.to_bytes(4,byteorder="little"))
		f.seek(bkstp+28, 2)
		f.write(num.to_bytes(2,byteorder="little"))
		f.seek(bkstp+30, 2)
		f.write(wid.to_bytes(2,byteorder="little"))
		f.seek(bkstp+32, 2)
		f.write(ht.to_bytes(2,byteorder="little"))
		f.seek(bkstp+34, 2)
		f.write(depth.to_bytes(2,byteorder="little"))
		f.seek(bkstp+36, 2)
		f.write(hxt.to_bytes(4,byteorder="little"))
		f.seek(bkstp+40, 2)
		f.write(strid.to_bytes(2,byteorder="little"))
		f.seek(bkstp+42, 2)
		f.write(res0.to_bytes(2,byteorder="little"))
		f.seek(bkstp+44, 2)
		f.write(expose.to_bytes(4,byteorder="little"))
		f.seek(bkstp+48, 2)
		f.write(res2.to_bytes(4,byteorder="little"))
		f.seek(bkstp+52, 2)
		f.write(desc.encode("ascii"))

		for i in range(64-len(desc)):
			f.write(filler.to_bytes(1,byteorder="little"))
		f.seek(0,2)

	dir_path, file_name = os.path.split(video_path)
	vid_file_name = os.path.basename(video_path).split('.')[0] + ".vid"

	if output_path is not None:
		vid_file_name = os.path.join(output_path, vid_file_name)

	# start_time = filenameToDatetime(file_name)
	start_time = (videoNameToDatetime(file_name)).replace(tzinfo=timezone.utc)
	frame_time = (start_time - datetime(1970,1,1, tzinfo=timezone.utc)).total_seconds()
	exposure_time = int(round((1/config.fps)*1000))

	with open(vid_file_name, "wb") as f:
		
		frame_index = 0

		cap = cv2.VideoCapture(video_path)
		frames = []

		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				# print("Can't receive frame")
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			writeFrame(f, gray, frame_time, exposure_time, frame_index)

			frame_index += 1
			frame_time += exposure_time/1000
		cap.release()


def videoToFF(video_path, output_path=None):

	cap = cv2.VideoCapture(video_path)
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Can't receive frame")
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frames.append(gray)
	cap.release()

	video = np.stack(frames, axis=0)

	maxpixel = video.max(axis=0)
	avepixel = video.mean(axis=0)
	stdpixel = video.std(axis=0)
	maxframe = video.argmax(axis=0)

	dir_path, file_name = os.path.split(video_path)

	ff = FFStruct()
	ff.ncols = config.width
	ff.nrows = config.height
	start_time = filenameToDatetimeStr(file_name)
	ff.starttime = start_time

	ff.array = np.stack([maxpixel, maxframe, avepixel, stdpixel], axis=0)

	# Write the FF to FITS
	if output_path is not None:
		if not os.path.exists(output_path):
			os.makedirs(output_path)
	else:
		output_path = os.path.join(dir_path, "FFfiles")
		if not os.path.exists(output_path):
			os.makedirs(output_path)

	output_file_name = file_name.split('.')[0]

	writeFF(ff, output_path, output_file_name[:-5]+"0000000")

	print("Wrote FF file to: ", os.path.join(output_path, output_file_name[:-5], "0000000.fits"))


def cutoutFromMKV(video_path, time=None, crop_area=None, type="vid", output_path=None):
	
	dir_path, file_name = os.path.split(video_path)
	# cutout_file_name = os.path.basename(video_path).split('.')[0] + ".vid"

	file_time = videoNameToDatetime(file_name)
	print(file_time)

	if time is None:
		start_time = file_time
		start_frame = 0
		end_time = None
		end_frame = None
	else:
		start_time = time[0]
		start_frame = int((start_time - file_time).total_seconds()*config.fps)
		start_frame_time = int((start_time - file_time).total_seconds())
		if start_frame < 0:
			start_frame = 0
			start_frame_time = 0

		end_time = time[1]
		end_frame = int((end_time - file_time).total_seconds()*config.fps)
		end_frame_time = int((end_time - file_time).total_seconds())

		new_file_time = start_time.strftime("%Y%m%d_%H%M%S_%fA")


	if config.stationID == "CAWES1":
		camera_id = "02L"
	elif config.stationID == "CAWES2":
		camera_id = "02M"

	output_file = "ev_" + new_file_time + "_" + camera_id + ".mp4"
	mp4_path = os.path.join("/mnt/RMS_data/dump.vid", camera_id, output_file)

	# print(os.path.join("/mnt/RMS_data/dump.vid", camera_id, output_file))
	if not os.path.isfile(mp4_path):
		video_in = ffmpeg.input(video_path, ss=start_frame_time, to=end_frame_time)
		video_out = ffmpeg.output(video_in, mp4_path)
		ffmpeg.run(video_out)
	else:
		print("Cutout already exists!")

	return mp4_path


	# frame_time = (start_time - datetime(1970,1,1, tzinfo=timezone.utc)).total_seconds()

	# cap = cv2.VideoCapture(video_path)
	# frames = []
	# while cap.isOpened():
	# 	ret, frame = cap.read()
	# 	if not ret:
	# 		print("Can't receive frame")
	# 		break
	# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
	# 	# Crop
	# 	gray = gray[888-20:888+20, 302-20:302+60]
	# 	frames.append(gray)

	# cap.release()

	# video = np.stack(frames, axis=0)

	# maxpixel = video.max(axis=0)
	# avepixel = video.mean(axis=0)
	# stdpixel = video.std(axis=0)
	# maxframe = video.argmax(axis=0)

	# plt.imshow(maxpixel)
	# plt.show()



def videoNameToDatetime(video_name):
	year = video_name.split("_")[1]
	time = video_name.split("_")[2]

	return (datetime.strptime(year + " " + time, "%Y%m%d %H%M%S")).replace(tzinfo=timezone.utc)


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

	# print(video_list)

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

	print(video_datetime_list)
	# print(ff_datetime[0])

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


	# if config.stationID == "CAWES1":
	# 	camera_id = "02L"
	# elif config.stationID == "CAWES2":
	# 	camera_id = "02M"

	# # cut_MKV_list = glob.glob(os.path.join("/mnt/RMS_data/dump.vid", camera_id, "*.mp4"))
	# # cut_MKV_list.sort()

	# for mp4 in cut_mp4_list:
	# 	videoToVID(mp4, os.path.join("/mnt/RMS_data/dump.vid", camera_id))


	# print(cut_mp4_list)
	
	# print(videos_to_archive)
	# 	# print(ftp_data[i][11][0][9])

	


	# for video in videos_to_archive:
	# 	if os.path.isfile(video):

	# 		print(os.path.exists(os.path.join("/mnt/RMS_data/dump.vid", os.path.basename(video).removesuffix(".mkv") + ".vid")))

	# 		if not os.path.exists(os.path.join("/mnt/RMS_data/dump.vid", os.path.basename(video).removesuffix(".mkv") + ".vid")):
	# 			videoToVID(video, output_path="/mnt/RMS_data/dump.vid")
	# 			print("Converted to VID...")
	# 		else:
	# 			print("File already exists...")



	# print(ff_filenames)

	# event_data = getData(ftp_dir)

	# # video_dir
	# print(os.path.join(config.data_dir, config.video_dir))

	# print(event_data)

	# if os.path.isdir(vid_path):
	# 	print("Directory")
	# 	video_list = glob.glob(os.path.join(vid_path, "*.mkv"))
	# 	for i in range(len(video_list)):
	# 		# videoToFF(video_list[i])
	# 		videoToVID(video_list[i])
	# elif os.path.isfile(vid_path):
	# 	# videoToFF(vid_path)
	# 	# videoToVID(vid_path)
	# 	cutoutFromMKV(vid_path)
	# else:
	# 	print("Somthing went wrong.")