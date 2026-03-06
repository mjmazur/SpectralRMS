# Process CAMO and EMCCD events
# rsync and parse corr.txt files from /srv/meteor/klingon/events

import glob, os
import numpy as np
import pandas as pd
import ffmpeg
from datetime import datetime, timezone, timedelta

def nearestDatetime(dt_list, ntime):
	time_diff = np.abs([date - ntime for date in dt_list])
	nt = min(dt_list, key=lambda x:abs(x - ntime))
	n = time_diff.argmin(0)
	return n, nt

def MKVNameToDatetime(mkv_filepath):
	mkv_name = os.path.basename(mkv_filepath)
	name_split = mkv_name.split("_")
	camera_name = name_split[0]
	date = name_split[1]
	hms_time = name_split[2]
	millis_time = name_split[3]

	mkv_datetime = datetime.strptime(date + " " + hms_time, "%Y%m%d %H%M%S").replace(tzinfo=timezone.utc)

	if millis_time.isnumeric():
		print("Milliseconds are in the filename.")
		millis_time = int(millis_time)
		mkv_datetime = mkv_datetime + timedelta(milliseconds=millis_time)
	else:
		print("This file name does not have millis...")
		millis_time = 0

	return camera_name, mkv_datetime

def videoNameToDatetime(video_name):
	year = video_name.split("_")[1]
	time = video_name.split("_")[2]

	return (datetime.strptime(year + " " + time, "%Y%m%d %H%M%S")).replace(tzinfo=timezone.utc)

def cutoutFromMKV(video_path, file_time, time=None, crop_area=None, type="vid", output_path=None):
	
	dir_path, file_name = os.path.split(video_path)
	# cutout_file_name = os.path.basename(video_path).split('.')[0] + ".vid"

	# file_time = videoNameToDatetime(file_name)

	# print(file_time)

	if time is None:
		start_time = file_time
		start_frame = 0
		end_time = None
		end_frame = None
	else:
		start_time = time[0]
		start_frame = int((start_time - file_time).total_seconds()*mkv_fps)
		start_frame_time = int((start_time - file_time).total_seconds())
		if start_frame < 0:
			start_frame = 0
			start_frame_time = 0

		end_time = time[1]
		end_frame = int((end_time - file_time).total_seconds()*mkv_fps)
		end_frame_time = int((end_time - file_time).total_seconds())

		new_file_time = start_time.strftime("%Y%m%d_%H%M%S_%fA")


	if stationID == "CAWES1":
		camera_id = "02L"
	elif stationID == "CAWES2":
		camera_id = "02M"

	output_file = "ev_" + new_file_time + "_" + camera_id + ".mp4"
	mp4_path = os.path.join(out_path, camera_id, output_file)

	# print(os.path.join("/mnt/RMS_data/dump.vid", camera_id, output_file))
	if not os.path.isfile(mp4_path):
		video_in = ffmpeg.input(video_path, ss=start_frame_time, to=end_frame_time)
		video_out = ffmpeg.output(video_in, mp4_path)
		ffmpeg.run(video_out)
	else:
		print("Cutout already exists!")

	return mp4_path

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

if __name__ == "__main__":
	
	data_paths = ["/mnt/RMS_data/CAWES1/CapturedFiles", "/mnt/RMS_data/CAWES2/CapturedFiles"]
	mkv_path = ["/mnt/RMS_data/CAWES1/VideoFiles", "/mnt/RMS_data/CAWES2/VideoFiles"]
	event_path = "/srv/meteor/klingon/events"
	out_path = "/mnt/RMS_data/dump.vid/Test"

	mkv_length = 30
	mkv_fps = 25
	buffer = 2
	meteor_length = 3

	# Get list of dates to process from dates of data currently saved in data directory
	dates = []
	for data_path in data_paths:
		dlist = glob.glob(os.path.join(data_path, "*"))
		for j in range(len(dlist)):
			dates.append(os.path.basename(dlist[j]).split("_")[1])

	dates = list(set(dates))
	dates.sort()

	# List of videos found in the MKV path(s)
	video_list = glob.glob(os.path.join(mkv_path[0], "**/*.mkv"), recursive=True)
	video_list2 = glob.glob(os.path.join(mkv_path[1], "**/*.mkv"), recursive=True)

	video_list.sort()
	video_list2.sort()

	# Make a list of the datetimes of the videos
	video_datetime_list = []
	video_datetime_list2 = []

	for i in range(len(video_list)):
		video_name = os.path.basename(video_list[i])
		video_datetime_list.append(videoNameToDatetime(video_name))

	for i in range(len(video_list2)):
		video_name2 = os.path.basename(video_list2[i])
		video_datetime_list2.append(videoNameToDatetime(video_name2))



	# Make a list of the events listed on Colossid
	event_dirs = []
	for date in dates:
		event_dirs.append(os.path.join(event_path, date))

	# Make a list of the datetimes of all of the events, for the dates specified
	event_datetime_list = []
	for event_dir in event_dirs:
		ev_file_list = glob.glob(os.path.join(event_dir, "ev*02T.txt"))
		ev_file_list.sort()
		for ev_file in ev_file_list:
			ev_file_split = ev_file.split("_")
			year = ev_file_split[1][:4]
			month = ev_file_split[1][4:6]
			day = ev_file_split[1][6:]
			hour = ev_file_split[2][:2]
			minute = ev_file_split[2][2:4]
			second = ev_file_split[2][4:6]

			event_datetime_list.append(datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc))

	# Loop over the datetimes for the events, and make the cutouts
	videos_to_archive = []
	event_archive_times = []
	videos_to_archive2 = []
	event_archive_times2 = []
	cut_start = []
	cut_end = []
	cut_start2 = []
	cut_end2 = []
	cut_mp4_list = []

	for i in range(len(event_datetime_list)):
		try:
			n, nt = nearestDatetime(video_datetime_list, event_datetime_list[i])
			# print(n, nt)
		except:
			print("Didna work.")
		
		td = (event_datetime_list[i]-nt).total_seconds()

		if td < 0:
			n -= 1
		if td > 0 and td < 30:
			# print(td)
			videos_to_archive.append(video_list[n])
			event_archive_times.append(event_datetime_list[i])

	for i in range(len(event_datetime_list)):
		try:
			n2, nt2 = nearestDatetime(video_datetime_list2, event_datetime_list[i])
			# print(n, nt)
		except:
			print("Didna work.")
		
		td2 = (event_datetime_list[i]-nt2).total_seconds()

		if td2 < 0:
			n2 -= 1
		if td2 > 0 and td2 < 30:
			# print(td)
			videos_to_archive2.append(video_list2[n2])
			event_archive_times2.append(event_datetime_list[i])

	for i in range(len(videos_to_archive)):
		stationID, mkv_time = MKVNameToDatetime(videos_to_archive[i])

		time_diff = event_archive_times[i] - mkv_time

		if time_diff.total_seconds() > buffer:
			start_offset = time_diff.total_seconds() - buffer
		else:
			start_offset = 0

		if time_diff.total_seconds() < (mkv_length - (buffer + meteor_length)):
			end_offset = time_diff.total_seconds() + (buffer + meteor_length)
		else:
			end_offset = mkv_length

		cut_frame_start = event_archive_times[i] + timedelta(seconds=start_offset)
		cut_frame_end = event_archive_times[i] + timedelta(seconds= end_offset)

		cut_start.append(cut_frame_start)
		cut_end.append(cut_frame_end)
		cut_mp4_list.append(cutoutFromMKV(videos_to_archive[i], mkv_time, [cut_start[-1], cut_end[-1]]))

	for i in range(len(videos_to_archive2)):
		stationID, mkv_time = MKVNameToDatetime(videos_to_archive2[i])

		time_diff = event_archive_times2[i] - mkv_time

		if time_diff.total_seconds() > buffer:
			start_offset = time_diff.total_seconds() - buffer
		else:
			start_offset = 0

		if time_diff.total_seconds() < (mkv_length - (buffer + meteor_length)):
			end_offset = time_diff.total_seconds() + (buffer + meteor_length)
		else:
			end_offset = mkv_length

		cut_frame_start = event_archive_times2[i] + timedelta(seconds=start_offset)
		cut_frame_end = event_archive_times2[i] + timedelta(seconds= end_offset)

		cut_start2.append(cut_frame_start)
		cut_end2.append(cut_frame_end)
		print(cut_start2[-1], cut_end2[-1])
		try:
			cut_mp4_list.append(cutoutFromMKV(videos_to_archive2[i], mkv_time, [cut_start2[-1], cut_end2[-1]]))
		except:
			print("Couldn't make mp4...")

		# print(cut_frame_start, cut_frame_end)
	# print(cut_mp4_list)
	# print(cut_start)
	# print(video_archive_times[-1], videos_to_archive[-1])


	# remote_ip = "10.24.8."
	# for i in range(len(dates)):
	# 	# scp optical@colossid.localnet:/blah/blah/meteor/klingon/evcorr/date/corr.txt /path/to/save/date+camo+corr.txt
	# 	# scp optical@colossid.localnet:/blah/blah/meteor/klingon/evcorr/date/corr.txt /path/to/save/date+emccd+corr.txt

	# 	os.system()

	# Open and parse corr.txt files
	# ev_date = []
	# ev_time = []
	# ev_stations = []
	# ev_vel = []
	# ev_mass = []
	# ev_beg = []
	# ev_end = []

	corr_filenames = ["20250618_emccd_corr.txt"]

	# for i in len(corr_filenames):
	# 	with open(corr_filenames[i]) as f:
	# 		for line in f:
	# 			if line.startswith("+") or line.startswith("%"):
	# 				line_split = line.split(" ")
	# 				ev_date.append(line_split[1])
	# 				ev_time.append(line_split[2])
	# 				ev_stations.append([line_split[4],line_split[5],line_split[6],line_split[7]])
	# 				ev_vel.append(line_split[9])
	# 				ev_mass.append(line_split[10])
	# 				ev_beg.append(line_split[11])
	# 				ev_end.append(line_split[12])
	# 				# print(line.split(" "))

	# 	column_names = ["date", "time", "stations", "vel", "mass", "beg", "end"]
	# 	ev_df = pd.DataFrame(list(zip(ev_date, ev_time, ev_stations, ev_vel, ev_mass, ev_beg, ev_end)), columns=column_names)
	# 	print(ev_df)