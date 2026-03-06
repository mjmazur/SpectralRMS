import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import argparse
import json
import os, glob, time, math, pprint
import datetime

import RMS.ConfigReader as cr
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.AST import AstPlate

from RMS.Astrometry.Conversions import datetime2JD
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Formats.Platepar import Platepar
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo, findFTPdetectinfoFile, readFTPdetectinfo

from LCAM.Core.Stash import retrieve as readStash
from LCAM.Core.Stash import listStashes
from LCAM.Core.Platepar import getPlateparAzAltCorners
from LCAM.Core.SphericalPolygonCheck import sphericalPolygonCheck
from LCAM.Core.SkyfieldTLESatellite import SkyfieldTLESatellite

from skyfield.api import load, wgs84, EarthSatellite, utc

# from scipy.interpolate import spline


def makeContactSheet(rms_detections, lcam_matches, vis_stash=None, sat_tles=None, output_dir=".", detections=False, video=False):
	"""
	"""
	def plotBoxes():
		""" Function to plot boxes on FF stacks """		
		dX = x[1] - x[0]
		dY = y[1] - y[0]

		pa = math.degrees(math.atan2(dY,dX))
		width = math.sqrt(dX*dX + dY*dY)

		midpt = [1/2*dX + x[0], 1/2*dY + y[0]]

		rec_tle = patches.Rectangle((-1/2*width, -1/2*box_width), width, box_width, \
		transform=mpl.transforms.Affine2D().rotate_deg_around(*(0,0),pa) \
		+ mpl.transforms.Affine2D().translate(midpt[0], midpt[1]) + ax.transData, \
		fill=False, edgecolor=box_color, linestyle='solid')

		# if arrow == True:
		# 	arrow_length = math.sqrt(dX*dX + dY*dY)
		# 	del_x = dX / arrow_length * 13
		# 	del_y = dY / arrow_length * 13
		# 	plt.arrow(x[1], y[1], del_x, del_y, color=box_color, head_width=10)

		# rec_visible.set_clip_box(clip_box)
		ax.add_patch(rec_tle)
		# if (on_image[0] == True and on_image[1] == True) or (on_image[0] == False and on_image[1] == True) or correlation == True:
		# 	plt.text(x[1]+10, y[1]+7, name, fontsize=8)

	def plotLines():
		""" Function to plot boxes on FF stacks """		
		dX = x[-1] - x[0]
		dY = y[-1] - y[0]

		pa = math.degrees(math.atan2(dY,dX))
		width = math.sqrt(dX*dX + dY*dY)

		midpt = [1/2*dX + x[0], 1/2*dY + y[0]]

		# rec_tle = patches.Rectangle((-1/2*width, -1/2*box_width), width, box_width, \
		# transform=mpl.transforms.Affine2D().rotate_deg_around(*(0,0),pa) \
		# + mpl.transforms.Affine2D().translate(midpt[0], midpt[1]) + ax.transData, \
		# fill=False, edgecolor=box_color, linestyle='solid')

		x_notvis = []
		y_notvis = []
		x_vis = []
		y_vis = []

		for s in range(len(sunlit)):
			if sunlit[s]:
				x_vis.append(x[s])
				y_vis.append(y[s])
			else:
				x_notvis.append(x[s])
				y_notvis.append(y[s])

		plt.plot(x_vis,y_vis, ls=":", color=box_color)
		plt.scatter(x_notvis,y_notvis, color="pink", s=3)

		if arrow == True:
			arrow_length = math.sqrt(dX*dX + dY*dY)
			del_x = dX / arrow_length * 1
			del_y = dY / arrow_length * 1
			plt.arrow(x[-1], y[-1], del_x, del_y, color=box_color, head_width=10)

		# rec_visible.set_clip_box(clip_box)
		# ax.add_patch(rec_tle)
		if (on_image[0] == True and on_image[-1] == True) or (on_image[0] == False and on_image[-1] == True) or correlation == True:
			plt.text(x[-1]+10, y[-1]+7, name, fontsize=8)
		else:
			for oi in range(len(on_image)):
				if on_image[-(oi+1)]:
					try:
						plt.text(x[-(oi+5)]+10, y[-(oi+5)]+7, name, fontsize=8)
					except:
						pass
					break
	

	# Start main part of makeContactSheet

	location = wgs84.latlon(config.latitude, config.longitude)
	eph = load("de421.bsp")

	plates = getPlatePar(ff_path)
	plate = Platepar()

	n_cols = 3
	n_rows = 3

	if len(rms_detections) != 0:
		detection_list = []

	for i in range(len(rms_detections)):
		if isinstance(ff_path, str):
			if rms_detections[i][3] > 10:
				detection_list.append(os.path.join(ff_path, rms_detections[i][0]))
		else:
			if event_data[i][3] > 10:
				detection_list.append(os.path.join(ff_path[0],rms_detections[i][0]))

	if video == True:
		ff_list = ff_path_list
	else:
		ff_list = list(dict.fromkeys(detection_list))

	ff_list.sort()

	chunk_size = 9
	chunks = [ff_list[i:i+chunk_size] for i in range(0, len(ff_list), chunk_size)]

	# if detections is not False:
	# 	detections = np.asarray(detections)

	if lcam_matches is not None:
		matches_known = [track for track in lcam_matches if track.satellite.getName() != "UNKNOWN"]
		matches_unknown = [track for track in lcam_matches if track.satellite.getName() == "UNKNOWN"]

	# ff_path = os.path.dirname(ff_list[0])
	
	contact_sheet_dir = os.path.join(output_dir, "ContactSheets_" + time.strftime("%Y%m%d_%H%M%S")) 

	print("Contact sheet: ", contact_sheet_dir)

	for i in range(len(chunks)):
		plt.figure(figsize=(config.width/200 * n_cols, config.height/200 * n_rows), layout="constrained")

		for j in range(len(chunks[i])):
			ff_file = chunks[i][j]

			if lcam_matches is not None:
				if video == False:
					tmp_all = [track for track in lcam_matches if track.detection.ff_name == os.path.split(ff_file)[1]]
					tmp_known = [track for track in matches_known if track.detection.ff_name == os.path.split(ff_file)[1]]
					tmp_unknown = [track for track in matches_unknown if track.detection.ff_name == os.path.split(ff_file)[1]]
				else:
					for track in lcam_matches:
						lcam_match_time = (filenameToDatetime(track.detection.ff_name) - datetime.datetime(1970,1,1)).total_seconds()
						ff_file_time = (filenameToDatetime(os.path.split(ff_file)[1]) - datetime.datetime(1970,1,1)).total_seconds()
						diff = lcam_match_time - ff_file_time
					
						if diff <= 27.999 and diff >= 0:
							track.detection.ff_name = os.path.split(ff_file)[1]

					tmp_all = [track for track in lcam_matches if track.detection.ff_name == os.path.split(ff_file)[1]]
					tmp_known = [track for track in matches_known if track.detection.ff_name == os.path.split(ff_file)[1]]
					tmp_unknown = [track for track in matches_unknown if track.detection.ff_name == os.path.split(ff_file)[1]]


			if os.path.exists(ff_file):
				ff_array = readFF(os.path.dirname(ff_file), os.path.basename(ff_file))
				try:
					plate.loadFromDict(plates[ff_file], use_flat=config.use_flat)
				except:
					plate.read("/home/lcam/source/Stations/CAOSA1/platepar_cmn2010.cal", use_flat=config.use_flat)

				time_beg = filenameToDatetime(os.path.basename(ff_file))
				time_beg = time_beg.replace(tzinfo=utc)

				if video == False:
					time_end = time_beg + datetime.timedelta(seconds=10)
				else:
					time_end = time_beg + datetime.timedelta(seconds=27.999)

				time_range = [time_beg, time_end]

				if cml_args.show == True or output_dir is not None:
					ax = plt.subplot(n_rows, n_cols, j+1)
					ax.set(xlim=(0,1920), ylim=(0,1080))
					ax.tick_params(axis='y', direction='in', pad=-18)
					ax.imshow(ff_array.maxpixel, cmap='gray_r', vmin=50, vmax=150)

					#####################################################
					# Plot positions of visible satellites from TLE set
					#####################################################
					if tle_path is not None:
						for k in range(len(sat_tles)):
							name = sat_tles[k].getTLE()["TLE_LINE0"]
							tle1 = sat_tles[k].getTLE()["TLE_LINE1"]
							tle2 = sat_tles[k].getTLE()["TLE_LINE2"]
							box_color = "blue"
							box_width = 10
							correlation = False
							arrow = True

							try:
								x, y, height, on_image, sunlit = getSatelliteCoords(location, time_range, tle1, tle2, eph, platepar=plate, sat_name=name)
							except:
								print("Failed")

							if (on_image[0] == True or on_image[1] == True) and (sunlit[0] == True or sunlit[1] == True): # and height < 15000:
								plotBoxes()

					# ############################################################################
					# # Plot positions of visible satellites from 'VisibleSatellites' stash file'
					# ############################################################################
					# for n in vis_stash.getVisibleSatellites(time_beg, time_end):
					# 	name = n.getTLE()["TLE_LINE0"]
					# 	tle1 = n.getTLE()["TLE_LINE1"]
					# 	tle2 = n.getTLE()["TLE_LINE2"]
					# 	box_color = "orange"
					# 	box_width = 50
					# 	correlation = False
					# 	arrow = True
		
					# 	try:
					# 		x, y, height, on_image, sunlit = getSatelliteCoords(location, time_range, tle1, tle2, eph, platepar=plate, sat_name=name)
					# 	except:
					# 		print("Failed!")

					# 	# if (on_image[0] == True or on_image[1] == True) and (sunlit[0] == True or sunlit[1] == True) and height.km[0] < 10000:
					# 	if (on_image[0] == True or on_image[1] == True) and height.km[0] < 10000:
					# 		if sunlit[0] == False and sunlit[1] == False:
					# 			box_color = "blue"
					# 			plotBoxes()
					# 		else:
					# 			plotBoxes()

					############################################################################
					# Plot positions of visible satellites from 'VisibleSatellites' stash file'
					############################################################################
					if vis_stash is not None:
						for n in vis_stash.getVisibleSatellites(time_beg, time_end):
							name = n.getTLE()["TLE_LINE0"]
							tle1 = n.getTLE()["TLE_LINE1"]
							tle2 = n.getTLE()["TLE_LINE2"]
							box_color = "orange"
							box_width = 50
							correlation = False
							arrow = True
			
							try:
								x, y, height, on_image, sunlit = getCurvedSatelliteCoords(location, time_range, tle1, tle2, eph, platepar=plate, sat_name=name)
								# print("Curve: ", x, y)
							except:
								print("Failed!")

							# # if (on_image[0] == True or on_image[1] == True) and (sunlit[0] == True or sunlit[1] == True) and height.km[0] < 10000:
							# if (on_image[0] == True or on_image[1] == True): # and height.km[0] < 10000:
							if any(on == True for on in on_image) and any(h < 10000 for h in height.km):	
								# if any(sun == True for sun in sunlit):
								# 	plotLines()
								# else:
								# 	box_color = "blue"
								# 	plotLines()
								plotLines()

								# if sunlit[0] == False and sunlit[1] == False:
								# 	box_color = "blue"
								# 	plotBoxes()
								# else:
								# 	plotBoxes()


						######################################################################
						# Plot positions of correlations from 'VisibleSatellites' stash file
						######################################################################
						for k in range(len(tmp_known)):
							# print(tmp_known[k.X])
							if tmp_known[k].X is not None:
								box_color = "cyan"
								box_width = 40
								name = tmp_known[k].satellite.getName()
								correlation = True
								arrow = False

								x = [tmp_known[k].X[0], tmp_known[k].X[1]]
								y = [tmp_known[k].Y[0], tmp_known[k].Y[1]]


								plotBoxes()


						######################################################################
						# Plot positions of detections from 'VisibleSatellites' stash file
						######################################################################
						for k in range(len(tmp_all)):
							if tmp_all[k].x_beg is not None:
								box_color = "red"
								box_width = 20
								correlation = False
								arrow = True

								x = [tmp_all[k].x_beg, tmp_all[k].x_end]
								y = [tmp_all[k].y_beg, tmp_all[k].y_end]

								plotBoxes()

					if detections != False:
						for rms_event in rms_detections:
							rms_time = (filenameToDatetime(rms_event[0])-datetime.datetime(1970,1,1)).total_seconds()
							ftp_time = (filenameToDatetime(os.path.split(ff_file)[1])-datetime.datetime(1970,1,1)).total_seconds()

							if rms_event[0] == os.path.split(ff_file)[1]:
							# if (rms_time - ftp_time) >= 0.0 and (rms_time-ftp_time) <= 27.999:
							
								print(rms_time-ftp_time, os.path.split(ff_file)[1])
								if  rms_event[11][0][2] is not None:
									box_color = 'purple'
									box_width = 35
									correlation = False
									arrow = True

									x = [rms_event[11][0][2], rms_event[11][-1][2]]
									y = [rms_event[11][0][3], rms_event[11][-1][3]]

									# plotBoxes()

									dX = rms_event[11][-1][2] - rms_event[11][0][2]
									dY = rms_event[11][-1][3] - rms_event[11][0][3]

									pa = math.degrees(math.atan2(dY,dX))
									width = math.sqrt(dX*dX + dY*dY)
									height = 9

									midpt = [1/2*dX + rms_event[11][0][2], 1/2*dY + rms_event[11][0][3]]

									rec_rms = patches.Rectangle((-1/2*width, -1/2*height), width, height, \
										transform=mpl.transforms.Affine2D().rotate_deg_around(*(0,0),pa) \
										+ mpl.transforms.Affine2D().translate(midpt[0], midpt[1]) + ax.transData, \
										fill=False, edgecolor=box_color, linestyle='dashed')

								ax.add_patch(rec_rms)

					plt.xlim(0,config.width)
					plt.xticks(fontsize=5)
					plt.ylim(0,config.height)
					plt.yticks(fontsize=5)
					plt.title(time_beg, fontsize=10)

			else:
				print('Sorry. That FF file does not exist.' + ff_file)

		if output_dir is not None:
			if not os.path.exists(contact_sheet_dir):
				os.makedirs(contact_sheet_dir)

			plt.savefig(os.path.join(contact_sheet_dir, os.path.splitext(os.path.split(ff_file)[1])[0] + "_Satellites.png"))
			plt.close()

		if cml_args.show == True:
			plt.show()

def getSatelliteCoords(location, time_range, tle1, tle2, eph, platepar=None, sat_name=None):
	'''
	As the name implies, this function is called when you want start and end positions for a
	satellite given a tle and start and end times.

	Arguments:
		location (SkyField location object) - The observer's location as defined with SkyField
		time_range [datetime]- A two-element array of start and end times as datetime objects
		tle1 (list) - Line 1 of the TLE of interest
		tle2 (list) - Line 2 of the TLE of interest
		eph (SkyField object) - Solar system object ephemerides
		platepar - RMS platepar to be used for transforming RA/Dec or Az/Alt to image x/y coordinates
		sat_name  - The name of the satellite (line 0 from the TLE)

	Returns:
		X, Y, height, on_image_by_azalt, sunlit
	'''
	ts = load.timescale()

	times = ts.from_datetimes(time_range)
	line1 = tle1
	line2 = tle2

	satellite = EarthSatellite(line1, line2, sat_name, ts)
	sunlit = satellite.at(times).is_sunlit(eph)
	# if sunlit[0] == False or sunlit[1] == False:
	# 	print(f"{sat_name} is sunlit? {sunlit}")

	# Find geocentric coordinates for begin and end times
	geocentric = satellite.at(times)
	height = wgs84.height_of(geocentric)

	# Where is the satellite relative to location
	difference = satellite - location

	# Get the topocentric coordinates
	topocentric = difference.at(times)

	# Get alt, az, and distance for the satellite
	alt, az, dist = topocentric.altaz()

	tjd = datetime2JD(time_range[0])

	az_alt_corners = getPlateparAzAltCorners(platepar, tjd)

	on_image_by_azalt = sphericalPolygonCheck(np.array(az_alt_corners), np.array(list(zip(az.degrees, alt.degrees))))

	# Get RA, Dec, and distance for the satellite
	ra, dec, dist = topocentric.radec()

	X, Y = raDecToXYPP(ra._degrees, dec.degrees, times.tdb[0], platepar)

	return X, Y, height, on_image_by_azalt, sunlit

def getCurvedSatelliteCoords(location, time_range, tle1, tle2, eph, platepar=None, sat_name=None, decim=50):
	'''
	As the name implies, this function is called when you want start and end positions for a
	satellite given a tle and start and end times.

	Arguments:
		location (SkyField location object) - The observer's location as defined with SkyField
		time_range [datetime]- A two-element array of start and end times as datetime objects
		tle1 (list) - Line 1 of the TLE of interest
		tle2 (list) - Line 2 of the TLE of interest
		eph (SkyField object) - Solar system object ephemerides
		platepar - RMS platepar to be used for transforming RA/Dec or Az/Alt to image x/y coordinates
		sat_name  - The name of the satellite (line 0 from the TLE)

	Returns:
		X, Y, height, on_image_by_azalt, sunlit
	'''
	ts = load.timescale()

	t0 = ts.from_datetime(time_range[0])
	t1 = ts.from_datetime(time_range[1])

	# times = ts.from_datetimes(time_range)
	times = ts.linspace(t0, t1, decim)
	line1 = tle1
	line2 = tle2

	satellite = EarthSatellite(line1, line2, sat_name, ts)
	sunlit = satellite.at(times).is_sunlit(eph)
	# if sunlit[0] == False or sunlit[1] == False:
	# 	print(f"{sat_name} is sunlit? {sunlit}")

	# Find geocentric coordinates for begin and end times
	geocentric = satellite.at(times)
	height = wgs84.height_of(geocentric)

	# Where is the satellite relative to location
	difference = satellite - location

	# Get the topocentric coordinates
	topocentric = difference.at(times)

	# Get alt, az, and distance for the satellite
	alt, az, dist = topocentric.altaz()


	tjd = datetime2JD(time_range[0])

	az_alt_corners = getPlateparAzAltCorners(platepar, tjd)

	on_image_by_azalt = sphericalPolygonCheck(np.array(az_alt_corners), np.array(list(zip(az.degrees, alt.degrees))))

	# Get RA, Dec, and distance for the satellite
	ra, dec, dist = topocentric.radec()

	X, Y = raDecToXYPP(ra._degrees, dec.degrees, times.tdb[0], platepar)

	return X, Y, height, on_image_by_azalt, sunlit

def get_data(ff_path):
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


def getPlatePar(ff_path):
	# Load the correct platepar for the event
		# Find recalibrated platepars file per FF file
		recalibrated_platepars = {}

		platepars_recalibrated_file = glob.glob(os.path.join(ff_path, config.platepars_recalibrated_name))

		if len(platepars_recalibrated_file) != 1:
			print('unable to find a unique platepars file in {}'.format(ff_dir))
			return False

		with open(platepars_recalibrated_file[0]) as f:
			pp_per_dir = json.load(f)
			# Put the full path in all the keys
			for key in pp_per_dir:
				recalibrated_platepars[os.path.join(os.path.dirname(platepars_recalibrated_file[0]), key)] = pp_per_dir[key]

		return recalibrated_platepars


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

def main(ff_path, tle_path=None, stash=None, vis=None):

# ##########
# # Make a contact sheet of detections and satellite correlations

	# output_dir = "/home/mmazur/source/Projects/LCAM"

	rms_detections = get_data(ff_path)

	if tle_path is not None:
		# getTLEsFromTxt(tle_path)
		# tles = load.tle_file(tle_path, reload=True)
		# print('Loaded', len(tles), 'satellites')
		# tles = [t for t in tles if name_filter == t['OBJECT_NAME']]		
		satellites = list(SkyfieldTLESatellite.loadTLEFileSatellites(tle_path))
	else:
		satellites=None

	if stash is not None:
		stashes = listStashes(stash[0])
		# print("stashes: ", stashes)
		lcam_matches = readStash(stashes[0])
		print("lcam_matches", lcam_matches)
	else:
		lcam_matches = None

	if vis is not None:
		vis_stashes = listStashes(vis[0])
		# print("vis stashes: ", vis_stashes)
		vis_stash = readStash(vis_stashes[0])
		print("vis_stash: ", vis_stash)
	else:
		vis_stash = None

	makeContactSheet(rms_detections, lcam_matches, vis_stash, satellites, output_dir, detections=True, video=from_video)


if __name__=="__main__":

	# Read arguments with argparser
	arg_parser = argparse.ArgumentParser(description="Create a contact sheet with observations and TLE overlays")
	arg_parser.add_argument('ff_path', nargs='+', metavar='DIR_PATH', type=str, \
		help='Path to the image file directory.')
	arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
		help='Path to a config file which will be used instead of the default config file.')
	arg_parser.add_argument('--stash', nargs=1, metavar='STASH_FILE', type=str, \
		help='Path to the stash file containing LCAM correlations. If not specified, \
		LCAM correlations will not be plotted.')
	arg_parser.add_argument('--vis', nargs=1, metavar='VISIBILITY_FILE', type=str, \
		help='')
	arg_parser.add_argument('-p', '--plate', metavar='ASTRO_PLATE', type=str, \
		help='Full path to astrometric plate file.')
	arg_parser.add_argument('-e', '--elements', metavar='TLE_PATH', type=str, \
		help='Full path to TLE file.')
	arg_parser.add_argument('-o', '--output', metavar='OUTPUT_DIR', type=str, \
		help='Full path to output directory where plots will be saved.')
	arg_parser.add_argument('-s', '--show', action='store_true', \
		help='Show plots as they are generated. Pauses until plot is closed.')
	arg_parser.add_argument('--noimages', action='store_true', \
		help='Used if you do not want to show an background images.')
	arg_parser.add_argument('--video', action='store_true', \
		help='If your FF files are from MKVs')

	# Parse command-line arguments
	cml_args = arg_parser.parse_args()

	# Get the path to the ff files from the parser
	ff_path = cml_args.ff_path

	from_video = cml_args.video

	if len(ff_path) == 1:
		# If a directory
		if os.path.isdir(ff_path[0]):
			ff_path_list = glob.glob(os.path.join(cml_args.ff_path[0], "*.fits"))

		# If a file is given
		elif os.path.isfile(ff_path[0]):
			ff_path_list = cml_args.ff_path

		else:
			print("No valid file or directory given!")
			sys.exit()

	# If multiple files are given, parse them into a list
	else:
		ff_path_list = []
		for entry in ff_path:
			ff_path_list += glob.glob(entry)

	# Read the configuration file
	print("Loading %s" % cml_args.config)
	config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath("."))

	# Get the 'stash' file name
	if cml_args.stash is not None:
		stash_file = cml_args.stash
	else:
		stash_file = None

	if cml_args.vis is not None:
			vis_file = cml_args.vis
	else:
		vis_file = None

	if cml_args.elements is None:
		print("No TLE file specified")
		tle_path = None
	else:
		tle_path = cml_args.elements
		# print(tle_path)

	# Check to see if ff_path is a list. If it is, set ff_path to the first element of the list.
	if isinstance(ff_path, list):
		ff_path = ff_path[0]

	# Check to see if an output directory is given. If not, write the output to the ff_path directory.
	if cml_args.output is None:
		output_dir = ff_path[0]
		print("output dir: ", output_dir)
	else:
		output_dir = cml_args.output
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)

	if cml_args.noimages is False:
		no_images = False

	# file_list = glob.glob(os.path.join(ff_path, "FTP*"))
	# ftp_file = glob.glob(os.path.join(ff_path, "FTP*"))[0]
	print(ff_path, tle_path, stash_file, vis_file)
	main(ff_path, tle_path, stash_file, vis=vis_file)