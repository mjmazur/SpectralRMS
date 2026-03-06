import os, sys, glob
import argparse
import datetime
import zlib

import numpy as np
import pandas as pd

import RMS.ConfigReader as cr

from RMS.Formats.FFfile import validFFName
from RMS.Formats.FFfile import read as readFFfile
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo, findFTPdetectinfoFile, readFTPdetectinfo

def getPlatePar(ff_path):
	# Load the correct platepar for the event
		# Find recalibrated platepars file per FF file
		recalibrated_platepars = {}

		platepars_recalibrated_file = glob.glob(os.path.join(ff_path, config.platepars_recalibrated_name))

		print(platepars_recalibrated_file)

		if len(platepars_recalibrated_file) != 1:
			print('unable to find a unique platepars file in {}'.format(ff_dir))
			return False

		with open(platepars_recalibrated_file[0]) as f:
			pp_per_dir = json.load(f)
			# Put the full path in all the keys
			for key in pp_per_dir:
				recalibrated_platepars[os.path.join(os.path.dirname(platepars_recalibrated_file[0]), key)] = pp_per_dir[key]

		return recalibrated_platepars

def cutoutFromMKV(video_path, time=None, crop_area=None, type="vid"):
	
	dir_path, file_name = os.path.split(video_path)
	# cutout_file_name = os.path.basename(video_path).split('.')[0] + ".vid"

	start_time = filenameToDatetime(file_name)
	frame_time = (start_time - datetime.datetime(1970,1,1)).total_seconds()

	cap = cv2.VideoCapture(video_path)
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Can't receive frame")
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Crop
		gray = gray[888-20:888+20, 302-20:302+60]
		frames.append(gray)

	cap.release()

	video = np.stack(frames, axis=0)

	maxpixel = video.max(axis=0)
	avepixel = video.mean(axis=0)
	stdpixel = video.std(axis=0)
	maxframe = video.argmax(axis=0)

	plt.imshow(maxpixel)
	plt.show()

def loadGMNStarCatalog(file_path, years_from_J2000=0, lim_mag=None, mag_band_ratios=None, catalog_file=''):
    """
    Reads in the GMN Star Catalog from a compressed binary file, applying proper motion correction,
    magnitude limiting, and synthetic magnitude computation. Adjusts the RA/Dec positions to the J2000 epoch.

    Arguments:
        file_path: [str] Path to the binary file.

    Keyword arguments:
        years_from_J2000: [float] Years elapsed since J2000 for proper motion correction (default: No correction added).
        lim_mag: [float] Limiting magnitude for filtering stars (default: None).
        mag_band_ratios: [list] Relative contributions of photometric bands [B, V, R, I]
            to compute synthetic magnitudes (default: None).
        catalog_file: [str] Name of the catalog file (default: ''). Used for caching purposes.

    Returns:
        filtered_data: [ndarray] A filtered and corrected catalog contained as a structured NumPy array 
            (currently outputs only: ra, dec, mag)
        mag_band_string: [str] A string describing the magnitude band of the catalog.
        mag_band_ratios: [list] A list of BVRI magnitude band ratios for the given catalog.
    """

    # Catalog data used for caching
    cache_name = "_catalog_data_{:s}".format(catalog_file.replace(".", "_"))

    # Step 1: Cache the catalog data to avoid repeated decompression
    if not hasattr(loadGMNStarCatalog, cache_name):

        # Define the data structure for the catalog
        data_types = [
            ('designation', 'S30'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('pmra', 'f8'),
            ('pmdec', 'f8'),
            ('phot_g_mean_mag', 'f4'),
            ('phot_bp_mean_mag', 'f4'),
            ('phot_rp_mean_mag', 'f4'),
            ('classprob_dsc_specmod_star', 'f4'),
            ('classprob_dsc_specmod_binarystar', 'f4'),
            ('spectraltype_esphs', 'S8'),
            ('B', 'f4'),
            ('V', 'f4'),
            ('R', 'f4'),
            ('Ic', 'f4'),
            ('oid', 'i4'),
            ('preferred_name', 'S30'),
            ('Simbad_OType', 'S30')
        ]

        with open(file_path, 'rb') as fid:

            # Read the catalog header
            declared_header_size = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_rows = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            num_columns = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
            fid.read(declared_header_size - 12)  # Skip column names

            # Read and decompress the catalog data
            compressed_data = fid.read()
            decompressed_data = zlib.decompress(compressed_data)
            catalog_data = np.frombuffer(decompressed_data, dtype=data_types, count=num_rows)

        # Cache the catalog data for future use
        setattr(loadGMNStarCatalog, cache_name, catalog_data)
    
    else:
        catalog_data = getattr(loadGMNStarCatalog, cache_name)

    # Step 2: Compute synthetic magnitudes if required
    if mag_band_ratios is not None:
        
        # Compute synthetic magnitudes if band ratios are provided
        total_ratio = sum(mag_band_ratios)
        rb, rv, rr, ri, rg, rbp, rrp = [x/total_ratio for x in mag_band_ratios]
        synthetic_mag = (
            rb*catalog_data['B'] +
            rv*catalog_data['V'] +
            rr*catalog_data['R'] +
            ri*catalog_data['Ic'] +
            rg*catalog_data['phot_g_mean_mag'] +
            rbp*catalog_data['phot_bp_mean_mag'] +
            rrp*catalog_data['phot_rp_mean_mag']
        )
        mag_mask = synthetic_mag <= lim_mag

    else:
        synthetic_mag = catalog_data['V']

    # Step 3: Filter stars based on limiting magnitude
    if lim_mag is not None:
        
        # Generate a mask for stars fainter than the limiting magnitude
        mag_mask = synthetic_mag <= lim_mag

        # Apply the magnitude filter
        catalog_data = catalog_data[mag_mask]
        synthetic_mag = synthetic_mag[mag_mask]
        
    # Step 4: Apply proper motion correction
    mas_to_deg = 1/(3.6e6)  # Conversion factor for mas/yr to degrees/year
    
    # GMN catalog is relative to the J2016 epoch (from GAIA DR3)
    time_elapsed = years_from_J2000 - 16

    # Correct the RA and Dec relative to the years_from_J2000 argument
    corrected_ra = catalog_data['ra'] + catalog_data['pmra']*time_elapsed*mas_to_deg
    corrected_dec = catalog_data['dec'] + catalog_data['pmdec']*time_elapsed*mas_to_deg

    # Step 5: Prepare the filtered data for output
    filtered_data = np.zeros((len(catalog_data), 3), dtype=np.float64)
    filtered_data[:, 0] = corrected_ra  # RA
    filtered_data[:, 1] = corrected_dec  # Dec
    filtered_data[:, 2] = synthetic_mag  # Magnitude

    print(catalog_data['designation'])

    # Step 8: Return the filtered data, magnitude band string, and band ratios
    return catalog_data['preferred_name'], catalog_data['spectraltype_esphs'], catalog_data['Simbad_OType'], filtered_data

def createSpectralCatalogue(catalog_file, lim_mag):

	star_name, spec_type, o_type, star_data = loadGMNStarCatalog(catalog_file, lim_mag=lim_mag)

	columns = ["star_name", "spec_type", "o_type", "ra", "dec", "mag"]
	sn, st, ot, ra, de, ma = [], [], [], [], [], []

	for i in range(len(star_data)):
		sn.append(star_name[i].decode())
		st.append(spec_type[i].decode())
		ot.append(o_type[i].decode())
		ra.append(star_data[i,0])
		de.append(star_data[i,1])
		ma.append(star_data[i,2])
	
	data = {"star_name":sn, "spec_type":st, "o_type":ot, "ra":ra, "dec":de, "mag":ma}

	spectral_df = pd.DataFrame(data)

	return spectral_df



if __name__=="__main__":
	arg_parser = argparse.ArgumentParser(description="Extract spectra from FF and/or MKV files.")
	arg_parser.add_argument('data_path', nargs="+", metavar="DATA_PATH", type=str, \
		help="Path to image/video directory")
	arg_parser.add_argument("-c", "--config", nargs=1, metavar="CONFIG_PATH", type=str, \
		help="Path to a config file which will be used instead of the default config file.")
	arg_parser.add_argument("-p", "--plate", nargs=1, metavar="CONFIG_PATH", type=str, \
		help="Path to platepar file.")

	cml_args = arg_parser.parse_args()

	# Set the path to the data files
	data_path = cml_args.data_path

	# Read the config file. If not given, try to read .config in ~/source/RMS
	if cml_args.config is not None:
		config = cr.loadConfigFromDirectory(cml_args.config, data_path)
	else:
		config = cr.loadConfigFromDirectory("~/source/RMS_data", data_path)

	calstars_file = glob.glob(os.path.join(data_path[0], "CALSTARS*.txt"))

	calstars_list = readCALSTARS(data_path[0], os.path.basename(calstars_file[0]))
	# print(calstars_list[0])

	GMN_catalog_file = glob.glob(os.path.join(data_path[0], "GMN_StarCatalog*.bin"))
	# print(GMN_catalog_file)

	limiting_mag = 3.0
	spectral_catalog = createSpectralCatalogue(GMN_catalog_file[0], lim_mag=limiting_mag)
	spectral_catalog.to_pickle(os.path.join(data_path[0], "GMN_SpectralStarCatalog-M" + str(limiting_mag) + ".pkl"))

	if cml_args.plate is not None:
		plate_file = cml_args.plate
	else:
		plate_file = config.platepar_name

	print(plate_file)
