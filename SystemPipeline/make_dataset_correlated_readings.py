import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')

import numpy as np
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor
from time import time as now

from utils.dataset_handler import DatasetHandler
from preprocessing.processor import Processor
from segmentation.segmentation import Splicer
from features.features_transformer import FeaturesTransformer

def vector_maker (data, **kwargs):
	def _slicer (splicer, data, min_len=20):
		intensity_total = splicer.smooth_intensities(
			sum([
				splicer.measure_intensity(stream) for stream in data
			])
		)
		silence_regions = splicer.silence_segments(intensity_total)

		region_couples = [*zip_longest(silence_regions, silence_regions[1:])]
		if region_couples[-1][1] is None:
			region_couples[-1] = (region_couples[-1][0], (data.shape[1], data.shape[1]))

		for r in region_couples:
			if not (r[0][1] < r[1][0]):
				raise AssertionError('reversed regions')

		major_index = np.argmax(np.array([
			r[1][0] - r[0][1] for r in region_couples
		]))

		major_active = region_couples[major_index]

		# for keeping track of what happens .. could be obsolete ?
		print(region_couples)
		print(major_active)

		active_start = major_active[0][1]
		active_end   = major_active[1][0]

		return active_start, active_end

	def _processor (processors, data):
		def _time_differential ():
			'''
				generator providing the time differenece between its calls, exluding the first call
			'''
			previous = now()
			while True:
				current = now()
				yield current - previous
				previous = current

		time_differential = _time_differential()
		time_differential.__next__() # first differential is'nt valid

		def dt ():
			nonlocal time_differential
			return time_differential.__next__()

		# data[stream][row][reading]
		return np.array([
			[processor.updateIMU(*row[3:6], *row[0:3], dt()) for row in stream]
				for stream, processor in zip(data, processors)
		])

	# instance responsible for segmentation
	splicer    = Splicer(threshold=80)
	# instances responsible for preprocessing, one for each stream
	processors = [Processor() for _ in range(6)]

	# perform both preprocessing and segmentation region calcualtion in parallel
	with ThreadPoolExecutor(max_workers=2) as executor:
		segmentation_future  = executor.submit(_slicer, splicer, data)
		# data[0:-1] to ignore reference sensor from being preprocessed
		preprocessing_future = executor.submit(_processor, processors, data[0:5])

		processed_data = preprocessing_future.result()
		active_start, active_end = segmentation_future.result()

	ref_accel = np.array(data[5,active_start:active_end,2:6])
	ref_accel[:,0] = 0

	# get covariance of all readings per stream
	processed_data = np.apply_along_axis(np.cov, 0, processed_data[:,active_start:active_end])

	# all quaternion numbers from active reagion, fingers streams
	# acceleration readings from active region, reference stream
	active_region_data = np.array([processed_data, ref_accel])
	print(data.shape, active_region_data.shape)

	return FeaturesTransformer.transform(data=active_region_data, **kwargs)

dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data', overlap=0.0, vector_maker=vector_maker)
dataset.store_csv('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/proper.dataset.correlated.dump.csv')
