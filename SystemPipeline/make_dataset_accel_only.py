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

	# instance responsible for segmentation
	splicer = Splicer(threshold=80)

	active_start, active_end = _slicer(splicer, data)

	active_region_data = data[:,active_start:active_end,3:6]

	print(data.shape, active_region_data.shape)

	return FeaturesTransformer.transform(data=active_region_data, **kwargs)

dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data', overlap=0.0, vector_maker=vector_maker)
dataset.store_csv('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/floats.dataset.accel.only.withnative.dump.csv')
