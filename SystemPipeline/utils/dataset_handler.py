import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import os
import re
import numpy as np
# configure pringing optiosn of numpy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from utils.decorators import classinstancemethod
from features.features_transformer import FeaturesTransformer

_verbal_code = re.compile(r'\s+|array|\(|\)|\'|"')
_path_label_pattern = r'^.*/(\w*)(?:\.\w+)?/.*\.csv$'
_rgx = re.compile(_path_label_pattern)

_array_regex = re.compile(r'^\[.*\]$')
_float_regex = re.compile(r'^[+-]?\d+(?:\.\d+)?(?:e[+-]\d+)?$')
_cmplx_regex = re.compile(r'^\(?(?:[+-]?\d+(?:\.\d+)?(?:e[+-]\d+)?[+-])?[+-]?\d+(?:\.\d*)?(?:e[+-]\d+)?j\)?$')


class DatasetHandler:
	_FEATURE_VECTOR_EXTRACTOR = FeaturesTransformer.transform

	def __init__ (self, **kwargs):
		self.vector_names     = kwargs.pop('vector_names', ())
		self.dataset_iterator = kwargs['dataset_iterator']

		self.opts = kwargs

	@classmethod
	def from_csv_directory (cls, path, delimiter=',', **kwargs):
		files_iterator = (os.path.join(dirpath, file) for dirpath, subdirs, files in os.walk(path) for file in files if file.endswith('.csv'))
		vector_maker   = kwargs.pop('vector_maker', cls._FEATURE_VECTOR_EXTRACTOR)
		label_maker    = kwargs.pop('label_maker', lambda path: _rgx.search(path).groups()[0])

		def _dataset_generator ():
			for next_file in files_iterator:
				with open(next_file, 'r') as f:
					data_whole   = np.genfromtxt(next_file, delimiter=delimiter)
					# read only accel heek
					# data_streams = np.array([data_whole[:,r+3:r+6] for r in range(0, 54, 9)])
					data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])
					print(next_file)
					try:
						yield (label_maker(next_file),) + vector_maker(data=data_streams, **kwargs)
					except AssertionError:
						print('skipping due to AssertionError')
						continue

		return cls(dataset_iterator=_dataset_generator(), **kwargs)

	@classmethod
	def from_csv_file (cls, path, delimiter=',', **kwargs):
		input_generator = cls._input_generator(path)

		vector_names = tuple(col.strip() for col in input_generator.__next__().decode().split(delimiter)[1:])
		vector_maker = kwargs.pop('vector_maker', lambda row, names: (row[0], names, row[1:]))

		def _dataset_generator ():
			for row in input_generator:
				# THIS IS A BREAKING CHANGE !!!!
				yield vector_maker(cls._convert_str(row, delimiter), vector_names)

		return cls(dataset_iterator=_dataset_generator(), **kwargs) # , vector_names=vector_names

	def __iter__ (self):
		return self

	def __next__ (self):
		label, features_names, features_vector = self.dataset_iterator.__next__()

		if self.vector_names is ():
			self.vector_names = features_names
		elif self.vector_names != features_names:
			# raise ValueError('features labels do not match')
			print('skipping due to ValueError')
			return self.__next__()

		return label, features_vector

	# think a better implementation ?
	def as_arrays (self):
		_rm_nan = lambda _ : np.nan_to_num(_)

		dataset = [*self]

		labels = np.array([label for label, _ in dataset])
		values = np.array([value for _, value in dataset], dtype=self.opts.get('dtype', np.float))

		values = _rm_nan(values) if self.opts.get('nan_to_num', False) else values

		return labels, values

	def store_csv (self, csv_out):
		# append to file
		with open(csv_out, 'ab') as out:
			# perfrom initial retreival, this is to set the _vector_names parameter
			label, vector = self.__next__()
			header = ('label',) + self.vector_names

			if (self.opts.get('store_header', True)):
				out.write((','.join(header)).encode())
				out.write(b'\n')

			# write initial data
			out.write(b'%s,' % label.encode())
			np.savetxt(out, vector, delimiter=',', newline=',')
			out.write(b'\n')

			# write rest of data
			for label, vector in self:
				out.write(b'%s,' % label.encode())
				np.savetxt(out, vector, delimiter=',', newline=',')
				out.write(b'\n')

	@staticmethod
	def _input_generator (csv_in):
		with open(csv_in, 'rb') as in_:
			for line in in_:
				yield line

	# deprectaded refactor away from using it
	@staticmethod
	def _convert_str (string, delimiter):
		if isinstance(string, bytes):
			string = string.decode()

		string = re.sub(r',(?:\n)?$', '', string)

		is_array = lambda s: _array_regex.match(s) is not None
		is_float = lambda s: _float_regex.match(s) is not None
		is_cmplx = lambda s: _cmplx_regex.match(s) is not None

		cols = string.split(delimiter)

		def _converter ():
			for col in cols[1:]:
				ret = None
				col = col.strip()

				# because fuck numpy :|
				col = re.sub(r'(?:\+\-|\-\+)', '-', col)

				if is_array(col):
					ret = DatasetHandler._convert_str(col[1:-1], ',')
				elif is_float(col):
					ret = np.float(col)
				elif is_cmplx(col):
					ret = np.complex(col)
				else:
					raise ValueError('unhandled datatype :: "%s"' % col)

				yield ret

		return np.array([cols[0], *_converter()], dtype=object)




if __name__ == '__main__':
	from itertools import zip_longest
	from segmentation.segmentation import Splicer

	def vector_maker (data, **kwargs):
		def _slicer (data, silence_regions, min_len=20):
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

			print(region_couples)
			print(major_active)

			active_start = major_active[0][1]
			active_end   = major_active[1][0]

			return data[:,active_start:active_end]


		splicer = Splicer(threshold=80)
		intensity_total = splicer.smooth_intensities(
			sum([
				splicer.measure_intensity(stream) for stream in data
			])
		)
		silence_regions = splicer.silence_segments(intensity_total)

		print(silence_regions)
		# assert 0 < len(silence_regions) <= 2, 'assertion (one active region), failed'

		active_region_data = _slicer(data, silence_regions)

		print(data.shape, active_region_data.shape)

		features = FeaturesTransformer.transform(data=active_region_data, **kwargs)

		# print(features)

		return features


	dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data', overlap=0.0, vector_maker=vector_maker)

	dataset.store_csv('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset.new.rawdata.noseg.csv')

	# dataset = DatasetHandler.from_csv_file('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/datasetdump.new.csv', delimiter=',')

	# for l, d in dataset:
	# 	print(l)
	# 	for c in d:
	# 		assert not isinstance(c, Iterable), 'fuck'
	# 		print(type(c), end=' ;; ')
