import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import os
import re
import numpy as np
from glob import iglob

# configure pringing optiosn of numpy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from utils.decorators import classinstancemethod
from features.features_transformer import FeaturesTransformer

_verbal_code = re.compile(r'\s+|array|\(|\)|\'|"')
_path_label_pattern = r'^.*/(\w*)(?:\.\w+)?/.*\.csv$'
_rgx = re.compile(_path_label_pattern)


class DatasetHandler:
	_LABEL_MAKER = lambda path: _rgx.search(path).groups()[0]
	_FEATURE_VECTOR_MAKER = FeaturesTransformer.transform

	def __init__ (self, **kwargs):
		self.vector_names     = kwargs.pop('vector_names', ())
		self.dataset_iterator = kwargs['dataset_iterator']

		self.opts = kwargs

	@classmethod
	def from_csv_directory (cls, path, **kwargs):
		files_iterator = iglob('%s/**/**/*.csv' % path)
		vector_maker   = kwargs.pop('vector_maker', cls._FEATURE_VECTOR_MAKER)
		label_maker    = kwargs.pop('label_maker', cls._LABEL_MAKER)
		delimiter      = kwargs.pop('delimiter', ',')

		def _dataset_generator ():
			for next_file in files_iterator:
				with open(next_file, 'r') as f:
					data_whole   = np.genfromtxt(next_file, delimiter=delimiter)
					data_streams = np.array(data_whole[:,r:r+6] for r in range(0, 54, 9))

					yield (label_maker(next_file),) + vector_maker(data=data_streams, **kwargs)

		return cls(dataset_iterator=_dataset_generator(), **kwargs)

	@classmethod
	def from_csv_file (cls, path, delimiter=';', **kwargs):
		input_generator = cls._input_csv_generator(path)

		vector_names = tuple(col.strip() for col in input_generator.__next__().decode().split(delimiter)[1:])
		vector_maker = lambda row: (row[0], vector_names, row[1:])

		def _dataset_generator ():
			for row in np.genfromtxt(input_generator, delimiter=delimiter, dtype=np.float):
				yield vector_maker(row)

		return cls(dataset_iterator=_dataset_generator(), vector_names=vector_names, **kwargs)

	def __iter__ (self):
		return self

	def __next__ (self):
		label, features_names, feature_vector = self.dataset_iterator.__next__()

		if self.vector_names is ():
			self.vector_names = features_names
		elif self.vector_names != features_names:
			raise Exception('features labels do not match')

		return label, feature_vector

	def store_csv (self, csv_out):
		with open(csv_out, 'w') as out:
			# perfrom initial retreival, this is to set the _vector_names parameter
			label, vector = self.__next__()
			# write header
			header = self._inclose('label', *self.vector_names)
			out.write(header)
			out.write('\n')
			# write initial data
			first_vector = self._inclose(label, *vector)
			out.write(first_vector)
			out.write('\n')
			# write rest of data
			for label, vector in self:
				output = self._inclose(label, *vector)
				out.write(output)
				out.write('\n')

	@staticmethod
	def _inclose (*args):
		return ' ; '.join('%s' % _verbal_code.sub('', repr(a)) for a in args)

	@staticmethod
	def _input_csv_generator (csv_in):
		with open(csv_in, 'rb') as in_:
			for line in in_:
				yield line


if __name__ == '__main__':
	# dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data')

	# dataset.store_csv('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset_dump.csv')
	dataset = DatasetHandler.from_csv_file('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset_sample.csv')

	for d in dataset.dataset_iterator:
		print(type(d))