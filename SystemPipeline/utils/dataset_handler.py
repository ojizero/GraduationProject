import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import os
import re
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from glob import iglob

from features.features_transformer import FeaturesTransformer
from utils.decorators import classinstancemethod


_verbal_code = re.compile(r'\s+|array|\(|\)|\'|"')
_path_label_pattern = r'^.*/(\w*)(?:\.\w+)?/.*\.csv$'
_rgx = re.compile(_path_label_pattern)


class DatasetHandler:
	_LABEL_MAKER = lambda path: _rgx.search(path).groups()[0]
	_FEATURE_VECTOR_MAKER = FeaturesTransformer.transform

	def __init__ (self, **kwargs):
		self.vector_maker   = kwargs.pop('vector_maker', DatasetHandler._FEATURE_VECTOR_MAKER)
		self.label_maker    = kwargs.pop('label_maker', DatasetHandler._LABEL_MAKER)
		self.files_iterator = kwargs.pop('csv_data')
		self._vector_names  = ()

		self.path = kwargs.pop('path')
		self.opts = kwargs

	@classmethod
	def from_csv_directory (cls, path):
		return cls(csv_data=iglob('%s/*.csv' % path), path=path)

	@classmethod
	def from_csv_file (cls, **kwargs):
		pass

	def __iter__ (self):
		return self

	# find a way to reset the object, or store it somehow
	# currently it can be iterated over only once
	def __next__ (self):
		# try:
		next_file = self.files_iterator.__next__()
		label     = self.label_maker(next_file)

		data_whole   = np.genfromtxt(next_file, delimiter=',')
		data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

		features_names, feature_vector = self.vector_maker(data=data_streams, **self.opts)

		if self._vector_names is ():
			self._vector_names = features_names
		elif self._vector_names != features_names:
			raise Exception('features labels do not match')

		return label, feature_vector
		# except StopIteration as stop:
		# 	# reset object, for future use if needed
		# 	self.csv_data = iglob('%s/**/**/*.csv' % self.path)
		# 	# stop iterations
		# 	raise stop

	def __repr__ (self):
		return repr((*((label,) + vector for label, vector in self),))

	def __str__ (self):
		# perfrom initial retreival, this is to set the _vector_names parameter
		label, vector = self.__next__()
		# stringify header
		header = self._inclose('label', *self._vector_names)
		# stringify initial data
		first_vector = self._inclose(label, *vector)
		# generate the string object
		return '\n'.join([header, first_vector, *[self._inclose(label, *vector) for label, vector in self]])

	@staticmethod
	def _inclose (*args):
		return ' ; '.join('%s' % _verbal_code.sub('', repr(a)) for a in args)

	def store_csv (self, csv_out):
		out_str = str(self)
		with open(csv_out, 'w') as out:
			out.write(out_str)

	# instead of storing the whole dataset in memory as string
	# perform iteration while outputing to the file
	def store_csv_immediate (self, csv_out):
		with open(csv_out, 'w') as out:
			# perfrom initial retreival, this is to set the _vector_names parameter
			label, vector = self.__next__()
			# write header
			header = self._inclose('label', *self._vector_names)
			out.write(header)
			out.write('\n')
			# write initial data
			first_vector = self._inclose(label, *vector)
			out.write(first_vector)
			out.write('\n')

			for label, vector in self:
				output = self._inclose(label, *vector)
				out.write(output)
				out.write('\n')


if __name__ == '__main__':
	dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data/ameer/6a')

	dataset.store_csv('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset_dump.csv')
