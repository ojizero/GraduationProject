import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import os
import numpy as np
from glob import iglob

from features.features_transformer import FeaturesTransformer
from utils.decorators import classinstancemethod


class DatasetHandler:
	_FEATURE_VECTOR_MAKER = FeaturesTransformer.transform

	def __init__ (self, **kwargs):
		self.vector_maker   = kwargs.pop('vector_maker', DatasetHandler._FEATURE_VECTOR_MAKER)
		self.files_iterator = kwargs.pop('csv_data')
		self._vector_names  = ()

		self.path = kwargs.pop('path')
		self.opts = kwargs
		self._str = None
		self._rpr = None

	@classmethod
	def from_csv_directory (cls, path):
		return cls(csv_data=iglob('%s/**/**/*.csv' % path), path=path)

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
		label     = next_file[:next_file.find('.')] # fix had

		data_whole   = np.genfromtxt(next_file, delimiter=',')
		data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

		features_names, feature_vector = self.vector_maker(data=data_streams, **self.opts)

		print (next_file)
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
		if self._rpr is None:
			self._rpr = repr((*((label,) + vector for label, vector in self),))

		return self._rpr

	def __str__ (self):
		if self._str is None:
			# perfrom initial retreival, this is to set the _vector_names parameter
			label, vector = self.__next__()
			# stringify header
			header = self._inclose('label', *self._vector_names)
			# stringify initial data
			first_vector = self._inclose(label, *vector)
			# generate the string object
			self._str = '\n'.join([header, first_vector, *[self._inclose(label, *vector) for label, vector in self]])

		return self._str

	@staticmethod
	def _inclose (*args):
		return ', '.join(['"%s"' % str(a).replace('"', "'") for a in args])

	def store_csv (self, csv_out):
		out_str = str(self)
		with open(csv_out, 'w') as out:
			out.write(out_str)

if __name__ == '__main__':
	dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data')

	dataset.store_csv('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset_dump.csv')
