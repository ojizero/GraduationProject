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

	@classmethod
	def from_directory_csv (cls, **kwargs):
		path = kwargs.pop('path', os.getcwd())
		return cls(csv_data=iglob('%s/**/**/*.csv' % path), path=path)

	def __iter__ (self):
		return self

	def __next__ (self):
		try:
			next_file = self.files_iterator.__next__()
			label     = next_file[:next_file.find('.')] # fix had

			data_whole   = np.genfromtxt(next_file, delimiter=',')
			data_streams = np.array([data_whole[:,r:r+6] for r in range(0, 54, 9)])

			features_names, feature_vector = self.vector_maker(data=data_streams, **self.opts)

			if self._vector_names is ():
				self._vector_names = features_names
			elif self._vector_names != features_names:
				raise Exception('features labels do not match')

			return label, feature_vector
		except StopIteration as stop:
			# reset object, for future use if needed
			self.csv_data = iglob('%s/**/**/*.csv' % self.path)
			# stop iterations
			raise stop

	def __repr__ (self):
		return ((label,) + vector for label, vector in self)

	def __str__ (self):
		# perfrom initial retreival
		label, vector = self.__next__()
		# write header to file
		header = ('label',) + self._vector_names
		out.write(', '.join(header))
		out.write('\n')

		# write initial data to file

		# loop over rest of the data
		for label, vector in self:
			pass

	def store_csv (self, **kwargs):
		# , os.getcwd()
		with open(kwargs.pop('csv_out'), 'w') as out:
			out.write(str(self))

if __name__ == '__main__':
	dataset = DatasetHandler.from_directory_csv(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data')

	dataset.store_csv(csv_out='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset_dump.csv')
