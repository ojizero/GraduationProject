import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import numpy as np
from re import match

from utils.decorators import classinstancemethod


class Extractor:
	'''

	Main class responsible for extracting features from data.
	Sub-classes are responsible for defining said features
	and how they interpolate input data.

	'''
	_EXTRACT_ON  = lambda string: match('.*_feature$', string)
	_WINDOW_SIZE = 10
	_TRANSFORMER = lambda _: _

	@classinstancemethod
	def extract (obj, **kwargs):
		'''
		Extracts any number of feature from input data.
		This method can be called from an instance,
		or directly from the class, the method's
		behaviour is the same and the only
		difference is the internal behaviour
		for features.

		Keyworded Arguments:
			data        (array-like|required) :
				The input data, preferably NumPy array, currently
				it has to be shaped as (Streams, Samples, Readings),
				if (Samples, Readings) multi option must be set to False.
				- First dimension is the data stream
				- Second dimension is the samples
				- Third dimension is the sample readings.

			multi       (boolean|optional)   :
				Describes whether the data is 3D (True) or 2D (False)
				Defaults to True

			window_size (int|optional)       :
				Controls how many windows the streams will be split on,
				feature functions are applied at the window level.
				Defaults to the class _WINDOW_SIZE parameter,
				Base class default 10

			overlap     (float|optional)     :
				A number between greater than 0 and less than 1,
				describes the overlapping percentage betweent
				the windows.
				Defaults to 0

			extract_on  ((str) -> boolean|optional) :
				A function that decides (based on a function's name)
				whether or not it is a feature function.
				Base class default lambda string: match('.*_feature$', string)

			**kwargs    (dict|optional)      : Any additional passed kwargs are passed to internal functions

		Returns:
			NumPy Array :
				An array, with the first index representing to the column or readings index,
				which points to a dictionary mapping each feature's name to an array
				which is indexed by the stream index, then the window index, containing
				the feature output for that window.

				[column/reading]['feature_name'][sensor][window]
		'''
		assert kwargs.get('data', []) != [], '`data` is a required parameter'
		data = kwargs.pop('data')

		## parameters and their default values
		if not kwargs.get('multi', True):
			data = np.array([data])

		window_size = kwargs.pop('window_size', obj._WINDOW_SIZE)

		overlap = kwargs.pop('overlap', 0.0)
		assert 0.0 <= overlap < 1

		transformer = kwargs.pop('transformer', obj._TRANSFORMER)

		## method logic parameters
		begin = border = window_size // 2
		step  = window_size - round(overlap * window_size)
		# maybe set this to step size // 2 instead of window size ?
		end   = data.shape[1] - window_size // 2 + 1

		# window the data
		data_windowed = np.array([data[:,pivot-border:pivot+border,...] for pivot in range(begin, end, step)])

		readings = data_windowed.shape[-1]
		# returns R -> R[reading]['feature_method_name'][sensor][window]
		return transformer(np.array([obj._extract(data_windowed[...,col], **kwargs) for col in range(readings)]))

	@classinstancemethod
	def _extract (obj, data_column, **kwargs):
		'''
		Performs actual extraction, uses introspection to get all methods
		of the calling object, using the extrac_on method as a filter.
		'''
		extract_on = kwargs.get('extract_on', obj._EXTRACT_ON)

		if isinstance(obj, type):
			kwargs['cls'] = obj
		else:
			kwargs['self'] = obj

		kwargs['data_column'] = data_column
		# perform each method ending with `extract_on` from given class or instance on given data
		return {name: getattr(obj, name)(**kwargs) for name in dir(obj) if extract_on(name)}

	@staticmethod
	def _generic_loop (data_streams, feature_function):
		return np.array(np.nan_to_num([
			[feature_function(window) for window in stream]
				for stream in data_streams
		]))
