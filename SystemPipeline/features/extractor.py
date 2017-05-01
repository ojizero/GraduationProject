import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


import numpy as np
from re import match

from utils.decorators import classinstancemethod


class Extractor:
	_EXTRACT_ON  = lambda string: match('.*_feature$', string)
	_WINDOW_SIZE = 10

	@classinstancemethod
	def extract (obj, **kwargs):
		assert kwargs.get('data', []) != [], '`data` is a required parameter'
		data = kwargs['data']

		## parameters and their default values
		if not kwargs.get('multi', True):
			data = np.array([data])

		window_size = kwargs.get('window_size', obj._WINDOW_SIZE)

		overlap = kwargs.get('overlap', 0.0)
		assert 0.0 <= overlap < 1

		## method logic
		begin = border = window_size // 2
		step  = window_size - round(overlap * window_size)
		end   = data.shape[1] - window_size // 2

		# window the data
		data_windowed = np.array([data[:,pivot-border:pivot+border,...] for pivot in range(begin, end, step)])

		readings = data_windowed.shape[-1]
		# returns R -> R[sensor][window][reading]['feature_method_name']
		return np.array([obj._extract(data_windowed[...,col], **kwargs) for col in range(readings)])

	@classinstancemethod
	def _extract (obj, data_column, **kwargs):
		extract_on = kwargs.get('extract_on', obj._EXTRACT_ON)
		kwargs['self']        = obj
		kwargs['data_column'] = data_column
		# perform each method ending with `extract_on` from given class or instance on given data
		return {name: function(**kwargs) for name, function in obj.__dict__.items() if extract_on(name)}
		## add `**kwargs` as parameter to function call

	@staticmethod
	def _generic_loop (data_streams, feature_function):
		return np.array(np.nan_to_num([
			[feature_function(window) for window in stream]
				for stream in data_streams
		]))
