import numpy as np
from itertools import zip_longest
from collections import Iterable


def grouper (iterable, n, fillvalue=None):
	''' Collect data into fixed-length chunks or blocks '''
	args = [iterable] * n
	return zip_longest(*args, fillvalue=fillvalue)

def pipeline (data, *transformers):
	for transformer in transformers:
		data = transformer(data)
	return data

def _interpolate_nan_linear (input):
	if not np.any(np.isnan(input)):
		return input

	indices = np.where(np.logical_not(np.isnan(input)))
	for ix, i in enumerate(indices):
		if ix == 0:
			continue
		if ix == len(indices) - 1:
			break

		dif = i - indices[ix-1]
		if dif > 1:
			index = (i + indices[ix-1]) // 2
			input[index] = (input[i] + input[indices[ix-1]]) / 2

	return _interpolate_nan_linear(input)

# stupid implementation >_<
def _linear_normalizer (data, constraint, discrete=True):
	if not isinstance(data, np.array):
		data = np.array(data)

	operator = lambda a, b: a * b // a if discrete else a * b / a

	ret = np.array([np.nan] * constraint)
	for index, value in enumerate(data):
		ret[operator(index, constraint)] = value

	return _interpolate_nan_linear(ret)

def _logarithmic_normalizer (data, constraint):
	pass

def array_normalizer (array, constraint_length=10, normalization_method=_linear_normalizer):
	return normalization_method(array, constraint_length)

def _flatten (*l):
	for el in l:
		if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
			yield from _flatten(el)
		else:
			yield el

def flatten_vector (*args):
	for arg in args:
		if isinstance(arg, tuple):
			key, val = arg
			assert isinstance(key, str), 'keys must be strings'

			if isinstance(val, Iterable):
				flattened = flatten_vector(*val)
				yield from ((key + index, value) for index, value in enumerate(flattened))
			else:
				yield arg
		else:
			yield from _flatten(*arg)

