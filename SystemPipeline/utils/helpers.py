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
	# print(input)
	# exit(10)
	while np.any(np.isnan(input)):
		indices = np.where(np.logical_not(np.isnan(input)))
		for ix, i in enumerate(indices):
			if ix == 0:
				continue

			dif = i - indices[ix-1]
			if dif > 1:
				index = (i + indices[ix-1]) // 2
				input[index] = (input[i] + input[indices[ix-1]]) / 2

	return input

def _linear_normalizer (data, constraint, discrete=True):
	if not isinstance(data, np.ndarray):
		data = np.array(data)

	max_ = len(data)

	scale_operator = lambda a, b: a * b // max_ if discrete else a * b / max_

	ret = np.array([np.nan] * constraint, dtype=data.dtype)
	for index, value in enumerate(data):
		ret[int(scale_operator(index, constraint))] = value

	return _interpolate_nan_linear(ret)

def _logarithmic_normalizer (data, constraint):
	pass

def array_length_normalizer (array, constraint_length=10, normalization_method=_linear_normalizer):
	return normalization_method(array, constraint_length)

def flatten (l):
	for el in l:
		if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
			yield from flatten(el)
		else:
			yield el

def _spreader (key, val):
	for index, value in enumerate(val):
		if isinstance(value, tuple) and len(value) == 2:
			yield value
		elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
			yield from _spreader(*value)
		else:
			yield ('%s_%s' % (key, index), value)

def flatten_key_val_vector (*args):
	for arg in args:
		if isinstance(arg, tuple):
			key, val = arg
			assert isinstance(key, str), 'keys must be strings'

			if isinstance(val, dict):
				yield from flatten_key_val_vector(*val.items())
			elif isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
				flattened = flatten_key_val_vector(*val)
				yield from _spreader(key, flattened)
			else:
				yield arg
		else:
			if isinstance(arg, Iterable) and not isinstance(arg, (str, bytes)):
				yield from flatten_key_val_vector(*arg)
			else:
				yield arg

