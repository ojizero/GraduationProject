import numpy as np
from itertools import zip_longest
from collections import Iterable
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif as anova_score


def grouper (iterable, n, fillvalue=None):
	''' Collect data into fixed-length chunks or blocks '''
	args = [iterable] * n
	return zip_longest(*args, fillvalue=fillvalue)

def pipeline (data, *transformers):
	for transformer in transformers:
		data = transformer(data)
	return data

# case if logical to have one nan at end of vector not handled
def _interpolate_nan_linear (input_):
	while np.any(np.isnan(input_)):
		indices = np.where(np.logical_not(np.isnan(input_)))[0]

		for ix, i in enumerate(indices):
			if ix == 0:
				continue

			dif = i - indices[ix-1]
			if dif > 1:
				index = (i + indices[ix-1]) // 2
				input_[index] = (input_[i] + input_[indices[ix-1]]) / 2
				break

	return input_

def _linear_normalizer (data, constraint, discrete=True):
	if not isinstance(data, np.ndarray):
		data = np.array(data)

	max_ = len(data)

	ret = np.array([np.nan] * constraint, dtype=data.dtype)

	scale_operator = lambda a: a * (constraint - 1) // (max_ - 1) if discrete else a * (constraint - 1) / (max_ - 1)

	for index, value in enumerate(data):
		ret[int(scale_operator(index))] = value

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

def accuracy_beahviour (data, labels, classifier, score_function=anova_score):
	'''
		Generator for the behaviour of given classifier on given data, forall
		N in range(number of features) top features
	'''
	# split training and testing data
	training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels)
	for N in range(1, data.shape[1]):
		# feature selection object, using ANOVA F-value for scoring
		selector = SelectKBest(score_function, N)
		# the dataset using a sample of the features (top N features)
		selector.fit(data, labels)

		training_subset = selector.transform(training_data)
		testing_subset  = selector.transform(testing_data)

		# classifier object
		classifier_instance = classifier()
		# train on training subset
		classifier_instance.fit(training_subset, training_labels)
		# test on teseting subset
		testing  = classifier_instance.predict(testing_subset)
		accuracy = np.average(testing == testing_labels)

		yield accuracy
		print('accuracy for %s is %s' % (N, accuracy))
