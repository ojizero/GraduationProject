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
			yield from _spreader('%s_%s_' % (key, index), value)
		else:
			yield ('%s_%s' % (key, index), value)

_noop = lambda _ : _

def flatten_key_val_vector (vector, prefix='', data_modifier=_noop):
	for element in vector:
		if isinstance(element, tuple):
			key, val = element
			assert isinstance(key, (str, bytes)), 'keys must be strings'

			val = data_modifier(val)
			assert not isinstance(val, (str, bytes)), 'values are not fucking strings @'

			if isinstance(val, dict):
				yield from flatten_key_val_vector(val.items(), '%s%s_' % (prefix, key), data_modifier)
			elif isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
				spread = _spreader('%s%s' % (prefix, key), val)
				yield from flatten_key_val_vector(spread, '%s%s_' % (prefix, key), data_modifier)
			else:
				yield '%s%s' % (prefix, key), val
		else:
			element = data_modifier(element)

			if isinstance(element, dict):
				yield from flatten_key_val_vector(element.items(), prefix, data_modifier)
			elif isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
				yield from flatten_key_val_vector(element, prefix, data_modifier)
			else:
				yield element

def accuracy_beahviour (
	data, labels, classifier, clf_ops={}, score_function=anova_score, epislon=0.05, ksi=0.10, rep=5, step=1
):
	'''
		Generator for the behaviour of given classifier on given data, forall
		N in range(number of features) top features
	'''
	# split training and testing data
	prev_acc, prev_cnt = 0, 0

	training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels)
	for N in range(1, data.shape[1], step):
		# feature selection object, using ANOVA F-value for scoring
		selector = SelectKBest(score_function, N)
		# the dataset using a sample of the features (top N features)
		selector.fit(data, labels)

		training_subset = selector.transform(training_data)
		testing_subset  = selector.transform(testing_data)

		# classifier object
		classifier_instance = classifier(**clf_ops)
		# train on training subset
		classifier_instance.fit(training_subset, training_labels)
		# test on teseting subset
		testing  = classifier_instance.predict(testing_subset)
		accuracy = np.average(testing == testing_labels)

		yield accuracy
		print('accuracy for %s is %s' % (N, accuracy))

		# this code block is to avoid the segfault that occurs on the code in this case
		if abs(prev_acc - accuracy) <= epislon and accuracy <= ksi:
			prev_cnt += 1

		if prev_cnt >= rep:
			raise StopIteration('enough')

		prev_acc = accuracy

def complex_flattener (elem):
	if isinstance(elem, (np.complex, complex)):
		return {
			'real': elem.real,
			'imag': elem.imag
		}
	else:
		return elem
