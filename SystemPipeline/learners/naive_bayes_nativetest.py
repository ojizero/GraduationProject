import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	from sklearn.naive_bayes import GaussianNB
	from sklearn.feature_selection import SelectKBest, f_classif as anova_score

	from utils.helpers import accuracy_beahviour
	from utils.dataset_handler import DatasetHandler

	def vector_maker (data, **kwargs):
		def _slicer (splicer, data, min_len=20):
			intensity_total = splicer.smooth_intensities(
				sum([
					splicer.measure_intensity(stream) for stream in data
				])
			)
			silence_regions = splicer.silence_segments(intensity_total)

			region_couples = [*zip_longest(silence_regions, silence_regions[1:])]
			if region_couples[-1][1] is None:
				region_couples[-1] = (region_couples[-1][0], (data.shape[1], data.shape[1]))

			for r in region_couples:
				if not (r[0][1] < r[1][0]):
					raise AssertionError('reversed regions')

			major_index = np.argmax(np.array([
				r[1][0] - r[0][1] for r in region_couples
			]))

			major_active = region_couples[major_index]

			# for keeping track of what happens .. could be obsolete ?
			print(region_couples)
			print(major_active)

			active_start = major_active[0][1]
			active_end   = major_active[1][0]

			return active_start, active_end

		# instance responsible for segmentation
		splicer = Splicer(threshold=80)

		active_start, active_end = _slicer(splicer, data)

		active_region_data = data[:,active_start:active_end,3:6]

		print(data.shape, active_region_data.shape)

		return FeaturesTransformer.transform(data=active_region_data, **kwargs)

	dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data', delimiter=',', vector_maker=vector_maker, overlap=0.0)

	training_labels, training_data = dataset.as_arrays()

	dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/native_data', delimiter=',', vector_maker=vector_maker, overlap=0.0)

	testing_labels, testing_data = dataset.as_arrays()

	selector = SelectKBest(anova_score, 550)
	selector.fit(training_data, training_labels)

	training_data = selector.transform(training_data)
	testing_data  = selector.transform(testing_data)

	# classifier object
	classifier_instance = GaussianNB()
	# train on training subset
	classifier_instance.fit(training_data, training_labels)
	# test on teseting subset
	testing  = classifier_instance.predict(testing_data)
	accuracy = np.average(testing == testing_labels)

	print(accuracy)
