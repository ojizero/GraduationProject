import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


if __name__ == '__main__':
	import numpy as np
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split

	from utils.dataset_handler import DatasetHandler

	dataset = DatasetHandler.from_csv_file(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/dataset.new.rawdata.noseg.csv', delimiter=',')

	labels, data = dataset.as_arrays()
	# convert data to it's absolute values, since sklearn's RandomForestClassifier only handles Floats
	# also apparently sklearn's RandomForestClassifier for some god forsaken reason only internally uses Float32 !
	data = np.float32(np.absolute(data))

	assert not np.any(np.isnan(data)) and np.all(np.isfinite(data)) and np.all(data <= np.finfo(np.float32).max) and np.all(np.finfo(np.float32) >= np.finfo(np.float32).min), 'fml'

	# split training and testing data
	training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels)

	# classifier object
	random_forest_clf = RandomForestClassifier()
	random_forest_clf.fit(training_data, training_labels)

	testing  = random_forest_clf.predict(testing_data)
	accuracy = np.average(testing == testing_labels)

	print('accuracy =', accuracy)
