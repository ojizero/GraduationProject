import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


if __name__ == '__main__':
	import numpy as np
	from sklearn.naive_bayes import GaussianNB
	from sklearn.model_selection import train_test_split

	from utils.dataset_handler import DatasetHandler

	dataset = DatasetHandler.from_csv_file(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/proper.dataset.dump.csv', delimiter=',')

	labels, data = dataset.as_arrays()

	# # the straighforward way is futile as it would overfit and produce very poor results
	# # split training and testing data
	# training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels)

	# # classifier object
	# gaussian_nb_clf = GaussianNB()
	# gaussian_nb_clf.fit(training_data, training_labels)

	# testing  = gaussian_nb_clf.predict(testing_data)
	# accuracy = np.average(testing == testing_labels)

	# print('accuracy =', accuracy)
