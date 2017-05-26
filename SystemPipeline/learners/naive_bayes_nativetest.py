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

	dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/data', delimiter=',')

	training_labels, training_data = dataset.as_arrays()

	dataset = DatasetHandler.from_csv_directory(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/native_data', delimiter=',')

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