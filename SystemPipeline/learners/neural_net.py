import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	from sklearn.neural_network import MLPClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.feature_selection import SelectKBest, f_classif as anova_score

	from utils.helpers import accuracy_beahviour
	from utils.dataset_handler import DatasetHandler

	dataset = DatasetHandler.from_csv_file(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/floats.dataset.accel.only.withnative.dump.csv', delimiter=',')

	labels, data = dataset.as_arrays()

	print('done reading')

	# data = data[:,np.any(data != data[0])] # remove constant columns

	selector = SelectKBest(anova_score, 800)
	data = selector.fit_transform(data, labels)

	print('done modifying dataset', data.shape)

	training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels)

	# accuracies = [*accuracy_beahviour(data, labels, MLPClassifier, clf_ops={'hidden_layer_sizes' : (1000, 500, 250, 100)})]
	# plt.plot(accuracies)
	# plt.savefig('/Users/oji/Desktop/accel_neuralnet_1000_500_250_100.png')

	clf = MLPClassifier(hidden_layer_sizes=(1000, 500, 250, 100))
	clf.fit(training_data, training_labels)

	print('accuracy', clf.score(testing_data, testing_labels))