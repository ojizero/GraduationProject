import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	from sklearn.neural_network import MLPClassifier

	from utils.helpers import accuracy_beahviour
	from utils.dataset_handler import DatasetHandler

	dataset = DatasetHandler.from_csv_file(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/proper.dataset.accel.only.dump.csv', delimiter=',')

	labels, data = dataset.as_arrays()

	accuracies = [*accuracy_beahviour(data, labels, MLPClassifier, clf_ops={'hidden_layer_sizes' : (1000, 500, 250, 100)})]
	plt.plot(accuracies)
	plt.savefig('/Users/oji/Desktop/accel_svm.png')