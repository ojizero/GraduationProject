import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	from sklearn.naive_bayes import GaussianNB

	from utils.helpers import accuracy_beahviour
	from utils.dataset_handler import DatasetHandler

	dataset = DatasetHandler.from_csv_file(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/proper.dataset.accel.only.dump.csv', delimiter=',')

	labels, data = dataset.as_arrays()

	accuracies = [*accuracy_beahviour(data, labels, GaussianNB)]
	plt.plot(accuracies)
	plt.savefig('/Users/oji/Desktop/remade_accel_vs_numfeatures.png')