import sys
# Add project to path, to be able to import modules from it
sys.path.append('/Users/oji/Workspace/Self/GraduationProject/SystemPipeline')


if __name__ == '__main__':
	import numpy as np
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import cross_val_score

	from utils.dataset_handler import DatasetHandler

	dataset = DatasetHandler.from_csv_file(path='/Users/oji/Workspace/Self/GraduationProject/SystemPipeline/floats.dataset.accel.only.withnative.dump.csv', delimiter=',')

	labels, data = dataset.as_arrays()

	# classifier object
	random_forest_clf = RandomForestClassifier()

	# perform cross validation scoring
	# use K-folds, split data to 5 groups
	scores = cross_val_score(random_forest_clf, data, labels, cv=5)
	# get average accuracy and range of error
	avg_acc, std_acc = scores.mean(), scores.std() * 2

	print('%s[±%s]' % (avg_acc, std_acc))
