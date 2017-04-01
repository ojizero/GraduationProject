import numpy as np
from math import sqrt

## data is read row, col
## data [:,0] -> all of the first column
data = np.genfromtxt('/Users/oji/Workspace/Self/GraduationProject/echoed', delimiter=',')

def euclidean_magnitude (data_point):
	return sqrt(sum(map(lambda d: d**2, data_point)))

def intensity_calculator (data_array, gyro_ratio=1.0, accl_ratio=1.0, magnitude=euclidean_magnitude):
	gyro_data = np.array(list(map(magnitude, data_array[:,0:3])))
	accl_data = np.array(list(map(magnitude, data_array[:,3:6])))
	return ((gyro_ratio * gyro_data) + (accl_ratio * accl_data))

intensities = intensity_calculator(data)
