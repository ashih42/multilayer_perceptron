import numpy as np

class Description:

	def __init__(self, array):
		self.count = float(array.size)
		self.mean = np.sum(array) / self.count
		self.variance = np.sum((array - self.mean) ** 2) / self.count
		self.standard_deviation = self.variance ** 0.5
		# compute statistical measures which require sorted data
		sorted_array = np.sort(array)
		self.percentile_25 = self.__compute_percentile(sorted_array, 25)
		self.percentile_50 = self.__compute_percentile(sorted_array, 50)
		self.percentile_75 = self.__compute_percentile(sorted_array, 75)
		self.min = sorted_array[0]
		self.max = sorted_array[-1]
		# compute MODE
		values, counts = np.unique(array, return_counts=True)
		counts = counts.tolist()
		mode_index = counts.index(max(counts))
		self.mode = values[mode_index]

	def __compute_percentile(self, sorted_array, percent):
		index = percent / 100 * self.count
		if index.is_integer():
			index = int(index)
			return (sorted_array[index] + sorted_array[index - 1]) / 2
		else:
			index = int(index)
			return sorted_array[index]

	def get_dummy_value(self):
		return self.mean
