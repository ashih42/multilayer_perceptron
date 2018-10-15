from TrainingDataParser import TrainingDataParser
from Description import Description
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import string
import os

'''
Assumptions:
- 32 Columns
- Colume 0 is Patient ID		=> Patient ID (unused)
- Column 1 is LABEL {'M', 'B'}	=> LABEL { 'M', 'B' }
- Column 2...31					=> FEATURES { float }, 30 features total
'''

'''
m		=> sample_count
self.X	=> m x 30 matrix, where each row represents one datum with 30 features
self.Y	=> column vector containing {1.0, 0.0}, where 1.0 indicates 'M'
'''

class DataAnalyzer:
	__FEATURE_COUNT = 30
	__GRAPHING_ALPHA = 0.5
	__HISTOGRAM_DIRECTORY = 'Histograms/'
	__SCATTERPLOT_DIRECTORY = 'ScatterPlots/'
	__PAIRPLOT_DIRECTORY = 'PairPlots/'
	
	def __init__(self, filename):
		parser = TrainingDataParser(filename)
		data = np.array(parser.data, dtype=object)

		self.__M_data = data[ data[:, 0] == 'M' ]
		self.__B_data = data[ data[:, 0] == 'B' ]
		self.__M_data = self.__M_data[:, 1:]
		self.__B_data = self.__B_data[:, 1:]
		self.__M_data = self.__M_data.astype(float)
		self.__B_data = self.__B_data.astype(float)

		self.sample_count = data.shape[0]
		self.X = data[:, 1:]
		self.X = self.X.astype(float)

		Y_literal = data[:, 0]
		Y_literal = Y_literal.reshape(self.sample_count, 1)
		self.Y = Y_literal == 'M'
		self.Y = self.Y.astype(float)

		self.__all_descriptions = []
		for column in self.X.T:
			self.__all_descriptions.append(Description(column))
		self.__M_descriptions = []
		for column in self.__M_data.T:
			self.__M_descriptions.append(Description(column))
		self.__B_descriptions = []
		for column in self.__B_data.T:
			self.__B_descriptions.append(Description(column))

	def get_mean_list(self):
		return [ self.__all_descriptions[i].mean for i in range(len(self.__all_descriptions)) ]

	def get_stdev_list(self):
		return [ self.__all_descriptions[i].standard_deviation for i in range(len(self.__all_descriptions)) ]

	def show_pair_plot(self):
		try:
			os.stat(self.__PAIRPLOT_DIRECTORY)
		except:
			os.mkdir(self.__PAIRPLOT_DIRECTORY)
		fig = plt.figure(figsize=(40, 40))
		for i in range(0, DataAnalyzer.__FEATURE_COUNT):
			for j in range(0, DataAnalyzer.__FEATURE_COUNT):
				self.__generate_subplot(i, j)
		fig.tight_layout()
		filename = self.__PAIRPLOT_DIRECTORY + 'Pair Plot'
		plt.savefig(filename)
		print('Saved pair plot at ' + Fore.BLUE + filename + Fore.RESET)

	def __generate_subplot(self, i, j):
		plt.subplot(DataAnalyzer.__FEATURE_COUNT, DataAnalyzer.__FEATURE_COUNT, i * DataAnalyzer.__FEATURE_COUNT + j + 1)
		if i == j:
			plt.hist(self.__M_data[:,i].tolist(), label='Malignant', color='red',
				fill=True, alpha=DataAnalyzer.__GRAPHING_ALPHA)
			plt.hist(self.__B_data[:,i].tolist(), label='Benign', color='green',
				fill=True, alpha=DataAnalyzer.__GRAPHING_ALPHA)
		else:
			plt.scatter(self.__M_data[:, i].tolist(), self.__M_data[:, j].tolist(),
				label='Malignant', color='red', alpha=DataAnalyzer.__GRAPHING_ALPHA,)
			plt.scatter(self.__B_data[:, i].tolist(), self.__B_data[:, j].tolist(),
				label='Benign', color='green', alpha=DataAnalyzer.__GRAPHING_ALPHA,)
		plt.xticks([], [])
		plt.yticks([], [])
		if i == 0:
			plt.title(self.__get_feature_name(j))
		if j == 0:
			plt.ylabel(self.__get_feature_name(i))
		print('Generating subplot %02d, %02d ' % (i + 1, j + 1))

	def show_scatter_plots(self):
		try:
			os.stat(self.__SCATTERPLOT_DIRECTORY)
		except:
			os.mkdir(self.__SCATTERPLOT_DIRECTORY)
		for i in range(0, DataAnalyzer.__FEATURE_COUNT):
			for j in range(i + 1, DataAnalyzer.__FEATURE_COUNT):
				self.__generate_scatter_plot(i, j)

	def __generate_scatter_plot(self, i, j):
		plt.clf()
		plt.scatter(self.__M_data[:, i].tolist(), self.__M_data[:, j].tolist(),
			label='Malignant', color='red', alpha=DataAnalyzer.__GRAPHING_ALPHA)
		plt.scatter(self.__B_data[:, i].tolist(), self.__B_data[:, j].tolist(),
			label='Benign', color='green', alpha=DataAnalyzer.__GRAPHING_ALPHA)
		feature_x_name = self.__get_feature_name(i)
		feature_y_name = self.__get_feature_name(j)
		plt.xlabel(feature_x_name)
		plt.ylabel(feature_y_name)
		plt.title('%s vs %s' % (feature_y_name, feature_x_name))
		plt.legend(loc='upper right')
		filename = self.__SCATTERPLOT_DIRECTORY + '%02d, %02d : %s vs %s' % (
			i + 1, j + 1, feature_y_name, feature_x_name)
		plt.savefig(filename)
		print('Saved scatter plot at ' + Fore.BLUE + filename + Fore.RESET)

	def show_histograms(self):
		try:
			os.stat(self.__HISTOGRAM_DIRECTORY)
		except:
			os.mkdir(self.__HISTOGRAM_DIRECTORY)
		for i in range(DataAnalyzer.__FEATURE_COUNT):
			self.__generate_histogram(i)

	def __generate_histogram(self, i):
		plt.clf()
		plt.hist(self.__M_data[:,i].tolist(), label='Malignant', color='red',
			fill=True, alpha=DataAnalyzer.__GRAPHING_ALPHA)
		plt.hist(self.__B_data[:,i].tolist(), label='Benign', color='green',
			fill=True, alpha=DataAnalyzer.__GRAPHING_ALPHA)
		feature_name = self.__get_feature_name(i)
		plt.xlabel(feature_name)
		plt.ylabel('Count')
		plt.title('Distribution of ' + feature_name)
		plt.legend(loc='upper right')
		filename = self.__HISTOGRAM_DIRECTORY + '%02d : %s' % (i + 1, feature_name)
		plt.savefig(filename)
		print('Saved histogram at ' + Fore.BLUE + filename + Fore.RESET)

	def describe(self):
		print(Style.BRIGHT + Fore.CYAN + 'ALL PATIENTS:' + Style.RESET_ALL + Fore.RESET)
		self.__print_descriptions(self.__all_descriptions)
		print()
		print(Style.BRIGHT + Fore.CYAN + '[M] MALIGNANT:' + Style.RESET_ALL + Fore.RESET)
		self.__print_descriptions(self.__M_descriptions)
		print()
		print(Style.BRIGHT + Fore.CYAN + '[B] BENIGN:' + Style.RESET_ALL + Fore.RESET)
		self.__print_descriptions(self.__B_descriptions)
		print()
		
	def __print_descriptions(self, descriptions):
		# print feature headers
		print(Style.BRIGHT + Fore.BLUE + '\t\t\t', end='')
		for i in range(30):
			print('%-20.20s \t' % self.__get_feature_name(i), end='')
		print(Style.RESET_ALL + Fore.RESET)
		# print row of COUNT
		print(Style.BRIGHT + 'COUNT\t\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.count, end='')
		print()
		# print row of MEAN
		print(Style.BRIGHT + 'MEAN\t\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.mean, end='')
		print()
		# print row of STANDARD DEVIATION
		print(Style.BRIGHT + 'STANDARD DEVIATION\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.standard_deviation, end='')
		print()
		# print row of VARIANCE
		print(Style.BRIGHT + 'VARIANCE\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.variance, end='')
		print()
		# print row of MINIMUM
		print(Style.BRIGHT + 'MINIMUM\t\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.min, end='')
		print()
		# print row of 25 PERCENTILE
		print(Style.BRIGHT + '25 PERCENTILE\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.percentile_25, end='')
		print()
		# print row of 50 PERCENTILE
		print(Style.BRIGHT + '50 PERCENTILE\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.percentile_50, end='')
		print()
		# print row of 75 PERCENTILE
		print(Style.BRIGHT + '75 PERCENTILE\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.percentile_75, end='')
		print()
		# print row of MAXIMUM
		print(Style.BRIGHT + 'MAXIMUM\t\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.max, end='')
		print()
		# print row of MODE
		print(Style.BRIGHT + 'MODE\t\t\t' + Style.RESET_ALL, end='')
		for description in descriptions:
			print('%-20.6f \t' % description.mode, end='')
		print()

	def __get_feature_name(self, i):
		return 'Feature %02d' % (i + 1)
