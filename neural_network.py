from data_analyzer import DataAnalyzer
from validation_data_parser import ValidationDataParser
from param_parser import ParamParser

from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import math
import os

'''
Layer 1:	30 + 1 nodes
Layer 2:	30 + 1 nodes
Layer 3:	30 + 1 nodes
Layer 4:	1 node => is Malignant?

X:		m x 30
Y:		1 x 30

Theta_1: 30 x 31
Theta_2: 30 x 31
Theta_3: 1 x 31
'''

class NeuralNetwork:
	__SIGMOID = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
	__SHOW_PLOTS = os.getenv('MP_PLOTS') == 'TRUE'

	def __init__(self, param_filename=None):
		if param_filename is None:
			self.__initialize_random_theta()
		else:
			self.__load_param_from_file(param_filename)

	def __initialize_random_theta(self):
		np.random.seed(42)
		self.Theta_1 = np.random.rand(30, 31)
		self.Theta_2 = np.random.rand(30, 31)
		self.Theta_3 = np.random.rand(1, 31)

	def __load_param_from_file(self, filename):
		param_parser = ParamParser(filename)
		self.mean_list = param_parser.mean_list
		self.stdev_list = param_parser.stdev_list
		self.Theta_1 = np.array(param_parser.theta_1_list)
		self.Theta_2 = np.array(param_parser.theta_2_list)
		self.Theta_3 = np.array(param_parser.theta_3_list)
		print('Loaded model parameters from ' + Fore.CYAN + filename + Fore.RESET)
	
	def __apply_feature_scaling(self, array):
		num_columns = array.shape[1]
		assert num_columns == len(self.mean_list)
		assert num_columns == len(self.stdev_list)
		
		for i in range(num_columns):
			mean = self.mean_list[i]
			stdev = self.stdev_list[i]
			array[:, i] = (array[:, i] - mean) / stdev
		return array

	def __init_plots(self):
		# initialize 4 subplots
		self.fig = plt.figure(figsize=(10, 10))
		self.fig.tight_layout()
		self.ax_train_loss = self.fig.add_subplot(2, 2, 1)
		self.ax_train_error = self.fig.add_subplot(2, 2, 2)
		self.ax_valid_loss = self.fig.add_subplot(2, 2, 3)
		self.ax_valid_error = self.fig.add_subplot(2, 2, 4)
		# initialize lists for the plotted data
		self.epoch_list = []
		self.train_loss_list = []
		self.train_error_list = []
		self.valid_loss_list = []
		self.valid_error_list = []

	def __update_plots(self):
		# update Training Loss over Epoch
		self.ax_train_loss.clear()
		self.ax_train_loss.plot(self.epoch_list, self.train_loss_list)
		self.ax_train_loss.fill_between(self.epoch_list, 0, self.train_loss_list, facecolor='blue', alpha=0.5)
		self.ax_train_loss.set_xlabel('Epoch')
		self.ax_train_loss.set_ylabel('Loss')
		self.ax_train_loss.set_title('Training Loss at Epoch %d: %.3f' % (self.epoch_list[-1], self.train_loss_list[-1]))
		# update Training Error over Epoch
		self.ax_train_error.clear()
		self.ax_train_error.plot(self.epoch_list, self.train_error_list)
		self.ax_train_error.fill_between(self.epoch_list, 0, self.train_error_list, facecolor='cyan', alpha=0.5)
		self.ax_train_error.set_xlabel('Epoch')
		self.ax_train_error.set_ylabel('Error')
		self.ax_train_error.set_title('Training Error at Epoch %d: %d' % (self.epoch_list[-1], self.train_error_list[-1]))
		# update Validation Loss over Epoch
		self.ax_valid_loss.clear()
		self.ax_valid_loss.plot(self.epoch_list, self.valid_loss_list)
		self.ax_valid_loss.fill_between(self.epoch_list, 0, self.valid_loss_list, facecolor='red', alpha=0.5)
		self.ax_valid_loss.set_xlabel('Epoch')
		self.ax_valid_loss.set_ylabel('Loss')
		self.ax_valid_loss.set_title('Validation Loss at Epoch %d: %.3f' % (self.epoch_list[-1], self.valid_loss_list[-1]))
		# update Validation Error over Epoch
		self.ax_valid_error.clear()
		self.ax_valid_error.plot(self.epoch_list, self.valid_error_list)
		self.ax_valid_error.fill_between(self.epoch_list, 0, self.valid_error_list, facecolor='magenta', alpha=0.5)
		self.ax_valid_error.set_xlabel('Epoch')
		self.ax_valid_error.set_ylabel('Error')
		self.ax_valid_error.set_title('Validation Error at Epoch %d: %d' % (self.epoch_list[-1], self.valid_error_list[-1]))
		plt.pause(0.001)

	def __should_stop(self):
		if len(self.valid_loss_list) >= 1:
			return self.valid_loss_list[-1] < 0.0799
		return False

	def __init_training_data(self, training_filename):
		data_analyzer = DataAnalyzer(training_filename)
		self.mean_list = data_analyzer.get_mean_list()
		self.stdev_list = data_analyzer.get_stdev_list()
		# feature scale, and add column of 1s to X
		self.training_X = data_analyzer.X
		self.training_X = self.__apply_feature_scaling(self.training_X)
		self.training_X = np.c_[np.ones(self.training_X.shape[0]), self.training_X]
		self.training_Y = data_analyzer.Y

	def __init_validation_data(self, validation_filename):
		validation_data_parser = ValidationDataParser(validation_filename, self.mean_list)
		self.validation_id_list = validation_data_parser.patient_id_list

		data = np.array(validation_data_parser.data, dtype=object)
		self.validation_X = data[:, 1:]
		self.validation_X = self.validation_X.astype(float)
		# feature scale, and add column of 1s to X
		self.validation_X = self.__apply_feature_scaling(self.validation_X)
		self.validation_X = np.c_[np.ones(self.validation_X.shape[0]), self.validation_X]

		Y_literal = data[:, 0]
		Y_literal = Y_literal.reshape(data.shape[0], 1)
		self.validation_Y = Y_literal == 'M'
		self.validation_Y = self.validation_Y.astype(float)

	def predict(self, data_filename):
		self.__init_validation_data(data_filename)

		print(Style.BRIGHT + Fore.RED + 'ID\t\t% M\t\t% B\t\tPrediction\tActual\t\tCorrect?' + Style.RESET_ALL + Fore.RESET)

		X = self.validation_X
		Y = self.validation_Y
		m = X.shape[0]

		for i in range(m):

			a_1 = X[i, :]
			a_1 = a_1.reshape(1, a_1.shape[0])
			y = Y[i, :]
			y = y.reshape(1, y.shape[0])

			# FORWARD PROPAGATION

			a_2 = self.__SIGMOID(a_1 @ self.Theta_1.T)
			a_2 = np.c_[1, a_2]

			a_3 = self.__SIGMOID(a_2 @ self.Theta_2.T)
			a_3 = np.c_[1, a_3]

			a_4 = self.__SIGMOID(a_3 @ self.Theta_3.T)

			prob_malignant = a_4
			prob_benign = 1 - a_4

			prediction = 'M' if prob_malignant >= 0.5 else 'B'
			actual = 'M' if y == 1.0 else 'B'
			is_correct = (Fore.GREEN + 'Yes' + Fore.RESET) if prediction == actual else (Fore.RED + 'No' + Fore.RESET)

			print('%10s\t%5.2f %%\t\t%5.2f %%\t\t%s\t\t%s\t\t%s' % (self.validation_id_list[i], prob_malignant * 100, prob_benign * 100, prediction, actual, is_correct))

		print()
		prediction_loss, prediction_error = self.__compute_cost_error(X, Y)
		print('Prediction Loss: %.5f, Prediction Error: %d' % (prediction_loss, prediction_error))

	def __separate_data(self, data_filename):
		data_analyzer = DataAnalyzer(data_filename)
		self.mean_list = data_analyzer.get_mean_list()
		self.stdev_list = data_analyzer.get_stdev_list()
		
		# feature scale, and add column of 1s to X
		all_X = data_analyzer.X
		all_X = self.__apply_feature_scaling(all_X)
		all_X = np.c_[np.ones(all_X.shape[0]), all_X]
		all_Y = data_analyzer.Y
		all_data = np.c_[all_X, all_Y]

		np.random.shuffle(all_data)
		split_row_index = int(all_data.shape[0] * 0.8)	# top 80% of rows will be for training

		training_data = all_data[:split_row_index, :]
		validation_data = all_data[split_row_index:, :]

		self.training_X = training_data[:, :-1]
		self.training_Y = training_data[:, -1]
		self.training_Y = self.training_Y.reshape(self.training_Y.shape[0], 1)

		self.validation_X = validation_data[:, :-1]
		self.validation_Y = validation_data[:, -1]
		self.validation_Y = self.validation_Y.reshape(self.validation_Y.shape[0], 1)

	def separate_data_and_train(self, data_filename, do_online):
		self.__separate_data(data_filename)
		self.__init_plots()
		self.__start_training(do_online)

	def train(self, training_filename, validation_filename, do_online):
		self.__init_training_data(training_filename)
		self.__init_validation_data(validation_filename)
		self.__init_plots()
		self.__start_training(do_online)

	def __start_training(self, do_online):
		if do_online:
			self.V_1 = np.zeros(self.Theta_1.shape)
			self.V_2 = np.zeros(self.Theta_2.shape)
			self.V_3 = np.zeros(self.Theta_3.shape)
			self.momentum_constant = 0.9
			self.lambda_reg = 1
			self.epoch_limit = 100000
			self.learning_rate = 0.00003
		else:
			self.lambda_reg = 0
			self.epoch_limit = 100000
			self.learning_rate = 0.3

		for epoch_num in range(self.epoch_limit):
			train_loss, train_error = self.__compute_cost_error(self.training_X, self.training_Y, add_reg=True)
			valid_loss, valid_error = self.__compute_cost_error(self.validation_X, self.validation_Y)
			self.epoch_list.append(epoch_num)
			self.train_loss_list.append(train_loss)
			self.train_error_list.append(train_error)
			self.valid_loss_list.append(valid_loss)
			self.valid_error_list.append(valid_error)
			print('Epoch: %d, Training Loss: %.5f, Training Error: %d, Validation Loss: %.5f, Validation Error: %d' % \
				(epoch_num, train_loss, train_error, valid_loss, valid_error))

			if NeuralNetwork.__SHOW_PLOTS:
				self.__update_plots()
			if self.__should_stop():
				break
			if do_online:
				self.__online_update_theta_from_gradient()
			else:
				self.__batch_update_theta_from_gradient()

		self.__update_plots()
		plt.show()
		filename = 'PerformancePlots'
		plt.savefig(filename)
		print('Saved plots as ' + Fore.BLUE + ('%s.png' % filename) + Fore.RESET)

	def __write_theta(self, param_file, theta):
		# write theta dimensions in one row, followed by its values per row
		param_file.write('%d %d\n' % (theta.shape[0], theta.shape[1]))
		for row in theta:
			for num in row:
				param_file.write('%f ' % num)
			param_file.write('\n')

	def save_param_to_file(self, param_filename):
		with open(param_filename, 'w') as param_file:
			# write 1 row of all mean values
			for mean in self.mean_list:
				param_file.write('%f ' % mean)
			param_file.write('\n')
			# write 1 row of all standard deviation values
			for stdev in self.stdev_list:
				param_file.write('%f ' % stdev)
			param_file.write('\n')
			self.__write_theta(param_file, self.Theta_1)
			self.__write_theta(param_file, self.Theta_2)
			self.__write_theta(param_file, self.Theta_3)
			print('Saved model parameters in ' + Fore.CYAN + param_filename + Fore.RESET + '\n')

	def __online_update_theta_from_gradient(self):
		D_1 = np.zeros(self.Theta_1.shape)
		D_2 = np.zeros(self.Theta_2.shape)
		D_3 = np.zeros(self.Theta_3.shape)

		X = self.training_X
		Y = self.training_Y
		m = X.shape[0]

		for i in range(m):

			a_1 = X[i, :]
			a_1 = a_1.reshape(1, a_1.shape[0])
			y = Y[i, :]
			y = y.reshape(1, y.shape[0])

			# FORWARD PROPAGATION

			a_2 = self.__SIGMOID(a_1 @ (self.Theta_1.T - self.V_1.T))
			a_2 = np.c_[1, a_2]

			a_3 = self.__SIGMOID(a_2 @ (self.Theta_2.T - self.V_2.T))
			a_3 = np.c_[1, a_3]

			a_4 = self.__SIGMOID(a_3 @ (self.Theta_3.T - self.V_3.T))

			# BACK PROPAGATION

			d_4 = a_4 - y

			d_3 = (d_4 @ self.Theta_3) * (a_3 * (1 - a_3))
			D_3 += d_4.T @ a_3

			d_3 = d_3[:, 1:]
			d_2 = (d_3 @ self.Theta_2) * (a_2 * (1 - a_2))
			D_2 += d_3.T @ a_2

			d_2 = d_2[:, 1:]
			d_1 = d_2 @ self.Theta_1 * (a_1 * (1 - a_1))
			D_1 += d_2.T @ a_1

			# Add gradient contribution from Regularization
			Theta_1_reg = self.Theta_1.copy()
			Theta_2_reg = self.Theta_2.copy()
			Theta_3_reg = self.Theta_3.copy()
			Theta_1_reg[:, 0] = 0
			Theta_2_reg[:, 0] = 0
			Theta_3_reg[:, 0] = 0

			self.V_1 = self.momentum_constant * self.V_1 + self.learning_rate * (D_1 + self.lambda_reg * Theta_1_reg)
			self.V_2 = self.momentum_constant * self.V_2 + self.learning_rate * (D_2 + self.lambda_reg * Theta_2_reg)
			self.V_3 = self.momentum_constant * self.V_3 + self.learning_rate * (D_3 + self.lambda_reg * Theta_3_reg)

			self.Theta_1 -= self.V_1
			self.Theta_2 -= self.V_2
			self.Theta_3 -= self.V_3

	def __batch_update_theta_from_gradient(self):
		D_1 = np.zeros(self.Theta_1.shape)
		D_2 = np.zeros(self.Theta_2.shape)
		D_3 = np.zeros(self.Theta_3.shape)

		X = self.training_X
		Y = self.training_Y
		m = X.shape[0]

		for i in range(m):

			a_1 = X[i, :]
			a_1 = a_1.reshape(1, a_1.shape[0])
			y = Y[i, :]
			y = y.reshape(1, y.shape[0])

			# FORWARD PROPAGATION

			a_2 = self.__SIGMOID(a_1 @ self.Theta_1.T)
			a_2 = np.c_[1, a_2]

			a_3 = self.__SIGMOID(a_2 @ self.Theta_2.T)
			a_3 = np.c_[1, a_3]

			a_4 = self.__SIGMOID(a_3 @ self.Theta_3.T)

			# BACK PROPAGATION

			d_4 = a_4 - y

			d_3 = (d_4 @ self.Theta_3) * (a_3 * (1 - a_3))
			D_3 += d_4.T @ a_3

			d_3 = d_3[:, 1:]
			d_2 = (d_3 @ self.Theta_2) * (a_2 * (1 - a_2))
			D_2 += d_3.T @ a_2

			d_2 = d_2[:, 1:]
			d_1 = d_2 @ self.Theta_1 * (a_1 * (1 - a_1))
			D_1 += d_2.T @ a_1

		Grad_3 = D_3 / m
		Grad_2 = D_2 / m
		Grad_1 = D_1 / m

		# Add gradient contribution from Regularization
		Theta_1_reg = self.Theta_1.copy()
		Theta_2_reg = self.Theta_2.copy()
		Theta_3_reg = self.Theta_3.copy()
		Theta_1_reg[:, 0] = 0
		Theta_2_reg[:, 0] = 0
		Theta_3_reg[:, 0] = 0

		Grad_3 += self.lambda_reg / m * Theta_3_reg
		Grad_2 += self.lambda_reg / m * Theta_2_reg
		Grad_1 += self.lambda_reg / m * Theta_1_reg

		self.Theta_1 -= self.learning_rate * Grad_1
		self.Theta_2 -= self.learning_rate * Grad_2
		self.Theta_3 -= self.learning_rate * Grad_3

	# cost, also referred to as loss
	# error, meaning number of incorrect predictions made
	def __compute_cost_error(self, X, Y, add_reg=False):
		m = X.shape[0]

		A_2 = self.__SIGMOID(X @ self.Theta_1.T)
		A_2 = np.c_[np.ones(A_2.shape[0]), A_2]

		A_3 = self.__SIGMOID(A_2 @ self.Theta_2.T)
		A_3 = np.c_[np.ones(A_3.shape[0]), A_3]

		A_4 = self.__SIGMOID(A_3 @ self.Theta_3.T)

		cost = -1 / m * sum(Y * np.log(A_4) + (1 - Y) * np.log(1 - A_4))
		cost = cost.item()

		prediction = A_4 >= 0.5
		prediction = prediction.astype(float)
		classification_error = sum((prediction - Y) ** 2)
		classification_error = classification_error.item()

		# Add cost contribution from Regularization
		if add_reg:
			Theta_1_reg = self.Theta_1[:, 1:]
			Theta_2_reg = self.Theta_2[:, 1:]
			Theta_3_reg = self.Theta_3[:, 1:]

			cost_reg = sum(sum(Theta_1_reg ** 2)) + sum(sum(Theta_2_reg ** 2)) + sum(sum(Theta_3_reg ** 2))
			cost_reg = self.lambda_reg / (2 * m) * cost_reg
			cost += cost_reg.item()

		return cost, classification_error
