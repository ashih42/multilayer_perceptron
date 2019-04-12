from exceptions import ParserException

from colorama import Fore, Back, Style

class ParamParser:
	__FEATURE_COUNT = 30
	__THETA_1_ROWS = 30
	__THETA_1_COLS = 31
	__THETA_2_ROWS = 30
	__THETA_2_COLS = 31
	__THETA_3_ROWS = 1
	__THETA_3_COLS = 31

	def __init__(self, filename):
		self.__line_number = 0
		with open(filename, 'r') as param_file:
			self.mean_list = self.__parse_list(param_file.readline().strip(), ParamParser.__FEATURE_COUNT)
			self.stdev_list = self.__parse_list(param_file.readline().strip(), ParamParser.__FEATURE_COUNT)

			self.__parse_dimensions(param_file.readline().strip(), ParamParser.__THETA_1_ROWS, ParamParser.__THETA_1_COLS)
			self.theta_1_list = self.__parse_theta(param_file, ParamParser.__THETA_1_ROWS, ParamParser.__THETA_1_COLS)

			self.__parse_dimensions(param_file.readline().strip(), ParamParser.__THETA_2_ROWS, ParamParser.__THETA_2_COLS)
			self.theta_2_list = self.__parse_theta(param_file, ParamParser.__THETA_2_ROWS, ParamParser.__THETA_2_COLS)

			self.__parse_dimensions(param_file.readline().strip(), ParamParser.__THETA_3_ROWS, ParamParser.__THETA_3_COLS)
			self.theta_3_list = self.__parse_theta(param_file, ParamParser.__THETA_3_ROWS, ParamParser.__THETA_3_COLS)

	def __parse_theta(self, param_file, expected_rows, expected_cols):
		theta_list = []
		for _ in range(expected_rows):
			row = self.__parse_list(param_file.readline().strip(), expected_cols)
			theta_list.append(row)
		return theta_list

	def __parse_list(self, line, num_terms):
		self.__line_number += 1
		tokens = line.split()
		if len(tokens) != num_terms:
			raise ParserException('invalid number of terms at ' +
				Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
				Fore.MAGENTA + line + Fore.RESET)
		lst = []
		for token in tokens:
			try:
				lst.append(float(token))
			except ValueError:
				raise ParserException('invalid term at ' +
					Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
					Fore.MAGENTA + token + Fore.RESET)
		return lst

	def __parse_dimensions(self, line, expected_rows, expected_cols):
		self.__line_number += 1
		tokens = line.split()
		if len(tokens) != 2:
			raise ParserException('invalid dimensions at ' +
				Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
				Fore.MAGENTA + line + Fore.RESET)
		try:
			rows = int(tokens[0])
			cols = int(tokens[1])
			if not (rows == expected_rows and cols == expected_cols):
				raise ParserException('invalid dimensions at ' +
					Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
					Fore.MAGENTA + line + Fore.RESET)
		except ValueError:
			raise ParserException('invalid dimensions at ' +
				Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
				Fore.MAGENTA + line + Fore.RESET)
