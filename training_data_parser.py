from exceptions import ParserException

from colorama import Fore, Back, Style

'''
Assumptions:
- 32 Columns
- Colume 0 is Patient ID		=> Patient ID (unused)
- Column 1 is LABEL {'M', 'B'}	=> LABEL { 'M', 'B' }
- Column 2...31					=> FEATURES { float }
'''

class TrainingDataParser:
	__NUM_COLUMNS = 32

	def __init__(self, filename):
		self.data = []
		self.__line_number = 0

		with open(filename, 'r') as data_file:
			for line in data_file:
				try:
					self.__parse_line(line.strip())
				except ParserException as e:
					print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))

		print('Accepted %d, discarded %d rows of data\n' %
			(len(self.data), self.__line_number - len(self.data)))
		if len(self.data) == 0:
			raise ParserException('dataset is empty')
		
	def __parse_line(self, line):
		self.__line_number += 1
		tokens = line.split(',')

		# check number of columns
		if len(tokens) != TrainingDataParser.__NUM_COLUMNS:
			raise ParserException('invalid number of terms at ' +
				Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
				Fore.MAGENTA + line + Fore.RESET)

		# skip first column (probably patient ID)

		# check LABEL is a valid answer
		label = tokens[1]
		if not (label == 'M' or label == 'B'):
			raise ParserException('invalid label value at ' +	Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
				'LABEL: ' + Fore.MAGENTA + label + Fore.RESET)

		# check each FEATURE can be parsed to float
		row_data = [label]
		for i in range(2, TrainingDataParser.__NUM_COLUMNS):
			try:
				row_data.append(float(tokens[i]))
			except ValueError:
				raise ParserException('invalid feature value at ' + Fore.GREEN + 'line ' + str(self.__line_number) + Fore.RESET + ': ' +
					('Feature %02d: ' % (i - 1)) + Fore.MAGENTA + tokens[i] + Fore.RESET)

		self.data.append(row_data)
