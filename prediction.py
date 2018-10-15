# (☞ﾟヮﾟ)☞  PREDICTION.PY

from NeuralNetwork import NeuralNetwork
from exceptions import ParserException
from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) != 3:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' prediction.py ' + Fore.RESET + \
			'param.dat prediction_data.csv')
		sys.exit(-1)
	param_file = sys.argv[1]
	prediction_file = sys.argv[2]

	try:
		neural_network = NeuralNetwork(param_file)
		neural_network.predict(prediction_file)
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
