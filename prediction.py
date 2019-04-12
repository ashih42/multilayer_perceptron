# (☞ﾟヮﾟ)☞  PREDICTION.PY

from neural_network import NeuralNetwork
from exceptions import ParserException

from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) != 3:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' prediction.py ' + Fore.RESET + \
			'param.dat prediction_data.csv')
		sys.exit(-1)
	param_filename = sys.argv[1]
	prediction_filename = sys.argv[2]

	try:
		neural_network = NeuralNetwork(param_filename)
		neural_network.predict(prediction_filename)
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
