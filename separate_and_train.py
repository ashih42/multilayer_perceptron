# (☞ﾟヮﾟ)☞  SEPARATE_AND_TRAIN.PY

from neural_network import NeuralNetwork
from exceptions import ParserException

from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) < 2:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' separate_and_train.py ' + Fore.RESET + \
			'data.csv [-o]')
		sys.exit(-1)
	data_filename = sys.argv[1]
	do_online = False
	if len(sys.argv) == 3:
		do_online = sys.argv[2] == '-o'

	try:
		neural_network = NeuralNetwork()
		neural_network.separate_data_and_train(data_filename, do_online)
		neural_network.save_param_to_file('param.dat')
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
