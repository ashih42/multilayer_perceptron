# (☞ﾟヮﾟ)☞  TRAINING.PY

from neural_network import NeuralNetwork
from exceptions import ParserException

from colorama import Fore, Back, Style
import sys

def main():
	# check argv
	if len(sys.argv) < 3:
		print('usage: ' + Fore.RED + 'python3' + Fore.BLUE + ' training.py ' + Fore.RESET + \
			'training_data.csv validation_data.csv [-o]')
		sys.exit(-1)
	training_filename = sys.argv[1]
	validation_filename = sys.argv[2]
	do_online = False
	if len(sys.argv) == 4:
		do_online = sys.argv[3] == '-o'

	try:
		neural_network = NeuralNetwork()
		neural_network.train(training_filename, validation_filename, do_online)
		neural_network.save_param_to_file('param.dat')
	except IOError as e:
		print(Style.BRIGHT + Fore.RED + 'I/O Error: ' + Style.RESET_ALL + Fore.RESET + str(e))
	except ParserException as e:
		print(Style.BRIGHT + Fore.RED + 'ParserException: ' + Style.RESET_ALL + Fore.RESET + str(e))

if __name__ == '__main__':
	main()
