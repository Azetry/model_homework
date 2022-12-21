import pandas as pd
import argparse



parser = argparse.ArgumentParser(
                    prog = 'util',
                    description = '改作業用',
                    epilog = '...')

parser.add_argument('-i', '--test_data')
parser.add_argument('-f', '--features')
parser.add_argument('-t', '--type')


parser.parse_args()


print("test.")

