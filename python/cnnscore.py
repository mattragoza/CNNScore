import sys
import glob
import csv
import lmdb

if __name__ == "__main__":

	if len(sys.argv) > 2:

		for f in glob.glob(sys.argv[1]):

			print("Gathering data from " + f)
			try:
				input_file = open(f, 'rb')
			except IOError:
				print("Error: could not access " + f)
				continue

			input_data = []
			input_reader = csv.reader(input_file, delimiter=" ")
			for row in input_reader:
				print(row)

			input_file.close()

		print("Done.")
		sys.exit(0)

	else:

		print("Usage: python create_lmdb.py INPUT_PATTERN")
		sys.exit(1)
