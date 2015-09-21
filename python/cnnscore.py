import sys
import csv
import lmdb

KNOWN_FORMATS = {"DUDe.SCOREDATA":{"id":2, "group":1, "data":[], "label":[0]}}

class Database:

	def __init__(self, file, _format):

		columns = KNOWN_FORMATS[_format]
		self._data = []
		reader = csv.reader(file, delimiter=" ")
		for row in reader:
			s = Database.Sample(row[columns["id"]], row[columns["group"]],
				               [row[i] for i in columns["data"]],
				               [row[i] for i in columns["label"]])
			self._data.append(s)

	class Sample:

		def __init__(self, _id, group, data, label):

			self._id    = _id
			self._group = group
			self._data  = data
			self._label = label



if __name__ == "__main__":

	if len(sys.argv) > 2:

		if sys.argv[2] not in KNOWN_FORMATS:
			print("Error: unknown input format")
			sys.exit(1)

		try:
			input_file = open(sys.argv[1], 'r')
		except IOError:
			print("Error: could not access " + sys.argv[1])
			sys.exit(1)

		print("Gathering data from " + sys.argv[1])
		db = Database(file=input_file, _format=sys.argv[2])

		input_file.close()
		print("Done.")
		sys.exit(0)

	else:

		print("Usage: python create_lmdb.py FILENAME FORMAT")
		sys.exit(1)