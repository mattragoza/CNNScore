import sys
from sys import getsizeof as sizeof
import numpy
import csv
import lmdb
import caffe

KNOWN_FORMATS = {
    "DUDE_SCOREDATA":
    {
    	"delimiter":' ',
        "id":2,
        "group":1,
        "data":list(range(4, 65)),
        "label":[0]
    }
}

USAGE_STRING = {
	"general":"\
Usage: python cnnscore.py <command> [<args>]\n\
\n\
    Commands:\n\
        make_lmdb <csv file> <csv format>\n",

	"make_lmdb":"\
Usage: python cnnscore.py make_lmdb <csv file> <csv format>\n\
\n\
    Known CSV formats:\n\
        DUDE_SCOREDATA\n"
}




class Database:

	def __init__(self):

		self._data = []
		self._nbytes = 0

	def read_csv(self, file, _format):

		try: csv_format = KNOWN_FORMATS[_format]
		except KeyError:
			print("Error: unknown csv format")
			return

		csv_reader  = csv.reader(file, delimiter=csv_format["delimiter"])
		for row in csv_reader:
			s = Database.Sample(row[csv_format["id"]], row[csv_format["group"]],
				               [row[i] for i in csv_format["data"]],
				               [row[i] for i in csv_format["label"]],
				               csv_format)
			self._data.append(s)
		
		self.nbytes = sum([i._data.nbytes for i in self._data])

	def write_lmdb(self, mmap):

		with mmap.begin(write=True) as txn:

			for s in self._data:
				datum = caffe.io.array_to_datum(s._data)
				txn.put(key=s._id.encode("ascii"), value=datum.SerializeToString())


	class Sample:

		def __init__(self, _id, group, data, label, cf):

			self._id    = _id
			self._group = group
			self._data  = numpy.empty((len(cf["data"]), 1, 1),  dtype=numpy.float64)
			self._label = numpy.empty((len(cf["label"]), 1, 1), dtype=numpy.float64)
			for i in range(len(cf["data"])):  self._data[i, 0, 0]  = data[i]
			for i in range(len(cf["label"])): self._label[i, 0, 0] = label[i]




if __name__ == "__main__":

	if len(sys.argv) < 2:

		print(USAGE_STRING["general"])
		sys.exit(1)

	command_arg = sys.argv[1]
	if command_arg == "make_lmdb":

		if len(sys.argv) < 4:

			print(USAGE_STRING["make_lmdb"])
			sys.exit(1)

		file_arg = sys.argv[2]
		format_arg = sys.argv[3]

		try: input_file = open(file_arg, 'r')
		except IOError:
			print("Error: could not access " + file_arg)
			sys.exit(1)
		print("Gathering data from " + file_arg)
		db = Database()
		db.read_csv(file=input_file, _format=format_arg)
		input_file.close()

		try: lmdb_map = lmdb.open(file_arg+".lmdb", map_size=db.nbytes*2)
		except IOError:
			print("Error: could not open lmdb environment")
			sys.exit(1)
		print("Converting to lmdb format")
		db.write_lmdb(mmap=lmdb_map)
		lmdb_map.close()
		print("Done.")

	else:

		print(USAGE_STRING["general"])
		sys.exit(1)

	sys.exit(0)