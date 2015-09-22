import sys
import os
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
	"general":"""\
Usage: python cnnscore.py <command> [<args>]

    Commands:
        lmdb\tConvert a csv database to lmdb format
        train\tTrain a network using Caffe
	""",

	"lmdb":"""\
Usage: python cnnscore.py lmdb <csv file> <csv format>

    Known CSV formats:
        DUDE_SCOREDATA
	""",

	"train":"""\
Usage: python cnnscore.py train <solver prototxt>
	"""
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

	def write_lmdb(self, data_map, label_map):

		with data_map.begin(write=True) as txn:

			for s in self._data:
				datum = caffe.io.array_to_datum(s._data)
				txn.put(key=s._id.encode("ascii"), value=datum.SerializeToString())

		with label_map.begin(write=True) as txn:

			for s in self._data:
				datum = caffe.io.array_to_datum(s._label)
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
	if command_arg == "lmdb":

		if len(sys.argv) < 4:

			print(USAGE_STRING["lmdb"])
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

		try: 
			data_lmdb  = lmdb.open(file_arg+".data.lmdb",  map_size=db.nbytes*2)
			label_lmdb = lmdb.open(file_arg+".label.lmdb", map_size=db.nbytes*2)
		except IOError:
			print("Error: could not open lmdb environment")
			sys.exit(1)
		print("Converting to lmdb format")
		db.write_lmdb(data_lmdb, label_lmdb)
		data_lmdb.close()
		label_lmdb.close()
		print("Done.")

	elif command_arg == "train":

		if len(sys.argv) < 3:

			print(USAGE_STRING["train"])
			sys.exit(1)

		solver_arg = sys.argv[2]
		train_command = "$CAFFE_ROOT/build/tools/caffe train --solver=" + solver_arg 
		os.system(train_command)

	else:

		print(USAGE_STRING["general"])
		sys.exit(1)

	sys.exit(0)
