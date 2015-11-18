import sys
import os
import argparse
import csv
import lmdb
import caffe
import random
import numpy as np

# specify csv-type file formats here
CSV_FORMATS = {
    "DUDE_SCOREDATA": {
        "delimiter":' ',
        "id":2,                     # column specifying sample id
        "group":1,                  # column specifying target 
        "data":list(range(4, 65)),  # list of columns specifying input features
        "label":0                   # column specifying the class label
    },
}

# global variables for data partitioning mode
BALANCED_PARTITION = 0
SPLIT_BY_TARGET = 1



class Database:

    '''This class is used for reading in data from a csv file and
    performing preprocessing steps, such as normalization, shuffling
    the sample order, and generating partitions for cross-validation.
    It can also write the data out in lmdb format.'''

    def __init__(self, file_, format_):

        '''Construct a Database using a path to a csv file and a
        string specifying the csv format, which is used to lookup
        the format in the CSV_FORMATS global dictionary.'''

        self.read_csv(file_, format_)


    def read_csv(self, file_, format_):

        '''Reads data from a csv file in the specified format into
        the Database, overwriting any previous data it contained.'''

        # open csv file in a known format
        csv_file   = open(file_, "r")
        csv_format = CSV_FORMATS[format_] # throws KeyError if unknown format
        csv_reader = csv.reader(csv_file, delimiter=csv_format["delimiter"])

        self._source = file_        # path to source file
        self._format = csv_format   # format specification
        self._samples = []          # list of data samples
        self._groups = {}           # dictionary of grouped data samples
        self._parts = []            # list of data partitions

        # each row in the csv file is parsed into a dictionary with keys for
        # the sample id, sample group, list of features(x) and label(y)
        # csv_format is used to parse the csv columns into these values
        for row in csv_reader:

            # parse the row into a dictionary
            # throws IndexError if csv_format is incorrect
            s = dict(id=row[csv_format["id"]],
                     group=row[csv_format["group"]],
                     x=[float(row[i]) for i in csv_format["data"]],
                     y=int(row[csv_format["label"]])) 

            # append sample to master sample list
            self._samples.append(s)

            # also append sample to the dictionary of grouped samples
            if s["group"] in self._groups:
                self._groups[s["group"]].append(s)
            else:
                self._groups[s["group"]] = [s]
        
        csv_file.close()
        
        # get a *rough* estimate of the number of bytes of data
        # assuming 64bit float values
        self.nbytes = 8 * len(self._format["data"]) * len(self._samples)


    def write_lmdb(self, dir_):

        '''Writes out the entire dataset and each train/test partition in the
        lmdb format.'''

        # write the entire database
        source_file = os.path.basename(self._source)
        full_path = os.path.join(dir_, source_file+".full")
        full_lmdb = lmdb.open(full_path, map_size=4*self.nbytes)
        with full_lmdb.begin(write=True) as txn:
            for s in self._samples:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = len(self._format["data"])
                datum.height = 1
                datum.width  = 1
                datum.float_data.extend(s["x"]);
                datum.label = s["y"]
                txn.put(key=s["id"].encode("ascii"),
                        value=datum.SerializeToString())
        full_lmdb.close()

        # for each partition in the database
        for i in range(len(self._parts)):
            partition  = self._parts[i]
            n = partition._name

            # create lmdbs for train and test set, if they exist in the partition

            if partition._train_set is not None:
                train_path = os.path.join(dir_, source_file+"."+n+".train")
                train_lmdb = lmdb.open(train_path, map_size=4*self.nbytes)
                print("\t" + train_path)
                with train_lmdb.begin(write=True) as txn:
                    for s in partition.train_set():
                        datum = caffe.proto.caffe_pb2.Datum()
                        datum.channels = len(self._format["data"])
                        datum.height = 1
                        datum.width  = 1
                        datum.float_data.extend(s["x"]);
                        datum.label = s["y"]
                        txn.put(key=s["id"].encode("ascii"),
                                value=datum.SerializeToString())
                train_lmdb.close()

            if partition._test_set is not None:
                test_path  = os.path.join(dir_, source_file+"."+n+".test")
                test_lmdb  = lmdb.open(test_path, map_size=4*self.nbytes)
                print("\t" + test_path)
                with test_lmdb.begin(write=True) as txn:
                    for s in partition.test_set():
                        datum = caffe.proto.caffe_pb2.Datum()
                        datum.channels = len(self._format["data"])
                        datum.height = 1
                        datum.width  = 1
                        datum.float_data.extend(s["x"]);
                        datum.label = s["y"]
                        txn.put(key=s["id"].encode("ascii"),
                                value=datum.SerializeToString())
                test_lmdb.close()
    

    def balanced_partition(self, n=10):

        '''Generates n Database.Partitions by iterating back and forth through 
        the targets, trying to get 1/n of the entire dataset in each test set 
        and (n-1)/n in each train set, while keeping all samples of the same 
        target in the same set.'''

        self._parts = []

        # sort the groups based on number of samples
        sorted_groups = [(g, len(self._groups[g])) for g in self._groups]
        sorted_groups.sort(key=lambda tup: tup[1], reverse=True)

        index = 0
        forward = True
        folds = [[] for i in range(n)]
        for g in sorted_groups:
            folds[index].append(g[0])
            if forward:
                if index < n-1:
                    index += 1
                else:
                    forward = False
            else:
                if index > 0:
                    index -= 1
                else:
                    forward = True

        for i in range(n):
            test_set = folds[i]
            train_set = [j for f in folds[:i] for j in f] + [j for f in folds[i+1:] for j in f]
            self._parts.append(Database.Partition(self, "part"+str(i), train_set, test_set))


    def split_by_target(self):

        '''Generates a Database.Partition for each target. Does not specify a
        training partition, because right now the whole dataset is used to 
        train when testing individual targets.'''

        self._parts = []
        for g in self._groups:
            self._parts.append(Database.Partition(self, g, None, [g]))


    def normalize(self):

        '''Normalize the sample data. This calculates the mean and standard
        deviation of each input feature in the dataset, then for each sample
        shifts and scales all of its features.'''

        n = len(self._samples)
        d = len(self._format["data"])

        # compute mean for each feature
        mean = [0 for i in range(d)]
        for i in self._samples:
            for j in range(d):
                mean[j] += i["x"][j]
        mean = [i/n for i in mean]

        # compute standard deviation for each feature
        sd = [0 for i in range(d)]
        for i in self._samples:
            for j in range(d):
                sd[j] += (i["x"][j] - mean[j])**2
        sd = [(i/n)**(0.5) for i in sd]

        # normalize each feature: subtract mean, divide by std dev  
        for i in self._samples:
            for j in range(d):
                if sd[j]:
                    i["x"][j] = (i["x"][j] - mean[j])/float(sd[j])
                else:
                    i["x"][j] = 0.0

    def shuffle(self):

        '''Randomize the order of the samples in the master list,
        and also within each list grouped by target.'''
        
        random.shuffle(self._samples)
        for i in self._groups:
            random.shuffle(self._groups[i])

    def write_weight_matrix(self):

        # count number of samples in each class label
        pos, neg = 0, 0
        for i in self._samples:
            if i['y'] == 0:
                neg += 1
            else:
                pos += 1

        # calculate the imbalance, and weight the cost of
        # mispredicting the dominant class proportionally less
        if pos < neg:
            imb = pos / float(neg)
            weights = np.array([[imb, 0.0],
                                [0.0, 1.0]])
        else:
            imb = neg / float(pos)
            weights = np.array([[1.0, 0.0],
                                [0.0, imb]])

        # write the weight matrix in binary protobuf format
        weight_blob = caffe.io.array_to_blobproto(weights.reshape(1, 1, 2, 2))
        weight_matrix_file = self._source + '_weightmatrix.binaryproto'
        with open(weight_matrix_file, 'wb') as w:
            w.write(weight_blob.SerializeToString())

    class Partition:

        '''The Database.Partition class keeps track of a particular
        partitioning of the grouped data into a train set and test set.'''

        def __init__(self, db, name, train, test):

            '''Contruct a Database.Partition by specifying the Database to
            partition, a name for the partition, and a collection of the
            names of the groups in the train and test set.'''

            self._db = db
            self._name = name
            if train: self._train_set = set(train)
            else: self._train_set = None
            if test: self._test_set  = set(test)
            else: self._test_set = None

        def train_set(self):

            '''A generator function that yields samples from the Database
            which are in a group that's in the train set.'''

            for s in self._db._samples:
                if s["group"] in self._train_set:
                    yield s

        def test_set(self):

            '''A generator function that yields samples from the Database
            which are in a group taht's in the test set.'''

            for s in self._db._samples:
                if s["group"] in self._test_set:
                    yield s

def main(argv=sys.argv[1:]):

    args = parse_args(argv)
    if args.mode == 'bp':
        mode = BALANCED_PARTITION
    elif args.mode == 'sbt':
        mode = SPLIT_BY_TARGET
    else:
        mode = BALANCED_PARTITION

    print('Gathering data from '+args.INPUT_FILE)
    try:
        db = Database(args.INPUT_FILE, 'DUDE_SCOREDATA')
    except IOError:
        print('Error: could not access the input file')
        return 1
    except KeyError:
        print('Error: unknown input file format')
        return 1
    except IndexError:
        print('Error: incorrect input file format')
        return 1
    print(str(db.nbytes)+' bytes read')

    print('Normalizing and shuffling data')
    db.normalize()
    db.shuffle()

    if mode == BALANCED_PARTITION:
        print('Generating 10 balanced partitions')
        db.balanced_partition()
    elif mode == SPLIT_BY_TARGET:
        print('Splitting data by individual targets')
        db.split_by_target()

    print('Converting to lmdb format')
    try:
        db.write_lmdb(args.OUTPUT_DIR)
    except IOError:
        print('Error: could not access lmdb output location')
        return 1

    print('Creating class weight matrix')
    try:
        db.write_weight_matrix()
    except IOError:
        print('Error: could not access weight matrix output location')
        return 1

    print('Done, without errors.')
    return 0

def parse_args(argv):

    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Convert a csv database into partitioned lmdb files.')
    parser.add_argument('INPUT_FILE')
    parser.add_argument('OUTPUT_DIR')
    parser.add_argument('--mode', '-m', type=str)
    return parser.parse_args(argv)

if __name__ == '__main__':
    sys.exit(main())