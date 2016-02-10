import sys
import os
import argparse
import csv
import lmdb
import caffe
import random
import numpy as np
from operator import mul

# specify csv-type file formats here
CSV_FORMATS = {
    'LABELLED_SCOREDATA': {
        'delimiter': ' ',
        'label': 0,    # class label column
        'target': 1,   # target name column
        'id': 2,       # sample id column
        'data': 4,     # first column of data
        'shape': [61], # data dimensions
    },

    'UNLABELLED_SCOREDATA': {
        'delimiter': ' ',
        'label': None,
        'target': 1,
        'id': 2,
        'data': 4,
        'shape': [61]
    },
}


class PartitionMode:
    BALANCED = 0
    TARGETS = 1


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

        '''Read data from a csv file in the specified format into
        the Database, overwriting any previous data it contained.'''

        # open csv file using the delimiter in the format argument
        csv_file   = open(file_, 'r') # throws IOError if file can't be opened
        csv_format = CSV_FORMATS[format_] # throws KeyError if format is unknown
        csv_reader = csv.reader(csv_file, delimiter=csv_format['delimiter'])

        self.source = file_        # path to source file
        self._format = csv_format  # format specification
        self.samples = []          # list of all data samples
        self.targets = {}          # dictionary of samples grouped by target
        self.parts = []            # list of data partitions

        # determine data column range using start column and dimensions
        data_cols = reduce(mul, csv_format['shape'])
        data_range = range(csv_format['data'], csv_format['data'] + data_cols)

        # each row in the csv file is parsed into a dictionary with keys for
        # the sample id, target, data vector (x) and label (y)
        # csv_format is used to parse the row into these values
        for n, row in enumerate(csv_reader):

            if row[0] == 'mismatched':
                continue

            # throws IndexError if csv_format is incorrect
            id_ = str(n) + '_' + row[csv_format['id']] 
            target = row[csv_format['target']]
            x = [float(row[i]) for i in data_range]
            if csv_format['label']:
                y = int(row[csv_format['label']])
            else:
                y = None
            s = dict(id=id_, target=target, x=x, y=y)

            # append sample to master sample list
            self.samples.append(s)

            # also append sample to the target dictionary
            t = s['target']
            if t in self.targets:
                self.targets[t].append(s)
            else:
                self.targets[t] = [s]
        
        csv_file.close()
        
        self.ncols = data_cols
        self.nsamples = len(self.samples)
        # get a rough estimate of the number of bytes of data assuming 64bit floats
        # this should ONLY be used for determining map size in lmdb, nothing else
        self.nbytes = data_cols * self.nsamples * 8

    def _write_one_lmdb(self, path, filter_=None):

        '''Write a single lmdb to the given path. If a filter argument
        is provided, call it as a function on each sample and only includes
        the sample in the lmdb if condition evaluates to true.'''

        with lmdb.open(path, map_size=10*self.nbytes).begin(write=True) as db:
            for i in self.samples:
                if filter_ is None or filter_(i):

                    # make the sample into a caffe Datum
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.shape.dim.extend(self._format['shape'])
                    datum.float_data.extend(i['x']);
                    if i['y']: datum.label = i['y']

                    # put it as a key, value mapping in the lmdb
                    key = i['id'].encode('ascii')
                    value = datum.SerializeToString()
                    db.put(key, value)

    def write_lmdb(self, dir_, base_name=None):

        '''Write out the entire dataset and each train/test partition in
        lmdb format databases, in the argument directory. If no base
        name is provided, use the source file name as the base name.'''

        if base_name is None:
            base_name = os.path.basename(self.source)

        # write lmdb for the full database
        full_path = os.path.join(dir_, base_name + '.full')
        self._write_one_lmdb(full_path)
        print(full_path)

        # write train and test lmdb for each partition in the database
        for p in self.parts:
            n = p._name

            if p.train_set is not None:
                train_path = os.path.join(dir_, base_name + '.' + n + '.train')
                self._write_one_lmdb(train_path, lambda x: x['target'] in p.train_set)
                print(train_path)

            if p.test_set is not None:
                test_path  = os.path.join(dir_, base_name + '.' + n + '.test')
                self._write_one_lmdb(train_path, lambda x: x['target'] in p.test_set)
                print(test_path)
    

    def balanced_partition(self, n=10):

        '''Generates n Database.Partitions by iterating back and forth through 
        the targets, trying to get 1/n of the entire dataset in each test set 
        and (n-1)/n in each train set, while keeping all samples of the same 
        target in the same set.'''

        self.parts = []

        # sort the groups based on number of samples
        sorted_targets = [(t, len(self.targets[t])) for t in self.targets]
        sorted_targets.sort(key=lambda tup: tup[1], reverse=True)

        index = 0
        forward = True
        folds = [[] for i in range(n)]
        for t in sorted_targets:
            folds[index].append(t[0])
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
            self.parts.append(Database.Partition('part' + str(i), train_set, test_set))

    def partition_by_target(self):

        '''Generates a Database.Partition for each target. Does not specify a
        training partition, because right now the whole dataset is used to 
        train when testing individual targets.'''

        self.parts = []
        for target in self.targets:
            self.parts.append(Database.Partition(target, None, [target]))

    def normalize(self):

        '''Normalize the sample data. This calculates the mean and standard
        deviation of each input feature in the dataset, then for each sample
        shifts and scales all of its features.'''

        n = self.nsamples
        d = self.ncols

        # compute mean for each feature
        mean = [0 for i in range(d)]
        for i in self.samples:
            for j in range(d):
                mean[j] += i['x'][j]
        mean = [i/n for i in mean]

        # compute standard deviation for each feature
        sd = [0 for i in range(d)]
        for i in self.samples:
            for j in range(d):
                sd[j] += (i['x'][j] - mean[j])**2
        sd = [(i/n)**(0.5) for i in sd]

        # normalize each feature: subtract mean, divide by sd  
        for i in self.samples:
            for j in range(d):
                if sd[j]:
                    i['x'][j] = (i['x'][j] - mean[j])/float(sd[j])
                else:
                    i['x'][j] = 0.0 # don't divide by zero

    def shuffle(self):

        '''Randomize the order of the samples in the master list,
        and also within each list grouped by target.'''
        
        random.shuffle(self.samples)
        for i in self.targets:
            random.shuffle(self.targets[i])

    def reshape(self, new_shape):

        new_data_cols = reduce(mul, new_shape)

        if new_data_cols > self.ncols:
            raise IndexError('can\'t reshape ' + str(self.ncols) + \
                ' data features into shape ' + 'x'.join([str(i) for i in new_shape]))
        elif new_data_cols < self.ncols:
            print('Warning: ' + str(self.ncols - new_data_cols) + ' feature(s) were truncated')

        self._format['shape'] = new_shape

    def write_class_weight_matrix(self):

        '''Calculate the class imbalance in the dataset and write
        a class weight matrix in caffe blobproto format. This currently
        only supports binary classes.'''

        if self._format['label'] is None:
            raise KeyError('data must be labelled to determine class imbalance')

        # count number of samples in each class label
        pos, neg = 0, 0
        for i in self.samples:
            if i['y'] == 0:
                neg += 1
            elif i['y'] == 1:
                pos += 1

        # calculate the class imbalance and weight the cost of
        # mispredicting the dominant class proportionately less
        if pos < neg:
            imb = pos / float(neg)
            weights = np.array([[imb, 0.0],
                                [0.0, 1.0]])
        else:
            imb = neg / float(pos)
            weights = np.array([[1.0, 0.0],
                                [0.0, imb]])

        # write the weight matrix as a binary blobproto
        weight_blob = caffe.io.array_to_blobproto(weights.reshape(1, 1, 2, 2))
        weight_matrix_file = self.source + '_weightmatrix.binaryproto' # TODO
        with open(weight_matrix_file, 'wb') as f:
            f.write(weight_blob.SerializeToString())
        print(weight_matrix_file)

    class Partition:

        '''The Database.Partition class keeps track of a particular
        partitioning of the targets into a train set and test set.'''

        def __init__(self, name, train, test):

            '''Contruct a Database.Partition by specifying a name for the
            partition and lists of the names of which targets are in the
            train and test sets.'''

            self._name = name

            if train:
                self.train_set = set(train)
            else:
                self.train_set = None

            if test:
                self.test_set  = set(test)
            else:
                self.test_set = None


def main(argv=sys.argv[1:]):

    args = parse_args(argv)
    if not args.partition:
        part_mode = None
    elif args.partition == 'balanced':
        part_mode = PartitionMode.BALANCED
    elif args.partition == 'targets':
        part_mode = PartitionMode.TARGETS
    else:
        print('Error: options for partition argument are "balanced" or "targets"')
        return 1

    print('Gathering data from '+args.INPUT_FILE)
    try:
        db = Database(args.INPUT_FILE, args.INPUT_FORMAT)
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

    if part_mode == PartitionMode.BALANCED:
        print('Generating 10 balanced partitions')
        db.balanced_partition()
    elif part_mode == PartitionMode.TARGETS:
        print('Partitioning data by individual targets')
        db.partition_by_target()

    print('Converting to lmdb format')
    try:
        db.write_lmdb(args.OUTPUT_DIR)
    except IOError:
        print('Error: could not access lmdb output location')
        return 1

    print('Creating class weight matrix')
    try:
        db.write_class_weight_matrix()
    except KeyError:
        pass
    except IOError:
        print('Error: could not access weight matrix output location')
        return 1

    print('Done, without errors.')
    return 0

def parse_args(argv):

    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Convert a csv-like database into preprocessed lmdb format.')
    parser.add_argument('INPUT_FILE')
    parser.add_argument('INPUT_FORMAT')
    parser.add_argument('OUTPUT_DIR')
    parser.add_argument('--partition', '-p')
    return parser.parse_args(argv)

if __name__ == '__main__':
    sys.exit(main())