import matplotlib
matplotlib.use('Agg')
import sys
import os
import re
import numpy as np
import pandas
import caffe
from caffe.proto import caffe_pb2
from google import protobuf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.cm as colormap


class CNNScoreModel:

    def __init__(self, n_units, n_conv_per_unit, n_filters, batch_size=10,
                 rotate=24, pool=True, residual=False):
        '''
        Construct a base CNN model made up of n_units layer units. Each unit
        is a series of n_conv_per_unit convolution layers and ReLUs with the
        same number of filters, starting at n_filters.

        If downsample is set, 2x max pooling is applied between each unit and
        the number of filters doubles per unit.

        If residual is set, additive connections are included between the
        bottom and top of each unit to allow residual learning.
        '''
        self.name = '{}x{}x{}'.format(n_units, n_conv_per_unit, n_filters)
        self.proto = caffe_pb2.NetParameter()

        # data layer and initial downsampling convolution
        data, label = self._add_data_layer(batch_size, rotate)
        top = self._add_conv_layer(data, 'down_conv', n_filters, stride=2)
        n_filters *= 2

        # customizable convolution units
        for i in range(n_units):
            unit_name = 'unit{}'.format(i+1)
            top = self._add_unit(top, unit_name, n_conv_per_unit, n_filters,
                                 residual)
            if pool:
                pool_name = 'pool{}'.format(i+1)
                top = self._add_pool_layer(top, pool_name)
                n_filters *= 2

        # final fully connected layer and softmax prediction and loss
        top = self._add_fc_layer(top, 'fc', 2)
        self._add_pred_layer(top)
        self._add_loss_layer(top, label)


    def _add_data_layer(self, batch_size, rotate):

        data_layer = self.proto.layer.add()
        data_layer.name = 'data'
        data_layer.type = 'NDimData'
        data_layer.top.append('data')
        data_layer.top.append('label')
        data_layer.ndim_data_param.batch_size = batch_size
        data_layer.ndim_data_param.shape.dim.extend([34, 49, 49, 49])
        data_layer.ndim_data_param.balanced = True
        data_layer.ndim_data_param.shuffle = True
        data_layer.ndim_data_param.rotate = rotate
        self.xyz = 49
        return 'data', 'label'

    def _add_conv_layer(self, bottom, top, n_filters, kernel=3, pad=1, stride=1):

        new_xyz = (self.xyz + 2*pad - kernel)/stride + 1
        assert new_xyz > 0
        self.xyz = new_xyz

        conv_layer = self.proto.layer.add()
        conv_layer.name = top
        conv_layer.type = 'Convolution'
        conv_layer.bottom.append(bottom)
        conv_layer.top.append(top)
        conv_layer.convolution_param.kernel_size.append(kernel)
        conv_layer.convolution_param.pad.append(1)
        conv_layer.convolution_param.stride.append(stride)
        conv_layer.convolution_param.num_output = n_filters
        conv_layer.convolution_param.weight_filler.type = 'xavier'
        return top

    def _add_relu_layer(self, bottom, top):

        relu_layer = self.proto.layer.add()
        relu_layer.name = top
        relu_layer.type = 'ReLU'
        relu_layer.bottom.append(bottom)
        relu_layer.top.append(bottom) # in-place
        return bottom

    def _add_res_layer(self, bottom1, bottom2, top):

        res_layer = self.proto.layer.add()
        res_layer.name = top
        res_layer.type = 'Eltwise'
        res_layer.bottom.append(bottom1)
        res_layer.bottom.append(bottom2)
        res_layer.top.append(top)
        res_layer.eltwise_param.operation = res_layer.eltwise_param.SUM
        return top

    def _add_pool_layer(self, bottom, top):

        new_xyz = (self.xyz + 2*0 - 2)/2 + 1
        assert new_xyz > 0
        self.xyz = new_xyz

        pool_layer = self.proto.layer.add()
        pool_layer.name = top
        pool_layer.type = 'Pooling'
        pool_layer.bottom.append(bottom)
        pool_layer.top.append(top)
        pool_layer.pooling_param.pool = pool_layer.pooling_param.MAX
        pool_layer.pooling_param.kernel_size.append(2)
        pool_layer.pooling_param.pad.append(0)
        pool_layer.pooling_param.stride.append(2)
        return top

    def _add_fc_layer(self, bottom, top, n_output):

        fc_layer = self.proto.layer.add()
        fc_layer.name = top
        fc_layer.type = 'InnerProduct'
        fc_layer.bottom.append(bottom)
        fc_layer.top.append(top)
        fc_layer.inner_product_param.num_output = n_output
        fc_layer.inner_product_param.weight_filler.type = 'xavier'
        return top

    def _add_pred_layer(self, bottom):

        pred_layer = self.proto.layer.add()
        pred_layer.name = 'pred'
        pred_layer.type = 'Softmax'
        pred_layer.bottom.append(bottom)
        pred_layer.top.append('pred')
        return 'pred'

    def _add_loss_layer(self, bottom1, bottom2):

        loss_layer = self.proto.layer.add()
        loss_layer.name = 'loss'
        loss_layer.type = 'SoftmaxWithLoss'
        loss_layer.bottom.append(bottom1)
        loss_layer.bottom.append(bottom2)
        loss_layer.top.append('loss')
        return 'loss'

    def _add_unit(self, bottom, unit_name, n_conv, n_filters, resid=False):

        top = bottom
        for i in range(n_conv):
            conv_name = unit_name + '_conv{}'.format(i+1)
            top = self._add_conv_layer(top, conv_name, n_filters)

            relu_name = unit_name + '_relu{}'.format(i+1)
            top = self._add_relu_layer(top, relu_name)

        if resid:
            res_name = unit_name + '_resid'
            top = self._add_res_layer(bottom, top, res_name)

        return top


    def _get_instance(self, data_file, data_root, phase):
        '''
        Make a copy of the model protobuf that has the data source and root
        replaced in the data layer and other changes made according to the
        phase argument, which can be either "train" or "test".

        For training, the prediction layer is removed.

        For testing, the loss layer is removed, and the data shuffling and
        balancing options are turned off.
        '''
        proto = caffe_pb2.NetParameter()
        proto.CopyFrom(self.proto)
        proto.layer[0].ndim_data_param.source = data_file
        proto.layer[0].ndim_data_param.root_folder = data_root

        if phase == 'train':
            for i, layer in enumerate(proto.layer):
                if layer.name == 'pred':
                    del proto.layer[i]
        elif phase == 'test':
            for i, layer in enumerate(proto.layer):
                if layer.name == 'loss':
                    del proto.layer[i]
            proto.layer[0].ndim_data_param.shuffle = False
            proto.layer[0].ndim_data_param.balanced = False
        else:
            raise ValueError('phase must be train or test')

        return proto


    def train(self, data_file, data_root, k=3, base_lr=0.001, momentum=0.9,
        weight_decay=0.0001, max_iter=20000, output_dir='./', gpus=None,
        snapshot=1000):
        '''
        Train the model with a dataset using optional training parameters,
        taking a weight snapshot at every 1000 iterations. Then validate
        the model using saved weights from every snapshot iteration.

        If k is greater than 1, k-fold cross-validation is performed. First
        the model is trained and tested on the entire dataset, then k more
        times using train and test data partitions, which should be generated
        beforehand.

        All model and solver prototxt, Caffe model weights, and score files
        are output to the directory given by the output_dir argument, using
        generated filenames.
        '''
        if k == 1: k = 0

        # get data, crossval part, and iteration params
        data_dir = os.path.dirname(data_file)
        data_param, data_ext = os.path.splitext(os.path.basename(data_file))
        part_params = ['part'+str(i) if i else 'full' for i in range(k+1)]
        weight_iters = range(snapshot, max_iter+1, snapshot)
        iter_params = ['iter_'+str(i) for i in weight_iters]

        # create a base solver from training args
        solver = caffe_pb2.SolverParameter()
        solver.random_seed = 0
        solver.base_lr = base_lr
        solver.momentum = momentum
        solver.weight_decay = weight_decay
        solver.lr_policy = 'inv'
        solver.gamma = 0.0001
        solver.power = 0.75
        solver.display = 100
        solver.max_iter = max_iter
        solver.snapshot = 1000

        # TRAIN PHASE
        for i in range(k+1):

            # get filename of training data
            if i == 0:
                train_data_file = data_file
            else:
                train_data_file = join_filename_params(data_dir,
                    [data_param, part_params[i], 'train'], data_ext)

            # get training model instance and write the model file
            train_model_file = join_filename_params(output_dir,
                [self.param, part_params[i], 'train'], '.model.prototxt')
            train_model = self._get_instance(train_data_file, data_root, 'train')
            with open(train_model_file, 'w') as f:
                f.write(str(train_model))

            # set solver to train the training model and write the solver file
            solver.train_net = train_model_file
            solver.snapshot_prefix = join_filename_params(output_dir,
                [self.param, part_params[i]], '')
            solver_file = join_filename_params(output_dir,
                [self.param, part_params[i]], '.solver.prototxt')
            with open(solver_file, 'w') as f:
                f.write(str(solver))

            # call Caffe executable to start training
            command = 'caffe train -solver ' + solver_file
            if gpus:
                command += ' -gpu ' + ','.join(map(str, gpus))
            os.system(command)

        if gpus:
            caffe.set_device(gpus[0])
            caffe.set_mode_gpu()
            
        # TEST PHASE
        all_scored_data = [[], []]
        for i in range(k+1):

            # get filename of test data
            if i == 0:
                test_data_file = data_file
            else:
                test_data_file = join_filename_params(data_dir,
                    [data_param, part_params[i], 'test'], data_ext)

            # get test model insance and write to model file
            test_model = self._get_instance(test_data_file, data_root, 'test')
            test_model_file = join_filename_params(output_dir,
                [self.param, part_params[i], 'test'], '.model.prototxt')
            with open(test_model_file, 'w') as f:
                f.write(str(test_model))

            # score test data using model weights at each snapshot iteration
            for j, iter_param in enumerate(iter_params):

                weight_file = join_filename_params(output_dir,
                    [self.param, part_params[i], iter_param], '.caffemodel')
                score_file = join_filename_params(output_dir,
                    [self.param, part_params[i], iter_param], '.scores')

                scored_data = get_caffe_model_predictions(test_model_file,
                    weight_file)
                scored_data.to_csv(score_file, sep=' ', header=False,
                    index=False)

                if i == 0:
                    scored_data.name = self.param+'_'+iter_param+'_testontrain'
                    all_scored_data[0].append(scored_data)
                elif i == 1:
                    scored_data.name = self.param+'_'+iter_param+'_crossval'
                    all_scored_data[1].append(scored_data)
                else:
                    all_scored_data[1][j].append(scored_data)
                    all_scored_data[1][j].name = \
                        self.param+'_'+iter_param+'_crossval'

        return all_scored_data


    def test(self, data_file, data_root, weight_file, gpus=None):
        '''
        Test the model on a dataset using pretrained weights.
        '''
        data_dir = os.path.dirname(data_file)
        data_param, data_ext = os.path.splitext(os.path.basename(data_file))

        # get test model instance and write to model file
        test_model = self._get_instance(data_file, data_root, 'test')
        test_model_file = join_filename_params(output_dir,
            [self.param, data_param], '.model.prototxt')
        with open(test_model_file, 'w') as f:
            f.write(str(test_model))

        # score test data using trained model weights
        score_file = re.sub('.caffemodel', '.scores', weight_file)
        scored_data = get_caffe_model_predictions(test_model_file, weight_file)
        scored_data.to_csv(score_file, sep=' ', header=False, index=False)
        return scored_data


def join_filename_params(dir, params, ext):
    '''
    Combine directory, param strings, and extension into a filepath
    '''
    return os.path.join(dir, '_'.join(params) + ext)


def get_caffe_model_predictions(model_file, weight_file):
    '''
    Get the output predictions of a trained Caffe model on its test data and
    return as a column in a data frame.
    '''
    # read test data from model data layer into a data frame
    model = caffe_pb2.NetParameter()
    with open(model_file, 'r') as f:
        protobuf.text_format.Merge(f.read(), model)
    data_file = model.layer[0].ndim_data_param.source
    data = pandas.read_csv(data_file, header=None, sep=' ',
        names=['label', 'target', 'example'], usecols=range(3))
    data['score'] = np.nan
    
    # get Caffe model output for each batch of test data
    model = caffe.Net(model_file, weight_file, caffe.TEST)
    batch_size = model.blobs['data'].shape[0]
    num_examples = data.shape[0]
    num_batches = num_examples//batch_size + 1
    for i in range(num_batches):

        print('%s | %s: batch %d / %d' % \
            (weight_file, data_file, i, num_batches))

        output = model.forward()
        for j in range(batch_size):
            data_index = i*batch_size + j
            if data_index >= num_examples:
                break
            if data['label'][data_index] != int(output['label'][j]):
                raise IndexError('file data does not match model data')
            data['score', data_index] = output['pred'][j][1]

    return data


def plot_roc_curves(plot_file, plot_data):
    '''
    Create a plot with an ROC curve for each data series in plot_data.

    The plot_data argument should be a list or single data frame(s) with
    columns called label and score, as well as a name attribute for the
    plot legend.
    '''
    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], '--', color=(0.6,0.6,0.6), label='random guess')

    try: list(plot_data)
    except TypeError:
        plot_data = [plot_data]
    
    aucs = []
    colors = colormap.rainbow(np.linspace(0, 1, len(plot_data)))
    for i, data_series in enumerate(plot_data):

        fpr, tpr, thresholds = roc_curve(data_series.label, data_series.score)
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, '-', color=colors[i], linewidth=2,
            label=data_series.name + ' (AUC=%0.2f)' % area)
        aucs.append(area)

    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_file, bbox_extra_artists=(legend,), bbox_inches='tight')

    return aucs
