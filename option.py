import argparse
import os
import util


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='./data/', help='root path to images')
        self.parser.add_argument('--cap_scheme', type=str, default='min24', help='name of the experiment, it decides where to store samples and models')
        self.parser.add_argument('--cap_len', type=int, default=6, help='the length of captcha')
        self.parser.add_argument('--char_set_len', type=int, default=10, help='the numer of char to be used')
        self.parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
        self.parser.add_argument('--train_size', type=int, default=500, help='the number of transferring training set')
        self.parser.add_argument('--loadHeight', type=int, default=53, help='default to this height')
        self.parser.add_argument('--loadWidth', type=int, default=140, help='default to this width')
        self.parser.add_argument('--keep_prob', type=int, default=0.5, help='default dropout value')
        self.parser.add_argument('--isTune', action='store_true', default=True, help='True: train-error fine tune model; False: train-error base model')
        self.parser.add_argument('--model', type=str, default='LeNet5', help='choose which model to use, only exists LeNet5')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./data/checkpoints', help='models are saved here')
        self.parser.add_argument('--callbacks', action='store_true', default=True, help='save best val_acc model')
        self.parser.add_argument('--display_winsize', type=int, default=200, help='display window size, this is the width value')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train-error or test

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        args = vars(self.opt)

        # save to disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.cap_scheme, str(self.opt.train_size))
        util.mkdirs(expr_dir)
        if self.opt.isTrain:
            file_name = os.path.join(expr_dir, 'opt_train.txt')
        else:
            file_name = os.path.join(expr_dir, 'opt_test.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('-----------------Train Options----------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('---------------------End----------------------\n')
        return self.opt


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./data/results', help='save results here')
        self.parser.add_argument('--phase', type=str, default='train', help='the dataset folder, e.g. train, test, val')
        self.parser.add_argument('--base_model_name', type=str, default='min24_org.model', help='which model to load')
        # self.parser.add_argument('--base_model_name', type=str, default='min24-improvement-1238-0.83.hdf5', help='which model to load')
        self.parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
        self.isTrain = False


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--epoch', type=int, default=50, help='number of epoch')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate of loss function')
        self.parser.add_argument('--plot', action='store_true', default=False, help='plot val_acc and val_loss figure')
        self.isTrain = True
