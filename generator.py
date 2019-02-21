from keras.preprocessing import image
import numpy as np
import os
import util


class Generator:
    def __init__(self, opt):
        self.opt = opt
        self.datagen = image.ImageDataGenerator()
        self.dict = util.create_dict(opt.cap_scheme)
        if opt.isTune:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.data_load('real')
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.data_load('synthetic')

    # text -> vec
    # return character position in dictionary in vector form
    def text2vec(self, text):
        text_len = len(text)
        if text_len > self.opt.cap_len:
            raise ValueError(
                'The max length of this captcha is {}'.format(self.opt.cap_len))
        if self.opt.char_set_len != len(self.dict):
            raise ValueError(
                'The number of characters does not match to the dict')
        vector = np.zeros(self.opt.cap_len * self.opt.char_set_len)

        def char2pos(c):
            k = -1
            for (key, value) in self.dict.items():
                if value == c:
                    k = key
                    return k
            if k == -1:
                raise ValueError('Wrong with dict or text')

        for i, c in enumerate(text):
            idx = i * self.opt.char_set_len + char2pos(c)
            vector[idx] = 1

        return vector

    # load data for keras model
    def data_load(self, label):
        data_path = os.path.join(self.opt.dataroot, self.opt.cap_scheme,
                                 label)  # ex. ../data/min24/synthetic
        self.num_train_samples = min(self.opt.train_size,
                                     len(os.listdir(os.path.join(data_path,
                                                                 'train'))))  # number of train data in directory
        self.num_test_sample = min(2000,
                                   len(os.listdir(os.path.join(data_path,
                                                               'test'))))  # number of test data in directory

        # load training set
        x_train = np.empty((self.num_train_samples, self.opt.loadHeight,
                            self.opt.loadWidth, 1), dtype='uint8')  # initialize
        y_train = np.empty(
            (self.num_train_samples, self.opt.cap_len * self.opt.char_set_len),
            dtype='uint8')  # initialize
        train_labels = util.load_label(
            os.path.join(data_path, label + '_train.txt'))
        for i in range(self.num_train_samples):
            img_name = os.path.join(data_path, 'train', str(i) + '.png')
            x_train[i, :, :, :] = util.load_image(img_name)
            try:
                y_train[i, :] = self.text2vec(train_labels[i])
            except:
                print(i)
        # load test set
        x_test = np.empty(
            (self.num_test_sample, self.opt.loadHeight, self.opt.loadWidth, 1),
            dtype='uint8')
        y_test = np.empty(
            (self.num_test_sample, self.opt.cap_len * self.opt.char_set_len),
            dtype='uint8')
        test_labels = util.load_label(
            os.path.join(data_path, label + '_test.txt'))
        for i in range(self.num_test_sample):
            img_name = os.path.join(data_path, 'test', str(i) + '.png')
            x_test[i, :, :, :] = util.load_image(img_name)
            try:
                y_test[i, :] = self.text2vec(test_labels[i])
            except:
                print(i)
        return (x_train, y_train), (x_test, y_test)

    # Synthetic data generator
    def synth_generator(self, phase):
        if phase == 'train':
            return self.datagen.flow(self.x_train, self.y_train,
                                     batch_size=self.opt.batchSize)
        elif phase == 'val':
            return self.datagen.flow(self.x_test, self.y_test,
                                     batch_size=self.opt.batchSize,
                                     shuffle=False)
        else:
            raise ValueError('Please input train or val phase')

    # Real data generator
    def real_generator(self, phase):
        return self.synth_generator(phase)
