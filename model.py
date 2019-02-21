from keras import models, layers, optimizers, losses
import os
import numpy as np
import util
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import h5py


def create_model(opt):
    model = None
    model = LeNetModel()
    model.initialize(opt)
    return model


class LeNetModel:
    def initialize(self, opt):
        os.environ["PATH"] += os.pathsep + 'C:/Users/argos/PycharmProjects/keras/venv/graphviz-2.38/release/bin/'
        self.opt = opt
        self.batchSize = opt.batchSize
        self.keep_prob = opt.keep_prob
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.cap_scheme, str(opt.train_size))

        self.model = self.define_lenet5()

        if self.opt.isTrain:
            adam = optimizers.adam(lr=opt.lr)       # adam optimizer with learning rate from option
            self.model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])       # compile model
        if self.opt.callbacks:
            bast_model_name = os.path.join(self.save_dir, self.opt.cap_scheme + '-improvement-{epoch:02d}.hdf5')
            # bast_model_name = os.path.join(self.save_dir, self.opt.cap_scheme + '-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
            checkpoint = ModelCheckpoint(bast_model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            self.callbacks_list = [checkpoint]
        else:
            self.callbacks_list = None

    def define_lenet5(self):
        model = models.Sequential()
        # 5 Convolutional Layers
        model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                input_shape=(
                                self.opt.loadHeight, self.opt.loadWidth, 1)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(layers.Dropout(self.keep_prob))

        # Fully Connected Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(3072))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(self.keep_prob))

        model.add(layers.Dense(self.opt.cap_len * self.opt.char_set_len))
        model.add(layers.Activation('softmax'))

        return model

    def fit_generator(self,
                      generator,
                      steps_per_epoch=20,
                      epochs=20,
                      validation_data=None,
                      validation_steps=2,
                      class_weight='auto',
                      callbacks=None
                      ):
        return self.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=callbacks
        )

    def predict(self, test_data, batch_size=32):
        x_test, y_test = test_data[0], test_data[1]
        # Normalize data
        x_test = (x_test - 255 / 2) / (255 / 2)
        preds = self.model.predict(x=x_test, batch_size=batch_size)
        preds = np.argmax(
            preds.reshape(-1, self.opt.cap_len, self.opt.char_set_len), axis=2)
        self.opt.preds = preds
        self.opt.reals = y_test
        util.print_predict(self.opt)

    def predict_generator(self, generator, steps=1, max_queue_size=100):
        preds = self.model.predict_generator(
            generator=generator,
            steps=steps,
            max_queue_size=max_queue_size
        )
        preds = np.argmax(
            preds.reshape(-1, self.opt.cap_len, self.opt.char_set_len), axis=2)
        self.opt.preds = preds
        util.print_predict(self.opt)

    def save_model(self):
        if self.opt.isTune:
            model_checkpoint_base_name = os.path.join(self.save_dir,
                                                      self.opt.cap_scheme + '_finetune.model')
        else:
            model_checkpoint_base_name = os.path.join(self.save_dir,
                                                      self.opt.cap_scheme + '_org.model')
        plot_model(self.model, to_file=model_checkpoint_base_name + '.png',
                   show_shapes=True)
        print('train model: ' + model_checkpoint_base_name)
        self.model.save_weights(model_checkpoint_base_name)

    def load_weight(self):
        model_checkpoint_base_name = os.path.join(self.save_dir,
                                                  self.opt.base_model_name)
        print('test model: ' + model_checkpoint_base_name)
        self.model.load_weights(model_checkpoint_base_name)

    def save(self, history):
        # save loss to opt_train.txt
        print('Print the training history:')
        with open(self.save_dir + '/opt_train.txt', 'a') as opt_file:
            opt_file.write(str(history.history))
        self.save_model()

    def setup_to_finetune(self):
        print('layers number of the model {} '.format(len(self.model.layers)))
        for layer in self.model.layers[:self.opt.nb_retain_layers]:
            layer.trainable = False
        for layer in self.model.layers[self.opt.nb_retain_layers:]:
            weights = layer.get_weights()
            weights = weights * 0
            layer.trainable = True
        self.model.compile(optimizer=optimizers.adam(lr=self.opt.lr),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def visualize_model(self, input_images, layer_index):
        output_layer = self.model.layers[layer_index].output
        input_layer = self.model.layers[0].input
        output_fn = K.function([input_layer], [output_layer])
        output_image = output_fn([input_images])[0]
        print("Output image shape:", output_image.shape)

        fig = plt.figure()
        plt.title("%dth convolutional later view of output" % layer_index)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

        for i in range(32):
            ax = fig.add_subplot(4, 8, i + 1)
            im = ax.imshow(output_image[0, :, :, i], cmap='Greys')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.07, 0.05, 0.821])
        fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()

        plt.show()

    def plot_training(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        print("run to here")
        print(plt.get_backend())

        plt.plot(epochs, acc, 'r.', label='train_acc')
        plt.plot(epochs, val_acc, 'b', label='val_acc')
        plt.title("Training and validation accuracy")
        plt.legend(loc=0, ncol=2)
        plt.savefig('./data/accuracy.png')

        plt.figure()
        plt.plot(epochs, loss, 'r.', label='train_loss')
        plt.plot(epochs, val_loss, 'b-', label='val_loss')
        plt.title("Training and validation loss")
        plt.legend(loc=0, ncol=2)
        plt.savefig('./data/loss.png')
        plt.show()
