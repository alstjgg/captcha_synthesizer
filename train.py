from option import TrainOptions
from option import TestOptions
from generator import Generator
from model import create_model


def train():
    opt = TrainOptions().parse()
    generator = Generator(opt)
    train_generator = generator.synth_generator('train')
    val_generator = generator.synth_generator('val')
    model = create_model(opt)

    print('num_train_sample = {}, num_test_sample = {}, batchSize = {}'.format(generator.num_train_samples, generator.num_test_sample, opt.batchSize))
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=generator.num_train_samples // opt.batchSize,
        epochs=opt.epoch,
        validation_data=val_generator,
        validation_steps=generator.num_test_sample // opt.batchSize,
        class_weight='auto'
    )

    # save model
    model.save(history)

    # plot validation accuracy and loss
    if opt.plot:
        print('plot training history')
        model.plot_training(history)
    model.plot_training(history)


def test():
    opt = TestOptions().parse()
    opt.isTune = False
    # model_path = os.path.join(opt.checkpoints_dr, opt.cap_scheme, opt.model_name)
    model = create_model(opt)
    model.load_weight()

    generator = Generator(opt)
    # test_generator = generator.synth_generator(opt.phase)
    test_data = (generator.x_test, generator.y_test)
    model.predict(test_data, batch_size=opt.batchSize)

