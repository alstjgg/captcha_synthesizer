import os
from PIL import Image
import numpy as np
import shutil


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


# if directory does not already exist, create path
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# return image saved in path
def load_image(path):
    with Image.open(path).convert('L') as img:      # convert image to b&w
        img = np.array(img)
        img = np.expand_dims(img, axis=2)       # expand shape of array, new axis is placed in position 2
        return img


#
def load_label(txt):        # txt = dataroot, cap_scheme, label, label + '_train/test.txt'
    with open(txt, 'r') as fp_txt:
        labels = fp_txt.read()
        labels = labels.split('#')
        return labels


# return dictionary created from character set used in captcha scheme
def char2dict(char_set):
    dict = {}
    for i, char in enumerate(char_set):
        dict[i] = char
    return dict


# return character dictionary used in given captcha scheme
def create_dict(cap_scheme):
    char_set = []
    scheme = CaptchaSchemes()
    if cap_scheme == 'min24':
        char_set = scheme.min24
    return char2dict(char_set)


# vector -> text
# vector: indicate where character position in dictionary
def vec2text(vect, dict):
    text = ''
    for i, key in enumerate(vect):
        value = dict[key]
        text += value
    return text


# print result
def print_predict(opt):
    scheme = CaptchaSchemes()
    preds = opt.preds
    preds_right_num = 0     # initialize
    char_set = []       # initialize
    total_num = len(preds)
    print('total_num : ', total_num)
    if opt.cap_scheme == 'min24':       # if the scheme is 민원24
        char_set = scheme.min24     # return character set of min24
    dict = char2dict(char_set)      # and create dictionary based on set
    if opt.isTune:
        label = 'real'
    else:
        label = 'synthetic'    # create and sort labels
    if opt.phase == 'train':
        real_label_name = os.path.join(opt.dataroot, opt.cap_scheme, label, label + '_train.txt')
    else:       # if phase is validation or testing
        real_label_name = os.path.join(opt.dataroot, opt.cap_scheme, label, label + '_test.txt')
    real_labels = load_label(real_label_name)
    for i, pred in enumerate(preds):        # for all images
        pred = vec2text(pred, dict)     # recognized text value of iamge
        if pred.lower() == real_labels[i].lower():      # convert to lowercase and compare
            preds_right_num += 1
            print('Correct Prediction')
        print('No.{} \t Predict: {} \t Real: {}' .format(i, pred, real_labels[i]))
    print('Recognition Accuracy : ', preds_right_num/total_num)


# define character set of each captcha scheme
class CaptchaSchemes:
    def __init__(self):
        self.min24 = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]


#
if __name__ == '__main__':
    dict = create_dict('min24')     # create dictionary for min24 captcha scheme
    c = 'Z'

    def char2pos(c):
        k = -1
        for (key, value) in dict.items():
            if value == c:
                k = key
        return k

    key = char2pos(c)
    print(key)


def create_labels():
    img_path = '../data/min24/synthetic/min240_4246/'
    img_save_path = '../data/min24/synthetic/min244000-4246/'

    j = 0
    for i in range(4000, 4247, 1):
        shutil.copyfile(img_path + str(i) + '.jpg',
                        img_save_path + str(j) + '.jpg')
        j += 1
