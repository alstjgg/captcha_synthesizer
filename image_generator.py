import numpy as np
import cv2
import string
import math
import os


# Create clean captcha image with only security functions that cannot be removed
# Security Functions that can be used
# 1. Character Overlapping
# # 2. Character Set
# # 3. Font Style
# 4. Waving
# # 5. Distortion
# # 6. Rotation
# 7. Font Color


# 2. Character Set
CHAR_SET = string.digits
NUM_LETTER = 6

# 3. Font Style
FONTS = ['FONT_HERSHEY_TRIPLEX']
FONT_SCALE = 1.1
FONT_THICKNESS = 2


class Captcha:
    def __init__(self, width, high, folder='./data/clean'):
        self.letter = [i for i in CHAR_SET]
        self.width, self.high = width, high
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    # 1. Character Overlapping
    def overlap(self, init, start, letter_width, on=True):
        if start is not init and on:
            return start - np.random.randint(letter_width/5, letter_width/3)
        elif on:
            return
        else:
            return start

    # 5. Distortion
    def distort(self, img):
        high, width, _ = img.shape
        tmp_img = img.copy()
        tmp_img.fill(255)

        coef_vertical = np.random.randint(3, 8)
        coef_horizontal = np.random.choice([4, 5, 6]) * math.pi / width
        scale_biase = np.random.randint(0, 360) * math.pi / 180

        def new_coordinate(x, y):
            return int(x+coef_vertical*math.sin(coef_horizontal*y+scale_biase))

        for y in range(width):
            for x in range(high):
                new_x = new_coordinate(x, y)
                try:
                    tmp_img[x, y, :] = img[new_x, y, :]
                except IndexError:
                    pass

        img[:, :, :] = tmp_img[:, :, :]

    # 6. Roatation
    def rotate(self, img):
        tmp_img = img.copy()
        tmp_img.fill(255)
        tile_angle = np.random.randint(
            int(100*-math.pi/6), int(100*math.pi/6)
        ) / 100
        high, width, _ = img.shape
        for y in range(width):
            for x in range(high):
                new_y = int(y + (x-high/2)*math.tanh(tile_angle))
                try:
                    tmp_img[x, new_y, :] = img[x, y, :]
                except IndexError:
                    pass
        img[:, :, :] = tmp_img[:, :, :]

    # 7. Font Color
    def color(self, on=True):
        if on:
            return tuple(int(np.random.choice(range(0, 156))) for _ in range(3))
        else:
            return 0, 0, 0

    def _draw_basic(self, img, text):
        init = int((self.width/NUM_LETTER)/8)
        start = init
        for word in (text):
            font_face = getattr(cv2, np.random.choice(FONTS))

            (width, high), _ = cv2.getTextSize(word, font_face, FONT_SCALE, FONT_THICKNESS)
            vertical_range = self.high - high
            delta_high = int(3 * vertical_range / NUM_LETTER)

            this_x = self.overlap(init, start, width, on=False)        # Turn Overlapping on or not
            this_y = self.high - delta_high
            start += width
            bottom_left_coordinate = (this_x, this_y)
            font_color = self.color(on=False)       # Turn letter Color on or not
            cv2.putText(img, word, bottom_left_coordinate, font_face,
                        FONT_SCALE, font_color, FONT_THICKNESS)

    def create_img(self, text):
        img = np.zeros((self.high, self.width, 3), np.uint8)
        img.fill(255)       # Create empty image
        self._draw_basic(img, text)
        # self.rotate(img)
        # self.noise(img)
        # self.distort(img)
        # self.line(img)

        cv2.imwrite('{}/{}.jpg'.format(self.folder, text), img)

    def batch_create_img(self, total):
        exist = set()
        while(len(exist)) < total:
            word = ''.join(np.random.choice(self.letter, NUM_LETTER))
            if word not in exist:
                exist.add(word)
                self.create_img(word)

        print('\r{0: .2f}% complete..'.format((len(exist)*100) / total), end='')
        print('\n{} captchas generated and saved in {}.'.format(len(exist), self.folder))


# if __name__ == '__main__':
c = Captcha(140, 53)
c.batch_create_img(3)
