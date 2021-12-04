import array
import random
from math import ceil

import cv2.ml
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
from cv2 import ml_KNearest
from numpy import ndarray

# References
# Digit OCR Training: https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
# Draw Digit with Font: https://stackoverflow.com/questions/37191008/load-truetype-font-to-opencv

FONT_HEIGHT = 28
FONT_WIDTH = 15
LINE_LENGTH = 20


def generate_train_image(chars: str):
    REPEAT = 16

    H_BORDER = 32
    V_BORDER = 32

    char_list = list(chars)
    char_seq = []
    for i in range(REPEAT):
        random.shuffle(char_list)
        for char in char_list:
            char_seq.append(char)
    char_seq_copy = char_seq.copy()

    width = FONT_WIDTH * LINE_LENGTH + H_BORDER
    height = FONT_HEIGHT * ceil(len(char_seq) / LINE_LENGTH) + V_BORDER
    img = np.zeros((height, width, 3), np.uint8)
    b, g, r, a = 255, 255, 255, 0

    font = ImageFont.truetype("test/train_fonts/NotoSansSC-Light.otf", FONT_HEIGHT)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    x = round(H_BORDER / 2)
    y = round((V_BORDER - 16) / 2)
    while len(char_seq) > 0:
        text = "".join(char_seq[:LINE_LENGTH])
        del char_seq[:LINE_LENGTH]

        draw.text((x, y), text, font=font, fill=(b, g, r, a))
        y += FONT_HEIGHT

    img = np.array(img_pil)
    return char_seq_copy, img


def generate_samples_responses():
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    samples = np.empty((0, 100))
    responses = []
    i = 0
    for contour in reversed(contours):
        [x, y, w, h] = cv.boundingRect(contour)
        if FONT_WIDTH + 4 > w > FONT_WIDTH - 4 and FONT_HEIGHT + 4 > h > FONT_HEIGHT - 4:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), None, cv.LINE_AA)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv.resize(roi, (10, 10))

            cv.imshow('contour', img)
            # cv.waitKey()

            responses.append(char_seq[i])
            sample = roismall.reshape((1, 100))
            samples = np.append(samples, sample)

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), None, cv.LINE_AA)
            i += 1
    cv.destroyWindow('contour')
    responses = np.array(responses, np.intc)
    responses = responses.reshape((responses.size, 1))
    samples = samples.reshape((responses.size, int(samples.size / responses.size)))
    return np.float32(samples), np.float32(responses)


def obtain_model() -> ml_KNearest:
    model = cv.ml.KNearest_create()
    model.train(samples, cv.ml.ROW_SAMPLE, responses)
    return model


def test():
    (test_char_seq, test_img) = generate_train_image("0123456789")
    out = np.zeros(test_img.shape, np.uint8)
    gray = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        [x, y, w, h] = cv.boundingRect(contour)
        if FONT_WIDTH + 4 > w > FONT_WIDTH - 4 and FONT_HEIGHT + 4 > h > FONT_HEIGHT - 4:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), None, cv.LINE_AA)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=12)
            best_result = results[0][0]
            string = str(int(best_result))
            cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
    return test_img, out


def find_digits(img):
    digits = []
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        [x, y, w, h] = cv.boundingRect(contour)
        if FONT_WIDTH + 4 > w > FONT_WIDTH - 4 and FONT_HEIGHT + 4 > h > FONT_HEIGHT - 4:
            digits.append(img[y:y + h, x:x + w])
    return reversed(digits)  # digits are often be found backward


def match_digit(model, digit):
    digit_small = cv.resize(digit, (10, 10))
    digit_small = digit_small.reshape((1, 100))
    digit_small = np.float32(digit_small)
    return model.findNearest(digit_small, k=12)


if __name__ == '__main__':
    char_seq, img = generate_train_image("0123456789")
    samples, responses = generate_samples_responses()
    model = obtain_model()
    # test_img, out = test()

    test_img = cv.imread("test/test_images/number_0.png")
    test_img = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    test_img = cv.GaussianBlur(test_img, (5, 5), 0)
    test_img = cv.adaptiveThreshold(test_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    digits = find_digits(test_img)
    for digit in digits:
        retval, results, neigh_resp, dists = match_digit(model, digit)
        print(retval)
        cv.imshow(None, digit)
        cv.waitKey()

    cv.destroyAllWindows()
    cv2.waitKey()
