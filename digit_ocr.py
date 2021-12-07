import random
from math import ceil

import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from cv2 import ml_KNearest

FONT_H = 28
FONT_W = 15
LINE_LENGTH = 20

DIGIT_BORDER_TOP = 116 / 173  # =67.05%
DIGIT_BORDER_BOTTOM = 21 / 173  # =~12.14%
DIGIT_BORDER_LEFT = 56 / 172  # =~32.56%
DIGIT_BORDER_RIGHT = 28 / 172  # =~16.28%


def generate_train_image(chars: str):
    REPEAT = 100

    H_BORDER = 32
    V_BORDER = 32

    char_list = list(chars)
    char_seq = []
    for i in range(REPEAT):
        random.shuffle(char_list)
        for char in char_list:
            char_seq.append(char)
    char_seq_copy = char_seq.copy()

    width = FONT_W * LINE_LENGTH + H_BORDER
    height = FONT_H * ceil(len(char_seq) / LINE_LENGTH) + V_BORDER
    img = np.zeros((height, width, 3), np.uint8)
    b, g, r, a = 255, 255, 255, 0

    font = ImageFont.truetype("test/train_fonts/NotoSansSC-Light.otf", FONT_H)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    x = round(H_BORDER / 2)
    y = round((V_BORDER - 16) / 2)
    while len(char_seq) > 0:
        text = "".join(char_seq[:LINE_LENGTH])
        del char_seq[:LINE_LENGTH]

        draw.text((x, y), text, font=font, fill=(b, g, r, a))
        y += FONT_H

    img = np.array(img_pil)
    return char_seq_copy, img


def generate_samples_responses(img, char_seq):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    samples = np.empty((0, 100))
    responses = []
    i = 0
    for contour in reversed(contours):
        [x, y, w, h] = cv.boundingRect(contour)
        if FONT_W + 4 > w > FONT_W - 4 and FONT_H + 4 > h > FONT_H - 4:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), None, cv.LINE_AA)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv.resize(roi, (10, 10))

            responses.append(char_seq[i])
            sample = roismall.reshape((1, 100))
            samples = np.append(samples, sample)

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), None, cv.LINE_AA)
            i += 1
    responses = np.array(responses, np.intc)
    responses = responses.reshape((responses.size, 1))
    samples = samples.reshape((responses.img_size, int(samples.size / responses.img_size)))
    return np.float32(samples), np.float32(responses)


def obtain_model(samples, responses) -> ml_KNearest:
    model = cv.ml.KNearest_create()
    model.train(samples, cv.ml.ROW_SAMPLE, responses)
    return model


def find_digits(img, scale):
    digits = []
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.GaussianBlur(img, (round_odd(5 * scale), round_odd(5 * scale)), 0)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours_boxes = list(map(lambda c: cv.boundingRect(c), contours))
    contours_boxes.sort(key=lambda b: b[0])
    for contour_box in contours_boxes:
        [x, y, w, h] = contour_box
        # print(w, h)
        # rect_img = cv.cvtColor(img.copy(), cv.COLOR_GRAY2RGB)
        # rect_img = cv.rectangle(rect_img, (x, y), (x + w, y + h), color=(0, 0, 255))
        # cv.imshow("rect_img", rect_img)
        # cv.waitKey()
        if (FONT_W + 6) * scale >= w >= (FONT_W - 6) * scale and (FONT_H + 6) * scale >= h >= (FONT_H - 6) * scale:
            digits.append(img[y:y + h, x:x + w])
    return digits


def match_digit(model, digit):
    digit_small = cv.resize(digit, (10, 10))
    digit_small = digit_small.reshape((1, 100))
    digit_small = np.float32(digit_small)
    retval, results, neigh_resp, dists = model.findNearest(digit_small, k=12)
    neigh_resps_1d = neigh_resp.reshape(-1)
    if np.all(neigh_resps_1d == neigh_resps_1d[0]):
        return int(retval)
    else:
        return -1


def round_odd(n):
    answer = round(n)
    if answer % 2 == 1:
        return answer
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return answer + 1
    else:
        return answer - 1


def crop_numbers(img_match):
    img_result_h, img_result_w = img_match.shape[:2]
    img_result_btop = round(DIGIT_BORDER_TOP * img_result_h)
    img_result_bbottom = round(DIGIT_BORDER_BOTTOM * img_result_h)
    img_result_bleft = round(DIGIT_BORDER_LEFT * img_result_w)
    img_result_bright = round(DIGIT_BORDER_RIGHT * img_result_w)
    return img_match[
           img_result_btop:img_result_h - img_result_bbottom,
           img_result_bleft: img_result_w - img_result_bright
           ]
