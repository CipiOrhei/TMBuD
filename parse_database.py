import glob
import os
import shutil
import cv2
import thinning
import numpy as np
from scipy.io import loadmat, savemat
from numba import jit

# edit here the new locations of the raw data and where to add the information

CSV_PARSING_FILE = r"c:\repos\eecvf\TestData\CM_Dataset_small\DATASET SPLIT.csv"

DATASET_LOCATION = r"c:\repos\eecvf\TestData\CM_Dataset_small"
INPUT_IMG_FOLDER = "images"
INPUT_EDGE_FOLDER = "gt_edge"
INPUT_LABEL_FOLDER = "gt_label"

EXTENSION = 'png'

OUTPUT_FOLDER = r"c:\repos\eecvf\TestData\CM_building_dataset"

BACKGROUND =    (0,     0,      0)
SKY =           (255,   0,      0)
VEGETATION =    (0,     255,    0)
BUILDING =      (125,   125,    0)
WINDOW =        (0,     255,    255)
GROUND =        (125,   125,    125)
NOISE =         (0,     0,      255)
DOOR =          (0,     125,    125)

COLORS = [BACKGROUND, BUILDING, DOOR, WINDOW, SKY, VEGETATION, GROUND, NOISE]


def read_csv_file():
    file = open(CSV_PARSING_FILE, "r")
    # get dictionary fields from csv header
    fields = file.readline().split(',')
    # eliminate new line character
    fields[-1] = fields[-1].split('\n')[0]
    list_images = list()

    for line in file.readlines():
        data = line.split(',')
        new_obj = dict()

        for idx_field in range(len(fields)):
            new_obj[fields[idx_field]] = data[idx_field]

        list_images.append(new_obj)

    return list_images


def create_img_sets(list_img, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'img'))
    print('OLD IMG FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        output_folder = os.path.join(OUTPUT_FOLDER, 'img', el['Dataset'], 'png')

        input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, el['Picture Name'] + '.png')
        output_file = os.path.join(output_folder, el['Picture Name'] + '.png')

        if verbose:
            img = cv2.imread(input_file)
            cv2.imshow(str(el['Picture Name']), img)
            cv2.waitKey(1)

        if not os.path.exists(os.path.join(output_folder)):
            os.makedirs(os.path.join(output_folder))

        shutil.copyfile(input_file, output_file)

    # check files.
    print("IMG TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'img', 'TRAIN', 'png')))
    print("IMG TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'img', 'TRAIN', 'png'))))
    print("IMG VAL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'img', 'VAL', 'png')))
    print("IMG VAL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'img', 'VAL', 'png'))))
    print("IMG TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'img', 'TEST', 'png')))
    print("IMG TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'img', 'TEST', 'png'))))


def remove_isolated_px(img):
    # load image, ensure binary, remove bar on the left
    input_image_comp = cv2.bitwise_not(img)  # could just use 255-img

    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)

    hitormiss1 = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel1)
    hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
    hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
    hitormiss_comp = cv2.bitwise_not(hitormiss)  # could just use 255-img
    del_isolated = cv2.bitwise_and(img, img, mask=hitormiss_comp)

    return del_isolated


def create_edge_sets(list_img, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'edge'))
    print('OLD EDGE FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        output_folder = os.path.join(OUTPUT_FOLDER, 'edge', el['Dataset'], 'png')
        output_folder_mat = os.path.join(OUTPUT_FOLDER, 'edge', el['Dataset'], 'mat')

        input_file = os.path.join(DATASET_LOCATION, INPUT_EDGE_FOLDER, el['Picture Name'] + '.png')
        output_file = os.path.join(output_folder, el['Picture Name'] + '.' + EXTENSION)
        output_file_mat = os.path.join(output_folder_mat, el['Picture Name'] + '.mat')

        img = cv2.imread(input_file, cv2.cv2.IMREAD_GRAYSCALE)
        # eliminate pixels that are not 255
        ret, img_final = cv2.threshold(src=img, thresh=1, maxval=255, type=cv2.NORM_MINMAX)
        # eliminate lines that are more than 2 px width
        img_final = thinning.guo_hall_thinning(img_final.copy())
        # eliminate isolated pixels
        img_final = remove_isolated_px(img_final)

        mat_file = []
        mat_file.append({'Segmentation': np.zeros(img_final.shape, dtype=np.dtype('H')), 'Boundaries': img_final // 255})

        if verbose:
            cv2.imshow('img', img)
            cv2.imshow('processed', img_final)
            cv2.imshow('diff', img - img_final)
            cv2.waitKey(500)

        if not os.path.exists(os.path.join(output_folder)):
            os.makedirs(os.path.join(output_folder))

        if not os.path.exists(os.path.join(output_folder_mat)):
            os.makedirs(os.path.join(output_folder_mat))

        savemat(output_file_mat, mdict={'groundTruth': mat_file})
        cv2.imwrite(output_file, img_final)

    # check files.
    print("EDGE TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'edge', 'TRAIN', 'png')))
    print("EDGE TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'edge', 'TRAIN', 'png'))))
    print("EDGE VAL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'edge', 'VAL', 'png')))
    print("EDGE VAL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'edge', 'VAL', 'png'))))
    print("EDGE TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'edge', 'TEST', 'png')))
    print("EDGE TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'edge', 'TEST', 'png'))))


@jit(nopython=True)
def correct_label(class_img, final_img, verbose, COLORS_COPY):
    thr = 35
    for w in range(0, class_img.shape[0]):
        for h in range(0, class_img.shape[1]):
            if class_img[w][h] == 0:
                # values_around = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
                values_around = [0] * len(COLORS_COPY)
                original_value = (final_img[w][h]).copy()
                fixed = False
                max_value = 0

                for color in range(1, len(COLORS_COPY)):
                    if (max(0, final_img[w][h][0] - thr) <= COLORS_COPY[color][0] <= min(255, final_img[w][h][0] + thr)) and\
                        (max(0, final_img[w][h][1] - thr) <= COLORS_COPY[color][1] <= min(255, final_img[w][h][1] + thr)) and\
                        (max(0, final_img[w][h][2] - thr) <= COLORS_COPY[color][2] <= min(255, final_img[w][h][2] + thr)):
                        final_img[w][h] = COLORS_COPY[color]
                        class_img[w][h] = color
                        fixed = True
                        if verbose:
                            print('CORRECT VALUES OF PIXEL ON (', w, ",", h, ') FROM ', original_value, 'TO', final_img[w][h], 'AND CLASS FROM 0 TO', color)
                        break

                if not fixed:
                    for i in range(-2, 3, 1):
                        for j in range(-2, 3, 1):
                            if (0 < w - i < class_img.shape[0]) and (0 < h - j < class_img.shape[1]):
                                v = class_img[w - i][h - j]
                                if 0 < v <= 7:
                                    values_around[class_img[w - i][h - j]] += 1
                                else:
                                    values_around[0] += 1

                    max_value = values_around.index(max(values_around))

                    if max_value != 0:
                        class_img[w][h] = max_value
                        final_img[w][h] = COLORS_COPY[max_value]
                        if verbose:
                            print('CHANGED VALUES OF PIXEL ON (', w, ",", h, ') FROM ', original_value, 'TO', final_img[w][h], 'AND CLASS FROM 0 TO', max_value)
                    else:
                        class_img[w][h] = 255
                        final_img[w][h] = COLORS_COPY[0]
                        if verbose:
                            print('COULD NOT FIX (', w, ",", h, ') value is: ', original_value)

    return class_img, final_img


def check_values_in_label_image(img, name, verbose):
    final_img = img.copy()
    class_img = np.zeros(img.shape[0:2])
    idx_tmp = False

    for el in range(1, len(COLORS)):
        ss = COLORS[el]
        t = np.logical_and(np.logical_and(img[:, :, 0] == COLORS[el][0], img[:, :, 1] == COLORS[el][1]), img[:, :, 2] == COLORS[el][2])
        class_img += t * el

    class_img = class_img.astype(np.uint8)

    errors = (class_img == 0).sum()

    while errors > 0:
        if verbose:
            print("FIXING ", name, " IMAGE")
        class_img, final_img = correct_label(class_img, final_img, verbose, COLORS.copy())
        new_errors = (class_img == 0).sum()

        if errors - new_errors == 0 :
            print('PIXELS STILL TO FIX: ', new_errors, ' PLEASE CORRECT MANUAL')
            break
        # elif new_errors > 10:
        #     print('PIXELS STILL TO FIX :', new_errors)
        #     break

        errors = new_errors

    return final_img, class_img


def create_label_sets(list_img, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'label'))
    print('OLD LABEL FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        output_folder = os.path.join(OUTPUT_FOLDER, 'label', el['Dataset'], 'png')
        output_folder_classes = os.path.join(OUTPUT_FOLDER, 'label', el['Dataset'], 'classes')

        input_file = os.path.join(DATASET_LOCATION, INPUT_LABEL_FOLDER, el['Picture Name'] + '.png')
        output_file = os.path.join(output_folder, el['Picture Name'] + '.' + EXTENSION)
        output_file_classes = os.path.join(output_folder_classes, el['Picture Name'] + '.png')

        img = cv2.imread(input_file)
        img_final, img_final_classes = check_values_in_label_image(img.copy(), el['Picture Name'], verbose)

        if verbose:
            cv2.imshow('img', img)
            cv2.imshow('processed', img_final)
            # cv2.imshow('classes', img_final_classes)
            cv2.imshow('diff', img - img_final)
            cv2.waitKey(1000)

        if not os.path.exists(os.path.join(output_folder)):
            os.makedirs(os.path.join(output_folder))

        if not os.path.exists(os.path.join(output_folder_classes)):
            os.makedirs(os.path.join(output_folder_classes))

        cv2.imwrite(output_file, img_final)
        cv2.imwrite(output_file_classes, img_final_classes)

    # check files.
    print("LABEL TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'label', 'TRAIN', 'png')))
    print("LABEL TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'label', 'TRAIN', 'png'))))
    print("LABEL VAL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'label', 'VAL', 'png')))
    print("LABEL VAL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'label', 'VAL', 'png'))))
    print("LABEL TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'label', 'TEST', 'png')))
    print("LABEL TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'label', 'TEST', 'png'))))


if __name__ == "__main__":
    list_img = read_csv_file()
    create_img_sets(list_img, verbose=False)
    create_edge_sets(list_img, verbose=False)
    create_label_sets(list_img, verbose=False)
