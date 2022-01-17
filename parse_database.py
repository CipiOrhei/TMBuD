import glob
import os
import shutil
import cv2
import thinning
import numpy as np
from scipy.io import loadmat, savemat
from numba import jit
import argparse
import matplotlib.pyplot as plt
import sys

# edit here the new locations of the raw data and where to add the information
EXTENSION = 'png'

OK_VARIANTA = ['STANDARD', 'BUILDING_DET_3', 'BUILDING_DET_3_NIGHT', 'BUILDING_DET_3_N', 'SEMSEG_EVAL_FULL']


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

        new_obj['Picture Name'] = new_obj['Picture Name'].rjust(5, '0')
        new_obj['Object class'] = int(new_obj['Picture Name'][:3]) + 1
        new_obj['Object view'] = int(new_obj['Picture Name'][-2:])

        list_images.append(new_obj)

    return list_images


def check_images_object(list_img, create_day_missing=False, create_night_missing=False):
    MIN_NR_DAY_IMG = 5
    MIN_NR_NIGHT_IMG = 1

    check_dict = dict()

    print('Checking csv master file')
    list_missing_edge = list()
    list_missing_label = list()

    for img in list_img:
        if img['Picture Name'][:3] not in check_dict.keys():
            check_dict[img['Picture Name'][:3]] = {'DAY': 0, 'NIGHT': 0, 'LABELS': 0, 'EDGE': 0, 'TRAIN_Dataset 3_2': 0,  'TEST_Dataset 3_2': 0}

        if img['GT salient edges'] == 'Done':
            if not os.path.exists(os.path.join(DATASET_LOCATION, INPUT_EDGE_FOLDER , img['Picture Name'] + '.png')):
                list_missing_edge.append(img['Picture Name'])
            else:
                check_dict[img['Picture Name'][:3]]['EDGE'] += 1

        if os.path.exists(os.path.join(DATASET_LOCATION, INPUT_EDGE_FOLDER, img['Picture Name'] + '.png')) and img['GT salient edges'] != 'Done':
            list_missing_edge.append(img['Picture Name'])

        if img['GT labels'] == 'Done':
            if not os.path.exists(os.path.join(DATASET_LOCATION, INPUT_LABEL_FOLDER , img['Picture Name'] + '.png')):
                list_missing_label.append(img['Picture Name'])
            else:
                check_dict[img['Picture Name'][:3]]['LABELS'] += 1

        if os.path.exists(os.path.join(DATASET_LOCATION, INPUT_LABEL_FOLDER, img['Picture Name'] + '.png')) and img['GT labels'] != 'Done':
            list_missing_label.append(img['Picture Name'])

        check_dict[img['Picture Name'][:3]][img['Condition']] += 1

        if img['Dataset 3_2'] != 'None':
            check_dict[img['Picture Name'][:3]][img['Dataset 3_2'] + '_Dataset 3_2'] += 1

    print('Mismarked edge images: ', list_missing_edge)
    print('Mismarked label images: ', list_missing_label)

    print('Checking missing DAY images')
    list_missing = []
    for key in check_dict.keys():
        if check_dict[key]['DAY'] < MIN_NR_DAY_IMG:
            list_missing.append(key)
    print('NOK number of images: ', list_missing)

    if create_day_missing:
        output_folder = os.path.join(OUTPUT_FOLDER, 'missing_day')

        for key in list_missing:
            input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, key + '01.png')
            output_file = os.path.join(output_folder, key + '01.png')

            if not os.path.exists(os.path.join(output_folder)):
                os.makedirs(os.path.join(output_folder))

            shutil.copyfile(input_file, output_file)

    print('Checking missing NIGHT images')
    list_missing = []
    for key in check_dict.keys():
        if check_dict[key]['NIGHT'] < MIN_NR_NIGHT_IMG:
            list_missing.append(key)
    print('NOK number of images: ', list_missing)

    if create_night_missing:
        output_folder = os.path.join(OUTPUT_FOLDER, 'missing_night')

        for key in list_missing:
            input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, key + '01.png')
            output_file = os.path.join(output_folder, key + '01.png')

            if not os.path.exists(os.path.join(output_folder)):
                os.makedirs(os.path.join(output_folder))

            shutil.copyfile(input_file, output_file)

    print('Checking missing Dataset 3_2 images')
    list_missing = []
    for key in check_dict.keys():
        if check_dict[key]['TRAIN_Dataset 3_2'] < 3 or check_dict[key]['TEST_Dataset 3_2'] < 2:
            list_missing.append(key)
    print('NOK number of images: ', list_missing)

    print('Checking missing Sematic Segmentation min images')
    list_missing = []
    for key in check_dict.keys():
        if check_dict[key]['LABELS'] < 2:
            list_missing.append(key)
    print('NOK number of semantic labels images: ', list_missing)


def create_img_sets(list_img, variant, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'img'))
    print('OLD IMG FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        if el['Dataset STANDARD'] != 'None':
            output_folder = os.path.join(OUTPUT_FOLDER, 'img', el[variant], 'png')

            input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, el['Picture Name'] + '.png')
            output_file = os.path.join(output_folder, el['Picture Name'] + '.png')

            if verbose:
                img = cv2.imread(input_file)
                cv2.imshow(str(el['Picture Name']), img)
                cv2.waitKey(1)

            if not os.path.exists(os.path.join(output_folder)):
                os.makedirs(os.path.join(output_folder))

            shutil.copyfile(input_file, output_file)
        else:
            pass

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


def create_edge_sets(list_img, variant, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge'))
    print('OLD EDGE FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        if el['Dataset STANDARD'] != 'None':
            output_folder = os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', el[variant], 'png')
            output_folder_mat = os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', el[variant], 'mat')

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
    print("EDGE TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', 'TRAIN', 'png')))
    print("EDGE TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', 'TRAIN', 'png'))))
    print("EDGE VAL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', 'VAL', 'png')))
    print("EDGE VAL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', 'VAL', 'png'))))
    print("EDGE TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', 'TEST', 'png')))
    print("EDGE TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'edge', 'TEST', 'png'))))


@jit(nopython=True)
def correct_label(class_img, final_img, verbose, COLORS_COPY):
    thr = 25
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


def create_label_sets(list_img, variant, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label'))
    print('OLD LABEL FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        if el['Dataset STANDARD'] != 'None':
            output_folder = os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', el[variant], 'png')
            output_folder_classes = os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', el[variant], 'classes')

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
    print("LABEL TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', 'TRAIN', 'png')))
    print("LABEL TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', 'TRAIN', 'png'))))
    print("LABEL VAL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', 'VAL', 'png')))
    print("LABEL VAL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', 'VAL', 'png'))))
    print("LABEL TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', 'TEST', 'png')))
    print("LABEL TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'label', 'TEST', 'png'))))


def create_label_sets_full_eval(list_img, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'label_full'))
    print('OLD LABEL FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        if el['GT labels'] != 'None':
            output_folder = os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'label_full', el['SEMSEG_EVAL_FULL'], 'png')
            output_folder_classes = os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'label_full', el['SEMSEG_EVAL_FULL'], 'classes')

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
    print("IMG TRAIN LABEL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'label_full', 'TRAIN', 'png')))
    print("IMG TRAIN LABEL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'label_full', 'TRAIN', 'png'))))
    print("IMG TEST LABEL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'label_full', 'TEST', 'png')))
    print("IMG TEST LABEL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'label_full', 'TEST', 'png'))))


def create_img_sets(list_img, variant, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER,'STANDARD', 'img'))
    print('OLD IMG FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        if el['Dataset STANDARD'] != 'None':
            output_folder = os.path.join(OUTPUT_FOLDER,'STANDARD', 'img', el[variant], 'png')

            input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, el['Picture Name'] + '.png')
            output_file = os.path.join(output_folder, el['Picture Name'] + '.png')

            if verbose:
                img = cv2.imread(input_file)
                cv2.imshow(str(el['Picture Name']), img)
                cv2.waitKey(1)

            if not os.path.exists(os.path.join(output_folder)):
                os.makedirs(os.path.join(output_folder))

            shutil.copyfile(input_file, output_file)
        else:
            pass

    # check files.
    print("IMG TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'img', 'TRAIN', 'png')))
    print("IMG TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'img', 'TRAIN', 'png'))))
    print("IMG VAL DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'img', 'VAL', 'png')))
    print("IMG VAL DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'img', 'VAL', 'png'))))
    print("IMG TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'img', 'TEST', 'png')))
    print("IMG TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'STANDARD', 'img', 'TEST', 'png'))))

def create_img_sets_label_full(list_img, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'img_label_full'))
    print('OLD IMG FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for el in list_img:
        if el['SEMSEG_EVAL_FULL'] != 'None':
            output_folder = os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'img_label_full', el['SEMSEG_EVAL_FULL'], 'png')

            input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, el['Picture Name'] + '.png')
            output_file = os.path.join(output_folder, el['Picture Name'] + '.png')

            if verbose:
                img = cv2.imread(input_file)
                cv2.imshow(str(el['Picture Name']), img)
                cv2.waitKey(1)

            if not os.path.exists(os.path.join(output_folder)):
                os.makedirs(os.path.join(output_folder))

            shutil.copyfile(input_file, output_file)
        else:
            pass

    # check files.
    print("IMG TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'img_label_full', 'TRAIN', 'png')))
    print("IMG TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'img_label_full', 'TRAIN', 'png'))))
    print("IMG TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'img_label_full', 'TEST', 'png')))
    print("IMG TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, 'SEMSEG_EVAL_FULL', 'img_label_full', 'TEST', 'png'))))



def create_img_detection_dataset(list_img, variant,  folder_out, verbose=False):
    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, folder_out))
    print('OLD IMG FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    new_obj_nr = 0
    new_obj_view = 0
    last_obj=0
    for object in list_img:
        if (object[variant] == 'TRAIN') or (object[variant] == 'TEST'):
            if last_obj != object['Object class']:
                last_obj = object['Object class']
                new_obj_nr += 1
                new_obj_view = 1

            object['Object class'] = new_obj_nr
            if object[variant] != 'TEST':
                object['Object view'] = new_obj_view
                new_obj_view += 1

    text_gt = "TEST	TRAIN\n"
    # text_gt = ""
    idx = 1

    for el in list_img:
        if el[variant] != 'None':
            output_folder = os.path.join(OUTPUT_FOLDER, folder_out, el[variant])
            input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, el['Picture Name'] + '.png')

            if el[variant] == 'TRAIN':
                output_file = os.path.join(output_folder, "object{0:04d}_view{1:02d}.png".format(el['Object class'], el['Object view']))
            else:
                output_file = os.path.join(output_folder, "qimg{0:04d}.png".format(idx))
                text_gt += "{0:03d}\t{1:03d}\n".format(idx, el['Object class'])
                idx += 1

            if verbose:
                img = cv2.imread(input_file)
                cv2.imshow(str(el['Picture Name']), img)
                cv2.waitKey(1)

            if not os.path.exists(os.path.join(output_folder)):
                os.makedirs(os.path.join(output_folder))

            shutil.copyfile(input_file, output_file)
        else:
            pass

    file_out = open(os.path.join(OUTPUT_FOLDER, folder_out, 'TMBuD_groundtruth.txt'), 'w')
    file_out.write(text_gt)
    file_out.close()
    # check files.
    print("IMG TRAIN DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, folder_out, 'TRAIN')))
    print("IMG TRAIN DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, folder_out, 'TRAIN'))))
    print("IMG TEST DATASET: ", os.listdir(os.path.join(OUTPUT_FOLDER, folder_out, 'TEST')))
    print("IMG TEST DATASET SIZE: ", len(os.listdir(os.path.join(OUTPUT_FOLDER, folder_out, 'TEST'))))


def do_lossless_JPG(list_img):

    list_avg_img = dict()
    output_folder = os.path.join(OUTPUT_FOLDER, 'lossless_JPG')

    # delete existing folder
    files = glob.glob(os.path.join(OUTPUT_FOLDER, output_folder))
    print('OLD IMG FOLDER IS DELETED')

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    for img in list_img:
        input_file = os.path.join(DATASET_LOCATION, INPUT_IMG_FOLDER, img['Picture Name'] + '.png')
        tmp_img = cv2.imread(input_file)

        if img['Object class'] not in list_avg_img.keys():
            list_avg_img[img['Object class']] = {'values': (tmp_img.astype('float64')).copy(), 'count': 1, 'size': 0}
        else:
            list_avg_img[img['Object class']]['values'] += (tmp_img.astype('float64')).copy()
            list_avg_img[img['Object class']]['count'] += 1

    for obj in list_avg_img.keys():
        new_img = (list_avg_img[obj]['values'] / list_avg_img[obj]['count'])


        if not os.path.exists(os.path.join(output_folder)):
            os.makedirs(os.path.join(output_folder))

        output_file = os.path.join(output_folder, str(obj-1) + '.png')

        cv2.imwrite(output_file, new_img, [cv2.IMWRITE_JPEG_QUALITY, 0])

        list_avg_img[obj]['size'] = os.path.getsize(output_file)/(1024*1024)

    list_size = [list_avg_img[obj]['size'] for obj in list_avg_img.keys()]
    list_val = [list_avg_img[obj]['count'] for obj in list_avg_img.keys()]
    list_obj = list(range(0, len(list_avg_img.keys())))

    fig = plt.gcf()
    fig.set_size_inches(w=15, h=10)
    # plt.plot(list_obj, list_size)
    plt.plot(list_obj, list_val)
    # plt.xlim([0, 256])
    plt.xlabel('Landmark', fontsize=18)
    plt.xlim(0, len(list_obj) + 1)
    plt.xticks(np.arange(0, len(list_obj) + 1, 5))

    # Calculate the simple average of the data
    y_mean = [np.mean(list_val)] * len(list_obj)
    # Plot the average line
    mean_line = plt.plot(list_obj, y_mean, label='Mean', linestyle='--')

    plt.ylabel('Number of images', fontsize=18)
    plt.title('Distribution of images per object', fontsize=18)
    plt.savefig(os.path.join(output_folder, 'plot_imgs_object.png'), bbox_inches='tight', dpi=1200)
    plt.clf()

    fig = plt.gcf()
    fig.set_size_inches(w=15, h=10)
    plt.plot(list_obj, list_size)
    # plt.plot(list_obj, list_val)
    # plt.xlim([0, 256])
    plt.xlabel('Landmark', fontsize=18)
    plt.xlim(0, len(list_obj) + 1)

    # Calculate the simple average of the data
    y_mean = [np.mean(list_size)] * len(list_obj)
    # Plot the average line
    mean_line = plt.plot(list_obj, y_mean, label='Mean', linestyle='--')

    plt.xticks(np.arange(0, len(list_obj) + 1, 5))
    plt.ylabel('lossless JPG size [MB]', fontsize=18)
    plt.title('Distribution of JPG size per object', fontsize=18)
    plt.savefig(os.path.join(output_folder, 'plot_lossless_object.png'), bbox_inches='tight', dpi=1200)
    plt.clf()
        # print(list_avg_img[obj]['size'])
    # pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variant to configure dataset")
    help_text="""
    Create standard TMBuD dataset: STANDARD
    Create building detection dataset 3_2: BUILDING_DET_3
    Create building detection dataset 3_2 plus night test images: BUILDING_DET_3_NIGHT
    Create building detection dataset 3 TRAIN images and n number of TEST images : BUILDING_DET_3_N
    Create building semantic segmentation evaluation dataset with all label images: SEMSEG_EVAL_FULL
    """
    parser.add_argument('--variant', help=help_text, required=False)
    parser.add_argument('--check', help=help_text, required=False)
    parser.add_argument('--lossless_jpeg', help=help_text, required=False)
    args = vars(parser.parse_args())
    print(args['variant'])
    print(args['check'])
    print(args['lossless_jpeg'])

    if args['variant'] in OK_VARIANTA or args['check'] or args['lossless_jpeg']:
    # if True:
        file = open('files.txt', 'r')
        for line in file.readlines():
            exec(line)
        list_img = read_csv_file()

        if bool(args['check']):
            check_images_object(list_img, create_day_missing=True, create_night_missing=True)

        if bool(args['lossless_jpeg']):
            do_lossless_JPG(list_img)

        if args['variant'] == 'STANDARD':
            create_img_sets(list_img=list_img, variant='Dataset STANDARD', verbose=False)
            create_edge_sets(list_img=list_img, variant='Dataset STANDARD', verbose=False)
            create_label_sets(list_img=list_img, variant='Dataset STANDARD', verbose=False)
        elif args['variant'] == 'BUILDING_DET_3':
            create_img_detection_dataset(list_img=list_img, variant='Dataset 3_2', folder_out='v3_2', verbose=False)
        elif args['variant'] == 'BUILDING_DET_3_NIGHT':
            create_img_detection_dataset(list_img=list_img, variant='Dataset 3_2_NIGHT', folder_out='v3_2_night', verbose=False)
            #TODO fix numbering of objects
        elif args['variant'] == 'BUILDING_DET_3_N':
            create_img_detection_dataset(list_img=list_img, variant='Dataset 3_N', folder_out='v3_n', verbose=False)
        elif args['variant'] == 'SEMSEG_EVAL_FULL':
            create_img_sets_label_full(list_img=list_img, verbose=False)
            create_label_sets_full_eval(list_img=list_img, verbose=False)
