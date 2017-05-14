
import cv2
import numpy as np
import csv
import os
import pandas as pd
import glob
from matplotlib import pyplot as plt
from __future__ import division


def max_width_height(path):
    os.chdir(path)
    WIDTH=[]
    HEIGHT=[]

    images_name = glob.glob("*.png")
    set_img = set([x.rsplit('.', 1)[0] for x in images_name])
    for img in set_img:
        w,h=img.shape[0:2]
        WIDTH.append(w)
        HEIGHT.append(h)
    max_width=max(WIDTH)
    max_height=max(HEIGHT)
    return max_width, max_height


def add_padding(max_width,max_height):

    path_char = '/home/ahmed/internship/cnn_ocr/scale_train/cropped_images/'
    path_switched_char = '/home/ahmed/internship/cnn_ocr/scale_train/switched_pixels/'
    output = '/home/ahmed/internship/cnn_ocr/scale_train/dataset/'

    os.chdir(path_char)
    img_char= glob.glob("*.png")

    os.chdir(path_switched_char)
    img_switched_char = glob.glob("*.png")

    set_img_char = set([x.rsplit('.', 1)[0] for x in img_char])
    set_img_switched_char = set([x.rsplit('.', 1)[0] for x in img_switched_char])
    images = []
    Lab=[]
    reference=[]
    for img in img_char:

        w,h=img.shape[0:2]
        width_diff=max_width-w
        height_diff=max_height-h

        left= width_diff/2
        right=width_diff/2
        top=height_diff/2
        bottom=height_diff/2
        if isinstance(left,float):
            left=int(left)
            right=left+1
        if isinstance(top,float):
            top=int(top)
            bottom=top+1

        white_pixels = [255, 255, 255]
        black_pixels = [0, 0, 0]
        img1 = cv2.imread(path_char+img+'.png')

        constant = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white_pixels)

        cv2.imwrite(output+img+'_whitepadding'+'.png',constant)
        constant2 = cv2.copyMakeBorder(img1,top, bottom, left, right, cv2.BORDER_CONSTANT, value=black_pixels)
        cv2.imwrite(output + img + '_blackpadding' + '.png', constant2)
        images.append(constant)
        images.append(constant2)
        label,sep,rest = img.partition('_')
        Lab.append(label)
        reference.apapend(rest)

    for img_switched in img_switched_char:

        w,h=img_switched_char.shape[0:2]
        width_diff=max_width-w
        height_diff=max_height-h

        left= width_diff/2
        right=width_diff/2
        top=height_diff/2
        bottom=height_diff/2
        if isinstance(left,float):
            left=int(left)
            right=left+1
        if isinstance(top,float):
            top=int(top)
            bottom=top+1


        white_pixels = [255, 255, 255]
        black_pixels = [0, 0, 0]
        img1 = cv2.imread(path_switched_char+img_switched+'.png')
        constant3 = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white_pixels)
        cv2.imwrite(output + img_switched + '_whitepadding' + '.png', constant3)
        constant4 = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black_pixels)
        cv2.imwrite(output + img_switched + '_blackpadding' + '.png', constant4)
        images.append(constant3)
        images.append(constant4)
        label,sep,rest = img_switched.partition('_')
        Lab.append(label)
        reference.append(rest)

    img_arr = np.stack(images, axis=0)


    np.savetxt('/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/pixels_dataset_large.csv', img_arr,
                   delimiter=",")



    x="perfect"
    return x

# *******************************************************#

path_csv="/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv"
path_img="/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/images"

path_save_cropped_images = '/home/ahmed/internship/cnn_ocr/scale_train/cropped_images/'
path_save_resized_images = '/home/ahmed/internship/cnn_ocr/scale_train/resized_images/'
path_save_switched_pixels = '/home/ahmed/internship/cnn_ocr/scale_train/switched_pixels/'
path_dataset = '/home/ahmed/internship/cnn_ocr/scale_train/dataset/'

os.chdir(path_img)
images_name = glob.glob("*.png")


os.chdir(path_csv)
csvs_name = glob.glob("*.csv")


set_csv= set([x.rsplit('.', 1)[0] for x in csvs_name])


for name in set_csv:
    npa = pd.read_csv(path_csv+"/"+name+".csv", usecols=[3,5,6, 7, 8, 9], header=None)
    nb_charac=npa.shape[0]-1
    characs=[]
    cpt=0
    f = open(path_csv+"/"+name+".csv", 'rt')
    reader = csv.reader(f)
    for row in reader:
        if cpt >= 1:  # skip header
            characs.append(str(row[5]))
        cpt += 1

    img = cv2.imread(os.path.join(path_img, name+'.png'))
    images = []
    images_names = []
    WIDTH=[]
    HEIGHT=[]

    for i in range(1, nb_charac):

        # get coordinates
        # coords=npa[i,:]
        coords = npa.iloc[[i]]
        charac = characs[i - 1]

        if (charac == "'/'"):
            charac = "'slash'"
        if (charac == "'.'"):
            charac = "'dot'"
        if (charac == "','"):
            charac = "'comma'"

        charac = charac.strip('"\'')

        # page_ref=str(list(coords[3])).replace('[','').replace(']','')
        page_ref = coords[3].item()
        j +=i+1
        img_charac = img[int(coords[8]):int(coords[9]), int(coords[6]):int(coords[7])]

        cv2.imwrite(
            path_save_cropped_images + charac + "_char_" + str(i) + "_" + str(img_charac.shape) + "_" + str(
                page_ref) + "_"
            + str(coords[6].item()) + "_" + str(coords[7].item()) + "_" + str(coords[8].item()) + "_" + str(
                coords[9].item()) + "_in_csv_" + str(j) + ".png", img_charac)


        img_charac_switched = 255 - img_charac
        cv2.imwrite(
            path_save_switched_pixels + charac + "switched_char_" + str(i) + "_" + str(img_charac_switched.shape) + "_" + str(
                page_ref) + "_"
            + str(coords[6].item()) + "_" + str(coords[7].item()) + "_" + str(coords[8].item()) + "_" + str(
                coords[9].item()) + "_in_csv_" + str(j) + ".png",
            img_charac_switched)

         ''''
        img1 = img_charac.reshape((-1, 1))
        img1 = img1.T
        img1 = img1.flatten()
        img1_label = charac

        img2 = img_charac_switched.reshape((-1, 1))
        img2 = img2.T
        img2 = img2.flatten()
        img2_label = charac

        images.append(img1)
        images.append(img2)
        # dataset
        img1_name = str(charac) + "_" + str(i) + "_" + str(img_charac.shape + "_" + str(page_ref)) + ".png"
        img2_name = str(charac) + "_switched" + "_" + str(i) + "_" + str(img_charac_switched.shape) + str(
            page_ref) + ".png"
        images_names.append(img1_name)
        images_names.append(img2_name)
        '''
        ''''
        cv2.imwrite(
            path_dataset + charac + "switched_char_" + str(i) + "_" + str(img_charac.shape) + "_" + str(page_ref) + "_"
            + str(coords[6].item()) + "_" + str(coords[7].item()) + "_" + str(coords[8].item()) + "_" + str(
                coords[9].item()) + "_in_csv_" + str(j) + ".png",
            img_charac_switched)
        cv2.imwrite(
            path_dataset + charac + + "_char_" + str(i) + "_" + str(img_charac.shape) + "_" + str(page_ref) + "_"
            + str(coords[6].item()) + "_" + str(coords[7].item()) + "_" + str(coords[8].item()) + "_" + str(
                coords[9].item()) + "_in_csv_" + str(j) + ".png",
            img_charac)
        '''
        '''
        w_img_charac,h_img_charac = img_charac.shape[0:2]
        HEIGHT.append(h_img_charac)
        WIDTH.append(w_img_charac)
        '''

        # cv2.imwrite(path_dataset+ str(charac) + "_switche
        # d" + "_" + str(i) + "_" + str(img_charac_switched.shape) +"_"+str(page_ref)+ ".png",img_charac_switched)
        # cv2.imwrite(path_dataset + str(charac) + "_" + str(i) + "_" + str(img_charac_resized.shape) +"_"+str(page_ref)+ ".png",  img_charac_resized)


        # images = [img1,img2]
        # img_arr = np.stack([img_arr,img1,img2], axis=0)
        # img_arr = np.stack(img2, axis=0)
        # data.append(img1)
        # data.append(img2)

        # print (img_arr.shape)



    print(i)

    w,h=max_width_height(path_save_cropped_images)
    x=add_padding(w,h)








