from __future__ import division
import cv2
import numpy as np
import csv
import os
import pandas as pd
import glob
from matplotlib import pyplot as plt
import h5py


# *******************************************************

def max_width_height(path):
    os.chdir(path)
    WIDTH=[]
    HEIGHT=[]

    images_name = glob.glob("*.png")
    set_img = set([x.rsplit('.', 1)[0] for x in images_name])
    for img in set_img:
        img_cv = cv2.imread(path+'/'+img+'.png')
        h=img_cv.shape[0]
        w=img_cv.shape[1]
        WIDTH.append(w)
        HEIGHT.append(h)
    max_width=max(WIDTH)
    max_height=max(HEIGHT)
    return max_height,max_width

# *******************************************

def add_padding(max_height,max_width):

    path_char = '/home/ahmed/internship/cnn_ocr/scale_train/cropped_images'
    path_switched_char = '/home/ahmed/internship/cnn_ocr/scale_train/switched_pixels/'
    output = '/home/ahmed/internship/cnn_ocr/scale_train/dataset/'
    abby_label = []
    reference = []
    os.chdir(path_char)
    img_char= glob.glob("*.png")


    os.chdir(path_switched_char)
    img_switched_char = glob.glob("*.png")

    set_img_char = set([x.rsplit('.', 1)[0] for x in img_char])
    set_img_switched_char = set([x.rsplit('.', 1)[0] for x in img_switched_char])
    images = []
    size= []
    for img in img_char:
        img_cv = cv2.imread(path_char+'/'+img)
        h,w=img_cv.shape[0:2]

        width_diff=max_width-w
        height_diff=max_height-h

        left = width_diff // 2
        right = width_diff - left
        top = height_diff // 2
        bottom = height_diff - top


        white_pixels = [255, 255, 255]
        black_pixels = [0, 0, 0]


        constant = cv2.copyMakeBorder(img_cv,top,bottom,left,right, cv2.BORDER_CONSTANT, value=white_pixels)
        #con=cv2.resize(constant, (max_height,max_width), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output+'_white_'+img,constant)
        constant2 = cv2.copyMakeBorder(img_cv,top,bottom,left,right, cv2.BORDER_CONSTANT, value=black_pixels)
        #con2= cv2.resize(constant2, (max_height, max_width), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output+'_black_'+img, constant2)

        const = constant.reshape((-1, 1))
        const = const.T
        const = const.flatten()
        images.append(const)

        const2 = constant2.reshape((-1, 1))
        const2 = const2.T
        const2= const2.flatten()
        images.append(const2)

        label, sep,rest = img.partition('_')
        abby_label.append(label)
        abby_label.append(label)
        reference.append(rest)
        reference.append(rest)




    for img_switched in img_switched_char:

        img_cv = cv2.imread(path_switched_char + img_switched )
        h,w=img_cv.shape[0:2]
        width_diff=max_width-w
        height_diff=max_height-h

        left = width_diff // 2
        right = width_diff - left
        top = height_diff // 2
        bottom = height_diff - top


        white_pixels = [255, 255, 255]
        black_pixels = [0, 0, 0]
        constant3 = cv2.copyMakeBorder(img_cv,top,bottom,left,right, cv2.BORDER_CONSTANT, value=white_pixels)
        #con3 = cv2.resize(constant3, (max_height, max_width), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output+'_white_'+img_switched, constant3)

        constant4 = cv2.copyMakeBorder(img_cv, top,bottom,left,right, cv2.BORDER_CONSTANT, value=black_pixels)
        #con4 = cv2.resize(constant4, (max_height, max_width), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output+'_black_'+img_switched, constant4)

        const3 = constant3.reshape((-1, 1))
        const3 = const3.T
        const3 = const3.flatten()
        images.append(const3)


        const4 = constant4.reshape((-1, 1))
        const4 = const4.T
        const4 = const4.flatten()
        images.append(const4)

        label, sep,rest = img_switched.partition('_')
        abby_label.append(label)
        abby_label.append(label)
        reference.append(rest)
        reference.append(rest)

    df = pd.DataFrame({'abby_label': abby_label, 'reference': reference})
    df.to_csv('/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv/abby_labels.csv')
    df2=pd.DataFrame({'dimension':size})
    df2.to_csv('/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv/dimension.csv')

    img_arr = np.stack(images, axis=0)


    np.savetxt('/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv/pixels_dataset_large.csv', img_arr,
                delimiter=",")
    h5f=h5py.File('/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv/dataset.h5','w')
    h5f.create_dataset('dataset_1',data=img_arr)
    h5f.close()

    return  img_arr

# *******************************************************#





path_csv="/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv"
path_img="/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/images"

path_save_cropped_images = '/home/ahmed/internship/cnn_ocr/scale_train/cropped_images/'
path_save_resized_images = '/home/ahmed/internship/cnn_ocr/scale_train/resized_images/'
path_save_switched_pixels = '/home/ahmed/internship/cnn_ocr/scale_train/switched_pixels/'
path_dataset = '/home/ahmed/internship/cnn_ocr/scale_train/dataset/'



npa = pd.read_csv("/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv/merged.csv", usecols=[3,5,6, 7, 8, 9], header=None)
nb_charac=npa.shape[0]-1
characs=[]
cpt=0
f = open("/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/csv/merged.csv", 'rt')
reader = csv.reader(f)
for row in reader:
    if cpt >= 1:  # skip header
        characs.append(str(row[5]))
    cpt += 1

path_img = "/home/ahmed/internship/cnn_ocr/scale_train/concatenate/clean_data/images"
os.chdir(path_img)
images_name = glob.glob("*.png")
#set_img = set([x.rsplit('.', 1)[0] for x in images_name])
#mask = npa.iloc[:,0].isin(set_img)

char=[]
for image in images_name:
    c,rest = image.split('.')

    char.append(c)

k=0
for img in char:

    img_cv = cv2.imread(path_img+'/'+ img +'.png')
    images = []
    images_names = []
    WIDTH=[]
    HEIGHT=[]
    for i in range(1, nb_charac):
        #if (img==npa[mask][3][i]):
        if (img==npa.iloc[i,0]):

            coords = npa.iloc[[i]]
            charac = characs[i - 1]
            if (charac == "'/'"):
                charac = "'slash'"
            if (charac == "'.'"):
                charac = "'dot'"
            if (charac == "','"):
                charac = "'comma'"
            charac = charac.strip('"\'')
            page_ref = coords[3].item()
             #j +=i
            img_charac = img_cv[int(coords[8]):int(coords[9]), int(coords[6]):int(coords[7])]

            cv2.imwrite(
            path_save_cropped_images + charac + "_char_" + str(i) + "_" + str(img_charac.shape) + "_" + str(
                page_ref) + "_"
            + str(coords[6].item()) + "_" + str(coords[7].item()) + "_" + str(coords[8].item()) + "_" + str(
                coords[9].item())+ ".png", img_charac)

            img_charac_switched = 255 - img_charac

            cv2.imwrite(
            path_save_switched_pixels + charac+'_' + "switched_char_" + str(i) + "_" + str(img_charac_switched.shape) + "_" + str(
                page_ref) + "_"
            + str(coords[6].item()) + "_" + str(coords[7].item()) + "_" + str(coords[8].item()) + "_" + str(
                coords[9].item()) + ".png",
            img_charac_switched)






h,w=max_width_height(path_save_cropped_images)
print(h,w)
x=add_padding(h,w)









