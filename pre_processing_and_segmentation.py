import numpy as np
#from numba import jit, cuda 
import os
import pandas as pd
from skimage.segmentation import quickshift,felzenszwalb,slic
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.segmentation import mark_boundaries
import cv2
from skimage.measure import regionprops
import math
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow
from PIL import Image
import multiprocessing as mp
import time


df_merge_col_lat=pd.read_csv(r'/home/prashant540490_gmail_com/merged_processed_image_lat.csv')


#art_supress

def largest_mask(img_bin):
    
    n_labels, img_labeled, statistics, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)
    largest_area_label = np.argmax(statistics[1:, 4]) + 1
    
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)         #get blackbackground image
    
    largest_mask[img_labeled == largest_area_label] = 255          #get largest mask from labeled image and assign pixel value 255
   
    #fill holes
    start_locations = np.where(img_labeled == 0)
    seed = (start_locations[0][0], start_locations[1][0])
    print(seed)
    img_floodfill = largest_mask.copy()
    h_, w_ = largest_mask.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
    cv2.floodFill(img_floodfill, mask_, seedPoint=seed, newVal=255) 
    holes_mask = cv2.bitwise_not(img_floodfill)                      # get mask of holes
    largest_mask = largest_mask + holes_mask                         #get largest mask after filling holes

    
    kernal = np.ones((15, 15), dtype=np.uint8)
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernal)          #Apply morphological opening operation and useful inremoving noise
        
    return largest_mask


def art_surpress(img) :
    img = cv2.equalizeHist(img)
    th1 = 18  
    _, im_bin = cv2.threshold(img, th1, maxval=255, type=cv2.THRESH_BINARY)
    

    axes[1].imshow(bin, cmap='gray')
    breast_mask = largest_mask(im_bin) 
    arti_suppr = cv2.bitwise_and(img, mammo_breast_mask)
    
    return arti_suppr,breast_mask

#pectoral muscle removal
def largest_mask_pect(img_bin):
    
    n_labels, img_labeled, statistics, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)
    leftmost_area_label = np.argmax(statistics[1:, 0]) + 1
    
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)         #get blackbackground image
    
    largest_mask[img_labeled == leftmost_area_label] = 255          #get largest mask from labeled image and assign pixel value 255
   
    #fill holes
    start_locations = np.where(img_labeled == 0)
    seed = (start_locations[0][0], start_locations[1][0])
    print(seed)
    img_floodfill = largest_mask.copy()
    height, widtth = largest_mask.shape
    mask_ = np.zeros((height + 2, widtth + 2), dtype=np.uint8)
    cv2.floodFill(img_floodfill, mask_, seedPoint=seed, newVal=255) 
    holes_mask = cv2.bitwise_not(img_floodfill)                      # get mask of holes
    largest_mask = largest_mask + holes_mask                         #get largest mask after filling holes

    
    kernel_ = np.ones((15, 15), dtype=np.uint8)
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)          #Apply morphological opening operation and useful inremoving noise
        
    return largest_mask

@jit
def pectorialMuscleRemoval(mammo_arti_suppr,mammo_breast_mask,Laterality):
    mammo_breast_equ = cv2.equalizeHist(mammo_arti_suppr)

    th2 = 210  
    _, img_pect = cv2.threshold(mammo_breast_equ, th2,maxval=255, type=cv2.THRESH_BINARY)    #get thresholded image using high value of thresholding
    
    img_pect_mark = np.zeros(img_pect.shape, dtype=np.int32)
    
    # Sure foreground
    pect_mask_init = largest_mask_pect(img_pect)
    kernel_ = np.ones((3, 3), dtype=np.uint8)  
    n_erosions = 10  
    pect_mask_eroded = cv2.erode(pect_mask_init, kernel_, iterations=n_erosions)              #Erosion operation
    img_pect_mark[pect_mask_eroded > 0] = 255

    
    n_dilations = 10 

    pect_mask_dilated = cv2.dilate(pect_mask_init, kernel_, iterations=n_dilations)          #dilationoperation
    img_pect_mark[pect_mask_dilated == 0] = 128
    
    img_pect_mark[mammo_breast_mask == 0] = 64

    
    img_breast = cv2.bitwise_and(mammo_breast_equ, img_pect_mark)
  
    

    return img_breast,pect_mask_init

# code to crop and extract largest part

def crop(mammo_breast_mask,mammo_arti_suppr,Laterality) :
    if (Laterality=='R'):
        mammo_breast_mask_rotate_flipHorizontal = cv2.flip(mammo_breast_mask, 1)
        mammo_arti_suppr_flip = cv2.flip(mammo_arti_suppr, 1)
    else:
        mammo_breast_mask_rotate_flipHorizontal =mammo_breast_mask
        mammo_arti_suppr_flip = mammo_arti_suppr
    max_X=0
    max_y=0

    max_below_x=0
    min_above_x=mammo_breast_mask_rotate_flipHorizontal.shape[0]
    positions = np.nonzero(mammo_breast_mask_rotate_flipHorizontal)

    min_above_x = positions[0].min()
    max_below_x = positions[0].max()
    #left = positions[1].min()
    max_y = positions[1].max()
    
    mammo_breast_mask_rotate_flipHorizontal_croped=mammo_arti_suppr_flip[min_above_x:max_below_x,0:max_y]

    return mammo_breast_mask_rotate_flipHorizontal_croped

#process code

import cv2
import os
folder = '/mnt/disk3/disser/raw_images/images'
#@jit(nopython=True, parallel=True)
def load_images_from_folder1(folder):
    label_fail=[]
    count=0
    count_n=0
    #count_m=0
    for filename1 in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder,filename1),0)
            #img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
            #label.append(filename)
            #filename=filename[-5:-4]
            filename=filename1[:-6]
    
            a=np.where(df_merge_col_lat['ImgID']==filename)
            #if(len(a[0])!=0):
            view_position=df_merge_col_lat.iloc[a[0][0],12]
            #print(view_position)
            Laterality=df_merge_col_lat.iloc[a[0][0],8]
            left=df_merge_col_lat.iloc[a[0][0],9]
            right=df_merge_col_lat.iloc[a[0][0],10]
            level=df_merge_col_lat.iloc[a[0][0],11]
            #print(Laterality)
            
            print(count)
            print(count_n)
            #artifact supression
            img_breast_art_supress,mammo_breast_mask=art_surpress(img)
            #If view is ML0 Remove artifacts
            if(view_position=='MLO'):
                    img_breast=pectorialMuscleRemoval(img_breast_art_supress,mammo_breast_mask,Laterality)
            #cropping operation
            mammo_breast_mask_rotate_flipHorizontal_croped=crop(mammo_breast_mask,img_breast,Laterality)
            W=mammo_breast_mask_rotate_flipHorizontal_croped.shape[0]
            L=mammo_breast_mask_rotate_flipHorizontal_croped.shape[1]
            a1=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[0:math.floor(W/4),0:math.floor(L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a2=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[0:math.floor(W/4),math.floor(L/4):math.floor(L/2)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a3=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[0:math.floor(W/4),math.floor(L/2):math.floor(3*L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a4=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[0:math.floor(W/4),math.floor(3*(L/4)):L], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a5=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/4):math.floor(W/2),0:math.floor(L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a6=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/4):math.floor(W/2),math.floor(L/4):math.floor(L/2)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a7=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/4):math.floor(W/2),math.floor(L/2):math.floor(3*L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a8=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/4):math.floor(W/2),math.floor(3*L/4):L], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                
            a9=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/2):math.floor(3*W/4),0:math.floor(L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a10=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/2):math.floor(3*W/4),math.floor(L/4):math.floor(L/2)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a11=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/2):math.floor(3*W/4),math.floor(L/2):math.floor(3*L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a12=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(W/2):math.floor(3*W/4),math.floor(3*L/4):L], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            
            a13=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(3*W/4):W,0:math.floor(L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a14=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(3*W/4):W,math.floor(L/4):math.floor(L/2)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a15=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(3*W/4):W,math.floor(L/2):math.floor(3*L/4)], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            a16=cv2.resize(mammo_breast_mask_rotate_flipHorizontal_croped[math.floor(3*W/4):W,math.floor(3*L/4):L], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                
            
                
        
            ima1 = Image.fromarray(a1)
            if((Laterality+"B1")  in str(left) or (Laterality+"B1")  in str(right)):
                ima1.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B1"+"_"+level+".png")
            elif (count_n < 10000):                                 #get only 10000 imgaes with label =N(Normal)
                ima1.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B1"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima2 = Image.fromarray(a2)
            if((Laterality+"C1")  in str(left) or (Laterality+"C1")  in str(right)):
                ima2.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C1"+"_"+level+".png")
            elif (count_n < 10000):
                ima2.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C1"+"_"+"N"+".png")
                count_n=count_n+1
            
            
            ima3 = Image.fromarray(a3)
            if((Laterality+"D1")  in str(left) or (Laterality+"D1")  in str(right)):
                ima3.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D1"+"_"+level+".png")
            elif (count_n < 10000):
                ima3.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D1"+"_"+"N"+".png")
                count_n=count_n+1
            
            
            ima4 = Image.fromarray(a4)
            if((Laterality+"E1")  in str(left) or (Laterality+"E1")  in str(right)):
                ima4.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E1"+"_"+level+".png")
            elif (count_n < 10000):
                ima4.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E1"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima5 = Image.fromarray(a5)
            if((Laterality+"B2")  in str(left) or (Laterality+"B2")  in str(right)):
                ima5.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B2"+"_"+level+".png")
            elif (count_n < 10000):
                ima5.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B2"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima6 = Image.fromarray(a6)
            if((Laterality+"C2")  in str(left) or (Laterality+"C2")  in str(right)):
                ima6.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C2"+"_"+level+".png")
            elif (count_n < 10000):
                ima6.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C2"+"_"+"N"+".png")
                count_n=count_n+1
            
            ima7 = Image.fromarray(a7)
            if((Laterality+"D2")  in str(left) or (Laterality+"D2")  in str(right)):
                ima7.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D2"+"_"+level+".png")
            elif (count_n < 10000):
                ima7.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D2"+"_"+"N"+".png")
                count_n=count_n+1
            
            ima8 = Image.fromarray(a8)
            if((Laterality+"E2")  in str(left) or (Laterality+"E2")  in str(right)):
                ima8.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E2"+"_"+level+".png")
            elif (count_n < 10000):
                ima8.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E2"+"_"+"N"+".png")
                count_n=count_n+1
            
            ima9 = Image.fromarray(a9)
            if((Laterality+"B3")  in str(left) or (Laterality+"B3")  in str(right)):
                ima9.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B3"+"_"+level+".png")
            elif (count_n < 10000):
                ima9.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B3"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima10 = Image.fromarray(a10)
            if((Laterality+"C3")  in str(left) or (Laterality+"C3")  in str(right)):
                ima10.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C3"+"_"+level+".png")
            elif (count_n < 10000):
                ima10.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C3"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima11 = Image.fromarray(a11)
            if((Laterality+"D3")  in str(left) or (Laterality+"D3")  in str(right)):
                ima11.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D3"+"_"+level+".png")
            elif (count_n < 10000):
                ima11.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D3"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima12 = Image.fromarray(a12)
            if((Laterality+"E3")  in str(left) or (Laterality+"E3")  in str(right)):
                ima12.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E3"+"_"+level+".png")
            elif (count_n < 10000):
                ima12.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E3"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima13 = Image.fromarray(a13)
            if((Laterality+"B4")  in str(left) or (Laterality+"B4")  in str(right)):
                ima13.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B4"+"_"+level+".png")
            elif (count_n < 10000):
                ima13.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"B4"+"_"+"N"+".png")
                count_n=count_n+1
                
            ima14 = Image.fromarray(a14)
            if((Laterality+"C4")  in str(left) or (Laterality+"C4")  in str(right)):
                ima14.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C4"+"_"+level+".png")
            elif (count_n < 10000):
                ima14.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"C4"+"_"+"N"+".png")
                count_n=count_n+1
            ima15 = Image.fromarray(a15)
            if((Laterality+"D4")  in str(left) or (Laterality+"D4")  in str(right)):
                ima15.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D4"+"_"+level+".png")
            elif (count_n < 10000):
                ima15.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"D4"+"_"+"N"+".png")
                count_n=count_n+1
            ima16 = Image.fromarray(a16)
            if((Laterality+"E4")  in str(left) or (Laterality+"E4")  in str(right)):
                ima16.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E4"+"_"+level+".png")
            elif (count_n < 10000):
                ima16.save("/mnt/disk2/segmented_cc_sample/"+filename+"_"+Laterality+"E4"+"_"+"N"+".png")
                count_n=count_n+1
            
            count=count+1
        except :
            print("**********************fail",filename)
            #label_fail.append(filename)
            pass



load_images_from_folder1(folder)

#df_images_label_fail.to_csv(r'df_images_label_fail_without_fail1.csv', index = None, header=True)
