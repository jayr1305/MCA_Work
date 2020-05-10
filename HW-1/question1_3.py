import os
import cv2 
import time
import json 
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix_det
from scipy.ndimage import gaussian_laplace as LoG

#Ref Links
#(1) https://stackoverflow.com/questions/51411156/is-laplacian-of-gaussian-for-blob-detection-or-for-edge-detection

#constants
img_show1 = False
img_show2 = False
thres = 0.003
ksze = 3
blob_list = []
base = 2**(1/2)
img = ""

def savelist(fname):
    save_name = "blob_json_surf/"+fname[:-4]+".json"
    save_dict = {}

    save_dict['blobs'] = []

    for blobs in blob_list:
        x,y,r = blobs[1],blobs[0],blobs[2]
        save_dict['blobs'].append([x.item(),y.item(),r.item()])

    with open(save_name, 'w') as outfile:
        json.dump(save_dict, outfile)
    
    if img_show2 :
        print("\t",fname,len(blob_list),"\n")
        _t, ax = plt.subplots()
        ax.imshow(img)
        for blobs in blob_list:
            x,y,r = blobs[1],blobs[0],blobs[2]
            ax.add_patch(plt.Circle((x, y), r*1.414, color='red', linewidth=0.7, fill=False))
        ax.plot()  
        plt.show()  


def blobWork(arr,a,b):
    pt = np.argmax(arr)
    val = np.max(arr)
    if(val<=thres):
        return
    else:
        co_ord = np.unravel_index(pt,arr.shape)
        blob_list.append((co_ord[1]+a-1,co_ord[2]+b-1,(base**co_ord[0])))

def save_blob(fname):
    global blob_list,img

    blob_list = []
    img = cv2.imread('images/'+fname,0)/255
    
    if img_show1==True:
        cv2.imshow("d",img)
        cv2.waitKey(3000)

    val_log = []
    for sig  in range(1,5):
        #(1) Ref Link
        img_g = cv2.GaussianBlur(img, (ksze,ksze), base**sig)
        LoG = cv2.Laplacian(img_g, cv2.CV_64F, ksize=ksze)
        val_log.append(hessian_matrix_det(LoG, sigma=base**sig))

    val_log = np.array(val_log)

    rl = val_log.shape[1]
    cl = val_log.shape[2]

    for i in range(0,rl,5):
        for j in range(0,cl,5):
            val_arr = np.absolute(val_log[:, max(i-1,0):min(i+2,rl-1) , max(j-1,0):min(j+2,cl-1)])
            blobWork(val_arr,i,j)
            # if(i%500==0 and j%500==0):
            #     print(i,j,val_arr.shape)

    savelist(fname)
            

def main(strt,end):
    directory = "images"
    c = 1
    start_time = time.time()
    for filename in os.listdir(directory):
        if(c>=strt and c<=end):
            print(c)
            save_blob(filename)
        c+=1
    
    print("\t",time.time() - start_time)

if __name__ == '__main__':
    main(1,5063)