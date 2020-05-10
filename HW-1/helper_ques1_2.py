import os
import json
import cv2
import matplotlib.pyplot as plt

directory = "images"
img_show2 = True

for filename in os.listdir(directory):
	with open("blob_json"+"/"+filename[:-4]+".json", 'rb') as f:
		blob_list = json.loads(f.read())['blobs']
		# print(len(blob_list))
	img = cv2.imread('images/'+filename,0)/255
    	
	if img_show2 :
		print("\t",filename,len(blob_list),"\n")

		_t, ax = plt.subplots()
		ax.imshow(img)
		for blobs in blob_list:
			x,y,r = blobs[0],blobs[1],blobs[2]
			ax.add_patch(plt.Circle((x, y), r*1.414, color='red', linewidth=0.7, fill=False))
		ax.plot()  
		plt.show()  