import os
import time
import pickle
import numpy as np
from PIL import Image 


start_time = time.time()

quan = 100
img = ""
rl = 0
cl = 0

color_prob = dict()
color_prob[1] = np.zeros(shape=(quan,2))
# color_prob[5] = np.zeros(shape=(quan,2))


def calc(i,j,d):
	clr = img[i][j]

	lef = max(0,j-d)
	rig = min(cl-1,j+d)

	up = max(0,i-d)
	dow = min(rl-1,i+d) 

	num = 0
	deno = 0
	
	if(i-d == up):
		temp = (img[up,lef:rig+1]==clr) 
		num += temp.sum()
		deno += len(temp)

	if(i+d == dow):
		temp = (img[dow,lef:rig+1]==clr) 
		num += temp.sum()
		deno += len(temp)

	if(j-d == lef):
		temp = (img[dow:up+1,lef]==clr) 
		num += temp.sum()
		deno += len(temp)

	if(j+d == rig):
		temp = (img[dow:up+1,rig]==clr) 
		num += temp.sum()
		deno += len(temp)
	
	if(num>deno):
		print(i,j,d,"error")

	color_prob[d][clr][0] += num
	color_prob[d][clr][1] += deno
	return 

def make_mtrx():
	print("\t",img.shape)
	
	for i in range(rl):
		for j in range(cl):
			calc(i,j,1)
			# calc(i,j,5)
			if(i%500==0 and j%500 == 0):
				print("\t",i,j) 

def set_var(fname):
	global img,rl,cl

	img = Image.open(fname).convert('L')
	img = img.quantize(quan)
	img = np.array(img)

	rl = img.shape[0]
	cl = img.shape[1]

def save_prob(fname):
	set_var("images/"+fname)
	make_mtrx()
	disti = 1

	##########
	# disti = 5
	# save_name = "prob5/"+fname[:-4]+".pickle"
	##########

	dist_l = color_prob[disti][:,0] / color_prob[disti][:,1]
	save_name = "prob/"+fname[:-4]+".pickle"
	
	print(save_name)

	pickle_prob = open(save_name,"wb")
	pickle.dump(dist_l,pickle_prob)
	pickle_prob.close()

	print("\t",time.time() - start_time)

def main(strt,end):
	directory = "images"
	c = 1
	for filename in os.listdir(directory):
		if(c>=strt and c<=end):
			print(filename,c)
			save_prob(filename)
		c+=1

if __name__ == '__main__':
	main(1,5063)
