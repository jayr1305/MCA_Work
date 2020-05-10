#Jay Rawal
#2017240
#MCA Assignment 2

#MFCC

#Reference
#https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial

import os
import pickle
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile as wf


# folder = "./Dataset/training"
folder = "./Dataset/validation"

nfft = 2048
hp_sz = 15
sr = 1

c=0
d = {}
check = False
result_mfcc = []

noises = []
add_noise = True
noise_factor = 0.1

def load_noises():
	global noises

	noisy_fold = "./Dataset/_background_noise_"
	for noise_f in os.listdir(noisy_fold):
		file_path = noisy_fold + "/" + noise_f
		fs,wave = wf.read(file_path)
		
		extra = 16000-len(wave)
		if(extra>0):
			wave = np.append(wave,np.zeros(extra))
		else:
			wave = wave[:16000]

		noises.append(wave)
	noises = np.array(noises)
	print("Number of noises",len(noises))


def main():
	global d

	print("NOISES? ",add_noise)

	d['zero'] = 0
	d['one'] = 1
	d['two'] = 2
	d['three'] = 3
	d['four'] = 4
	d['five'] = 5
	d['six'] = 6
	d['seven'] = 7
	d['eight'] = 8
	d['nine'] = 9
	
	if add_noise:
		load_noises()

	for sf in os.listdir(folder):
		global result_mfcc
		result_mfcc = []

		folder_path = folder + "/" + sf
		folder_work(folder_path,sf)
		
		print(result_mfcc[-1])

		pt = open("mfcc_"+sf+".pickle","wb")
		pickle.dump(result_mfcc,pt)
		pt.close()
		print("Saved "+ sf +" mfcc results")


def folder_work(folder_path,fnum):
	global c,d,result_mfcc
	noise_cnt = 0

	for file_name in os.listdir(folder_path):

		file_path = folder_path + "/" + file_name

		c+=1
		if(c%100==0): 
			print(file_path,c)

	
		# wave, fs = librosa.load(file_path)
		# extra = 22050-len(wave)
		# if(extra>0):
		# 	wave = np.append(wave,np.zeros(extra))
		# else:
		# 	wave = wave[:22050]

		#FOR NOISE**************************************
		if(add_noise):
			noise_cnt = (noise_cnt+1)%len(noises) 
			# wave = wave + noises[noise_cnt]*noise_factor
		#***********************************************
		
		# mfcc_c = mfcc(wave,fs)
		
		mfcc_c = mfccd(file_path,noise_cnt)
		
		result_mfcc.append([mfcc_c.flatten(),d[fnum]])
		
		if(check):
			plt.figure(figsize=(10, 4))
			librosa.display.specshow(mfcc_cd, sr=fs, x_axis='time')
			plt.colorbar()
			plt.show()
			print(fs, wave, wave.size/fs, min(wave), max(wave))
			break


def ftom(f):
	return 2595.0*np.log10(1.0+f/700.0)


def mtof(m):
	return 700.0*(10.0**(m/2595.0)-1.0)


def dct():
	base = np.empty((40,10))
	base[0,:] = 1/np.sqrt(10)
	samples = np.arange(1,20,2)*np.pi/20
	for i in range(1,40):
		base[i,:]=np.cos(i*samples)*np.sqrt(1/5)

	return base


def mfccd(file_path,noise_cnt):
	#implemented mfcc from reference!

	#Getting input
	sz = int(nfft/2)
	sr,wave = wf.read(file_path)

	extra = 16000-len(wave)
	if(extra>0):
		wave = np.append(wave,np.zeros(extra))
	else:
		wave = wave[:16000]

	#FOR NOISE**************************************
	if(add_noise):
		wave = wave + noises[noise_cnt]*noise_factor
	#***********************************************

	wave = wave/np.max(np.abs(wave))
	wave = np.pad(wave,sz,mode="reflect")

	#Create Fft processed Frames
	frame_sz = int(np.round(sr*hp_sz/1000))
	frame_num = int((len(wave) - nfft)/frame_sz)+1
	frames = np.zeros((frame_num,nfft))
	for x in range(frame_num):
		frames[x] = (wave[x*frame_sz : x*frame_sz + nfft])*np.hanning(nfft)
	frames_T = frames.T
	frames_fft = np.empty((int(1 + nfft // 2), frames_T.shape[1]), dtype=np.complex64, order='F')
	for x in range(frame_num):
		frames_fft[:,x] = np.fft.fft(frames_T[:,x],axis = 0)[:int(1 + nfft // 2)]
	frames_fft = np.square(np.abs(frames_fft.T))

	#Create filters to multiply
	frq_mn = ftom(0)
	frq_mx = ftom(sr/2)
	m_frqs = mtof(np.linspace(frq_mn,frq_mx,num=12))
	fl_pts = np.floor((nfft+1)/sr*m_frqs).astype(int)
	filters = np.zeros((len(fl_pts)-2,int(nfft/2+1)))
	for cx in range(len(fl_pts)-2):
		filters[cx, fl_pts[cx] : fl_pts[cx + 1]] = np.linspace(0, 1, fl_pts[cx + 1] - fl_pts[cx])
		filters[cx, fl_pts[cx + 1] : fl_pts[cx + 2]] = np.linspace(1, 0, fl_pts[cx + 2] - fl_pts[cx + 1])
	filters = filters* ((2.0/(m_frqs[2:12]-m_frqs[:10]))[:,np.newaxis])

	#Get coef by DCT
	coef = np.dot(dct(),10*np.log10(np.dot(filters,frames_fft.T) + 0.0000000001))
	return coef

	#implemented mfcc from reference!


def mfcc(wave,fs):
	#To check implementation
	lbs = librosa.feature.mfcc(wave, sr=fs)
	return lbs


if __name__ == '__main__':
	main()