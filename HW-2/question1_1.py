#Jay Rawal
#2017240
#MCA Assignment 2

#SPECTOGRAM

#Reference 
#https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
#https://towardsdatascience.com/fast-fourier-transform-937926e591cb 


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wf

# folder = "./Dataset/training"
folder = "./Dataset/validation"

cnst = 20
NFFT = 256
overlap = 0.1

c=0
d = {}
check = False
result_spec = []

noises = []
add_noise = False
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
		global result_spec
		result_spec = []

		folder_path = folder + "/" + sf
		folder_work(folder_path,sf)
		
		print(result_spec[-1])

		pt = open("specto_"+sf+".pickle","wb")
		pickle.dump(result_spec,pt)
		pt.close()
		print("Saved "+ sf +" spectrogram results")


def folder_work(folder_path,fnum):
	global c,d,result_spec
	noise_cnt = 0

	for file_name in os.listdir(folder_path):

		file_path = folder_path + "/" + file_name

		c+=1
		if(c%250==0): 
			print(file_path,c)
		
		fs,wave = wf.read(file_path)

		extra = 16000-len(wave)
		if(extra>0):
			wave = np.append(wave,np.zeros(extra))

		
		#FOR NOISE**************************************
		if(add_noise):
			noise_cnt = (noise_cnt+1)%len(noises)
			wave = wave + noises[noise_cnt]*noise_factor
		#***********************************************

		#Reducing Size
		wave = wave[3000:13000]
		fs = fs-6000
		
		#Conversion to DB
		result = spectrogram(wave,fs)
		result = np.log10(result)
		result = np.clip(cnst*result,-40,200)
		
		spec = result[:,::2].flatten()
		# spec = result.flatten()

		result_spec.append([spec,d[fnum]])
		
		if(check):
			#My spectogram
			img = plt.imshow(result, origin='lower', interpolation='nearest', aspect='auto')
			plt.xlabel('Time')
			plt.ylabel('Frequency')
			plt.show()

			print(spec.shape)
			print(fs, wave, wave.size/fs, min(wave), max(wave))
			
			plt.plot([i for i in range(fs)],wave)
			plt.show()
			break


def pre_process(seg):
	wndw = np.hanning(NFFT)
	double_pad = np.zeros(NFFT)
	seg_cos = seg*wndw
	return np.append(seg_cos,double_pad)


def dft(x):
	#Implemented from reference!
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def spectrogram(wave, fs):
	ovp = 1- overlap

	hop_size = np.int32(np.floor(NFFT*ovp))
	total_segments = np.int32(np.ceil(len(wave)/hop_size))

	to_process = np.concatenate((wave,np.zeros(NFFT)))
	result = np.empty((NFFT,total_segments), dtype = np.float32)

	for i in range(total_segments):
		strt = i*hop_size
		end = strt + NFFT

		#pre process
		seg_pad = pre_process(to_process[strt:end])

		#appliying fourier
		temp_res = np.fft.fft(seg_pad) / NFFT

		#post process
		conj_res = np.conj(temp_res)
		res = np.abs(temp_res*conj_res) 

		#Appending
		result[:,i] = res[:NFFT]

	if(check):
		#In-built
		plt.specgram(wave,Fs = fs)
		plt.xlabel('Time')
		plt.ylabel('Frequency')
		plt.show()

	return result


if __name__ == '__main__':
	main()