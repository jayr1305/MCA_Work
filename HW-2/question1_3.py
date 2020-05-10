import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

# cur = "spc"
cnst = 2
cur = "mfcc"

def load_pick(directory):
	flag=0
	total_list = np.array([])

	for filename in os.listdir(directory):
		with open(directory+"/"+filename, 'rb') as f:
			plist = np.array(pickle.load(f))
			print("\t",plist[-1,1])
			print(len(plist))

			if(flag==0):
				total_list = plist
				flag = 1
			else:
				total_list = np.concatenate((total_list, plist), axis=0)

	np.random.shuffle(total_list)
	return total_list


def save_model(clf,score):
	model_name = cur+"_model_"+str(score)[2:]+".sav"
	pt = open(model_name,"wb")
	pickle.dump(clf,pt)
	pt.close()
	print("Saved "+cur+" svm model.")


def main():
	
	print("Loading pickles")
	directory = "./"+cur+"/"+cur+"_pickle_train"
	# directory = "./"+cur+"/"+cur+"_noise_train"
	total_list = load_pick(directory)

	
	endi = int(len(total_list)/cnst)
	print("Length ",len(total_list[:endi,1]))

	# print(total_list[-1])
	
	print("Started Training\n")
	X = normalize(list(total_list[:endi,0]))
	Y = list(total_list[:endi,1])
	clf = SVC(gamma="auto", kernel = "linear")
	clf.fit(X,Y)

	np.random.shuffle(total_list)

	print("Started Testing")
	x_test = normalize(list(total_list[:endi,0]))
	y_test = list(total_list[:endi,1])
	score = clf.score(x_test,y_test)
	print("\tScore : ",score)
	

	save_model(clf,score)


if __name__ == '__main__':
	main()