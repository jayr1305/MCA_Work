import os
import pickle
import numpy as np
from sklearn.svm import SVC
from question1_3 import load_pick
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report,confusion_matrix

# directory = "./spc/spc_pickle_val"
# mod_file = "./model/model_spc.sav"

cnst = 1

directory = "./mfcc/mfcc_noise_val"
mod_file = "./model/model_mfcc_noisy.sav"


print("Loading pickles")
total_list =  load_pick(directory)


print("Loading "+mod_file)
clf = SVC()
with open(mod_file, 'rb') as f:
	clf = pickle.load(f)

endi = int(len(total_list)/cnst)

print("Length ",endi)

X = normalize(list(total_list[:endi,0]))
Y = list(total_list[:endi,1])


print("Getting report!")
y_pred = clf.predict(X)
y_true = Y

print(np.unique(y_pred))
print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))