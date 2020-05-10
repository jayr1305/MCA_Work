import os 
import time
import pickle
import numpy as np

folder  = "train"
directory = "prob"
directory1 = "prob5"

time_list = []

all_precision = []
all_recall = []
all_f1 = []

good_queries = []
ok_queries = []
junk_queries = []

def get_query(queryname):
	with open(directory+"/"+queryname, 'rb') as f:
		q1 = pickle.load(f)
	with open(directory1+"/"+queryname, 'rb') as f:
		q5 = pickle.load(f)

	list_simi = []
	for filename in os.listdir(directory):
		with open(directory+"/"+filename, 'rb') as f:
			plist = pickle.load(f)
		with open(directory1+"/"+filename, 'rb') as f:
			p5list = pickle.load(f)

		temp1_diff = np.absolute(np.subtract(q1, plist))
		temp5_diff = np.absolute(np.subtract(q5, p5list))
		
		temp1_add = np.add(1, np.add(q1, plist))
		temp5_add = np.add(1, np.add(q5, p5list))

		similarity = (np.divide(temp1_diff, temp1_add)).sum() + (np.divide(temp5_diff, temp5_add)).sum()
		list_simi.append([similarity/100,filename[:-7]])
	
	list_simi.sort()

	return list_simi

def get_precision(real,pred):
	true_pos = 0
	for val in pred:
		if(val[1] in real):
			true_pos+=1
	return true_pos*100/len(pred)


def get_recall(real,pred):
	true_pos = 0
	for val in pred:
		if(val[1] in real):
			true_pos+=1
	return true_pos*100/len(real)


def get_f1(Recall,Precision):
	if(Recall+Precision==0):
		return -1
	return 2*(Recall * Precision) / (Recall + Precision)

def stats_print(query_fname,val_type,top_predict,fname):

	open_file = folder+"/ground_truth/"+query_fname[:-9]+val_type+query_fname[-4:]
	
	with open(open_file, 'rb') as f2:
		results = str(f2.read()).split('\\n')[:-1]
		results[0] = results[0][2:]

	gp = get_precision(results,top_predict[:105])
	gr = get_recall(results,top_predict[:105])
	gf = get_f1(gr,gp)

	all_precision.append(gp)
	all_recall.append(gr)
	all_f1.append(gf)

	if(val_type=="good"):
		good_queries.append(gr)

	if(val_type=="ok"):
		ok_queries.append(gr)

	if(val_type=="junk"):
		junk_queries.append(gr)

	# print("\n\tPrecision for " + fname + " "+val_type+" : ",gp)
	# print("\tRecall for " + fname + " "+val_type+" : ",gr)
	# print("\tF1 for " + fname + " "+val_type+" : ",gf)

def main():
	for query_fname in os.listdir(folder+"/query/"):
		st_time = time.time()
		with open(folder+"/query/"+query_fname, 'rb') as f1:
			whole_query = str(f1.read())
			fname = whole_query[7:].split()[0]
			top_predict = get_query(fname+".pickle")
		
		print("\n",fname)
		stats_print(query_fname,"good",top_predict,fname)
		stats_print(query_fname,"ok",top_predict,fname)
		stats_print(query_fname,"junk",top_predict,fname)

		time_list.append(time.time()-st_time)

	print(folder)
	print("\tPrecision")
	print("\t\t Minimum Precision = ",min(all_precision))
	print("\t\t Maximum Precision = ",max(all_precision))
	print("\t\t Average Precision = ",np.mean(all_precision))

	print("\tRecall")
	print("\t\t Minimum Recall = ",min(all_recall))
	print("\t\t Maximum Recall = ",max(all_recall))
	print("\t\t Average Recall = ",np.mean(all_recall))

	print("\tF1")
	print("\t\t Minimum F1 score = ",min(all_f1))
	print("\t\t Maximum F1 score = ",max(all_f1))
	print("\t\t Average F1 score = ",np.mean(all_f1),"\n")

	print("\tAvg time required for retrieval = ",np.mean(time_list),"\n")

	print("Avg percentage of queries for good = ",np.mean(good_queries))
	print("Avg percentage of queries for ok = ",np.mean(ok_queries))
	print("Avg percentage of queries for junk = ",np.mean(junk_queries))


if __name__ == '__main__':
	main()


