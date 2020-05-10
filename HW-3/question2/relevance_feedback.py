import numpy as np
from evaluation import read_gt
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

alpha = 0.75
beta = 0.1


def rel_and_nonrel(vec_docs, sorted_result, all_truth, n):

	topn = sorted_result[:n]

	add_val = csr_matrix(vec_docs[0,:].shape)
	sub_val = csr_matrix(vec_docs[0,:].shape)


	# #Relevance based on ground truth as user suggestion
	# #If we take topn and evaluate on basis of ground truth relevant and non-relevant
	# for i,r in enumerate(topn):
	# 	if all_truth[r] == 1:
	# 		add_val = add_val + vec_docs[r,:]
	# 	else:
	# 		sub_val = sub_val + vec_docs[r,:]


	#Psuedo relevance feedback
	#If topn result are relevant ones and bottomn are non relevant
	for i,r in enumerate(topn):
	    add_val = add_val + vec_docs[r,:]
	botn = sorted_result[sorted_result.shape[0]-n:]
	for i,r in enumerate(botn):
	    sub_val = add_val + vec_docs[r,:]
	
	
	return add_val,sub_val


def update_query_vec(vec_docs, vec_queries, sim, gt, n):

	for q in range(vec_queries.shape[0]):
		add_val, sub_val = rel_and_nonrel(vec_docs,np.argsort(-sim[:, q]), gt[:, q], n)
		vec_queries[q,:] = vec_queries[q,:] + (alpha*add_val) - (beta*sub_val)

	return vec_queries


def relevance_feedback(vec_docs, vec_queries, sim, gt, n=10):
	"""
	relevance feedback
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		n: integer
			number of documents to assume relevant/non relevant
		gt: ground truth file, only if to be assumed as user feedback

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""
	y_true = read_gt(gt, sim.shape)
	i=0
	while(i<3):
		i+=1
		vec_queries = update_query_vec(vec_docs, vec_queries, sim, y_true, n)
		sim = cosine_similarity(vec_docs, vec_queries)

	rf_sim = sim
	return rf_sim


def get_topn_terms(tfidf_model, vec_docs, sorted_result,n):
	
	topn = sorted_result[:n]
	all_term = {}

	for i,r in enumerate(topn):
		response = vec_docs[r,:]
		sorted_nzs = np.argsort(response.data)[:-(n+1):-1]
		
		idx = response.indices[sorted_nzs]
		val = response.data[sorted_nzs]
		# print(idx,val)
		
		for j,term in enumerate(idx):
			if(term in all_term.keys()):
				all_term[term] = max(val[j],all_term[term])
			else:
				all_term[term] = val[j]


	term_list = sorted(all_term.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:min(n,len(all_term))]
	terms = np.zeros((1,vec_docs.shape[1]))

	for idx,val in term_list:
		terms[0][int(idx)] = val

	terms = csr_matrix(terms)
	return terms


def extend_query(vec_docs, vec_queries, sim, n, tfidf_model):
	feature_names = tfidf_model.get_feature_names()

	for q in range(vec_queries.shape[0]):
		terms = get_topn_terms(tfidf_model,vec_docs,np.argsort(-sim[:, q]),n)
		vec_queries[q,:] = vec_queries[q,:] + terms*0.5

	return vec_queries


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, gt, n=10):
	"""
	relevance feedback with expanded queries
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		tfidf_model: TfidfVectorizer,
			tf_idf pretrained model
		n: integer
			number of documents to assume relevant/non relevant
		gt: ground truth file, only if to be assumed as user feedback

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""

	y_true = read_gt(gt, sim.shape)
	i=0
	while(i<3):
		i+=1
		vec_queries = update_query_vec(vec_docs, vec_queries, sim, y_true, n)
		sim = cosine_similarity(vec_docs, vec_queries)

		vec_queries = extend_query(vec_docs, vec_queries, sim, n, tfidf_model)
		sim = cosine_similarity(vec_docs, vec_queries)
	
	rf_sim = sim
	return rf_sim