#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import regex as re
from scipy import sparse
from functions_TOKENS import get_tokens_from_sent

#------------------------------
def cosine_sim(vector1, vector2):
     
    result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) ) 
    
    return result
    

#------------------------------
def check_vectors_cos_sim(vector1, vector2, thershold = 0.8):
    
    cos = cosine_sim(vector1, vector2)
    
    if cos >= thershold:
        return True
    else:
        False


#------------------------------
def get_close_vecs(input_vec, vec_matrix, first_index = 0, n_close_vecs = 5):
    
    cosine_sim_list = []
    for i in range(len(vec_matrix)):
        coss_val = cosine_sim(input_vec, vec_matrix[i])
        if coss_val != 0:
            #if coss_val < 0:
            #    coss_val = coss_val *  (-1)
            cosine_sim_list.append([i, coss_val])
            #print(token, token_index, coss_val)
            #time.sleep(0.1)
            
    similar_vec_indexes = []
    c_array = np.array(cosine_sim_list)
    #organizando a lista
    for i in c_array[ : , 1].argsort()[ :: -1][  : n_close_vecs ]:
        similar_vec_indexes.append(i + first_index)
        #print(wv.index.values[i])
        
    return similar_vec_indexes 


#------------------------------
def get_largest_topic_value_indexes(input_topic_vec, n_topics = 10, get_only_positive=True):

    #copiando o vector
    topic_vec = input_topic_vec

    #coletando somente os indexes com valores de tópicos positivos
    if get_only_positive is True:
        positive_indexes = np.where(input_topic_vec > 0)[0]

    #determinando os argumentos dos maiores valores dos tópicos em ordem crecente
    indexes = np.argsort(topic_vec)

    #coletando somente os últimos "n_topics" (os quais possuem os maiores valores numéricos) e ordenando os indexes em ordem crescente
    indexes = np.sort(indexes[ -n_topics : ])
    
    if get_only_positive is True:
        filtered_indexes = np.array([index for index in indexes if index in positive_indexes])
    else:
        filtered_indexes = indexes
        
    #retornando os index organizados em ordem crescente
    return filtered_indexes


#------------------------------
def get_item_from_sparse(row_index, column_index, matrix):
    
    # Get row values
    row_start = matrix.indptr[row_index]
    row_end = matrix.indptr[row_index + 1]
    row_values = matrix.data[row_start:row_end]

    # Get column indices of occupied values
    index_start = matrix.indptr[row_index]
    index_end = matrix.indptr[row_index + 1]

    # contains indices of occupied cells at a specific row
    row_indices = list(matrix.indices[index_start:index_end])

    # Find a positional index for a specific column index
    value_index = row_indices.index(column_index)

    if value_index >= 0:
        return row_values[value_index]
    else:
        # non-zero value is not found
        return 0


#------------------------------
def get_neighbor_vecs(input_vec, vec_matrix, first_index = 0, n_close_vecs = 5):
    
    neighbor_vecs_list = []
    for i in range(len(vec_matrix)):
        dist_val = vec_dist(input_vec, vec_matrix[i])
        if dist_val != 0:
            neighbor_vecs_list.append([i, dist_val])

    closest_vec_indexes = []            
    c_array = np.array(neighbor_vecs_list)
    #organizando a lista
    for i in c_array[ : , 1].argsort()[  : n_close_vecs ]:
        closest_vec_indexes.append(i + first_index)
        
    return closest_vec_indexes


#------------------------------
def get_vector_from_string(string, vector_dim = 10, get_versor = False):

    find_list = re.findall(r'[\-0-9\*\s]*[0-9]+', string) if string is not None else []
    #print(find_list)
    topics = []
    vector_mags = []
    
    if len(find_list) > 0:
        for find in find_list:
            #print(find)
            match = re.search(r'([\-\s0-9]+)?[\*\s]*([0-9]+)', find)
            
            try:
                #print('match.group(1)', match.group(1) )
                vector_mags.append( int(''.join( match.group(1).split())) )
            except (AttributeError, ValueError):
                vector_mags.append(1)
            
            try:
                #print('match.group(2)', match.group(2) )
                topics.append( int(match.group(2)) )
            except (AttributeError, ValueError):
                continue
        
            #print(topics)
            #print(vector_mags)
    
        if len(topics) == len(vector_mags):

            #subtrai-se 1 pois o tópico 1 tem index = 0
            topic_index = np.array(topics, dtype = int) - 1
            vector_mags = np.array(vector_mags)

            vec = np.zeros(vector_dim, dtype = float)
            vec[topic_index] = vector_mags

            if get_versor is True:
                vec = get_1dversor(vec)

            return vec

        #caso não tenha sido encontrado o vector
        else:
            return None
    
    #caso não tenha sido encontrado o vector
    else:
        return None


#------------------------------
def get_tv_from_sent_index(sent_index, tv_stats = None, doc_topic_matrix = None, scaling = False, normalize = False):

    '''
    t0 = time.time()
    for article_filename in sent_indexes.index:
        initial_sent = sent_indexes.loc[ article_filename, 'initial_sent' ]
        last_sent = sent_indexes.loc[ article_filename, 'last_sent' ]
        if last_sent >= sent_index >= initial_sent:
            topic_vec = doc_topic_matrix[sent_index]
        else:
            continue    
    print('t1 = ', time.time()  - t0)
    '''
    topic_vec = doc_topic_matrix[sent_index]

    #caso o scaling do tópico esteja ligado
    if tv_stats is not None and scaling is True:
        scaled_topic_vec = scaling_vector( topic_vec, tv_stats, scaler_type = 'min_max' )
    else:
        scaled_topic_vec = topic_vec

    #caso a normalização esteja ligada
    if normalize is True:
        norm_scale_topic_vec = scale_normalize_1dvector(scaled_topic_vec)        
    else:
        norm_scale_topic_vec = scaled_topic_vec

    #print('\n( TV.shape :', topic_vec.shape)    
    return norm_scale_topic_vec


#------------------------------
def get_wv_from_sentence(sentence_str, 
                         wv_DF, 
                         wv_stats, 
                         stopwords_list = None, 
                         test=False,
                         spacy_tokenizer = None):
    
    #splitando a sentença em tokens
    #before = time.time()
    sent_tokens = get_tokens_from_sent(sentence_str, stopwords_list_to_remove = stopwords_list, spacy_tokenizer = spacy_tokenizer)
    #print('tokenizer time spent: ', time.time() - before)
    #print('\nTOKENS SENT: ', sent_tokens)
    
    #caso o filtro de stopwords seja usado
    if stopwords_list:
        sent_tokens = [ token for token in sent_tokens if token not in stopwords_list]
    #print('FILTERED TOKENS SENT: ', sent_tokens, '\n')
    
    #before = time.time()
    #gerando data para o treino
    for token in sent_tokens:
        
        try:
            scaled_wv = scaling_vector( wv_DF.loc[token].values, wv_stats, scaler_type = 'min_max' )
            sent_token_wvs = np.vstack((sent_token_wvs, scaled_wv))
            
            if test is True:
                print(sent_token_wvs)
                time.sleep(0.5)
        
        #caso seja o primeiro token da sentença
        except NameError:
            sent_token_wvs = scaled_wv
        
        except KeyError:
            continue
    
    #print('collecting wvs time spent: ', time.time() - before)
    #print('\n( sent_tokens_wv.shape: ', sent_token_wvs.shape , ' )')
    try:        
        return sent_token_wvs
    #caso nenhum token tenha sido encontrado na matriz de WV
    except NameError:
        return None


#------------------------------
def load_dense_matrix(path):

    return np.load(path + '.npy')


#------------------------------
def load_sparse_csr_matrix(path):

    from scipy import sparse
    return sparse.load_npz(path + '.npz').todense()


#------------------------------
def scale_normalize_1dvector(vector: np.array) -> np.array:
    
    #print('Normalizing vector... ')
    if ( vector.max() - vector.min() ) != 0:
        scaled_wv = ( vector - vector.min() ) / ( vector.max() - vector.min() )
        return ( scaled_wv / np.linalg.norm(scaled_wv, ord = 1) )
    else:
        return None


#------------------------------
def get_1dversor(vector: np.array) -> np.array:

    return vector / np.linalg.norm(vector, ord = 2)


#------------------------------
def save_dense_matrix(path, matrix):
    
    np.save(path + '.npy', matrix)


#------------------------------
def save_sparse_csr_matrix(path, matrix):

    sparse.save_npz(path + '.npz', sparse.csr_matrix(matrix, dtype = matrix.dtype))


#------------------------------
def scaling_vector(vector, matrix_vector_stats, scaler_type = 'min_max'):
    
    vector_min = matrix_vector_stats['min']
    vector_max = matrix_vector_stats['max']
    
    if scaler_type == 'min_max':
        vector_norm = (vector - vector_min) / (vector_max - vector_min)
    
    #print( vector_norm.min() , vector_norm.max() )
    return vector_norm


#------------------------------
def vec_dist(vector1, vector2):
    
    return np.linalg.norm(vector2 - vector1)