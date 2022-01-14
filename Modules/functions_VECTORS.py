#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#------------------------------
def cosine_sim(vector1, vector2):
    
    import numpy as np
     
    result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) ) 
    
    return result


#------------------------------
def get_close_vecs(input_vec, vec_matrix, first_index = 0, n_close_vecs = 5):

    import numpy as np
    
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
    #organizando o
    for i in c_array[ : , 1].argsort()[ :: -1][  : n_close_vecs ]:
        similar_vec_indexes.append(i + first_index)
        #print(wv.index.values[i])
        
    return similar_vec_indexes 


#------------------------------
def get_largest_topic_value_indexes(input_topic_vec, n_topics = 10, get_only_positive=True):
    
    import numpy as np
    #import time

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
def get_tv_from_sent_index(sent_index, tv_stats = None, DOC_TOPIC_matrix = None, scaling = False, normalize = False, diretorio=None):
    
    import time
    import pandas as pd
    
    #carregando os indexes de sentenças
    send_indexes = pd.read_csv(diretorio + '/Outputs/LOG/SENT_INDEX.csv', index_col = 0)
    
    for pdf_filename in send_indexes.index:
        initial_sent = send_indexes.loc[pdf_filename, 'initial_sent']
        last_sent = send_indexes.loc[pdf_filename, 'last_sent']
        if last_sent >= sent_index >= initial_sent:
            topic_vec = DOC_TOPIC_matrix[sent_index]
        else:
            continue    
    
    #caso o scaling do tópico esteja ligado
    if tv_stats is not None and scaling is True:
        scaled_topic_vec = scaling_vector( topic_vec, tv_stats, scaler_type = 'min_max' )
    else:
        scaled_topic_vec = topic_vec
    
    #caso a normalização esteja ligada
    if normalize is True:
        norm_scale_topic_vec = normalize_1dvector(scaled_topic_vec)        
    else:
        norm_scale_topic_vec = scaled_topic_vec
    
    #print('( TV.shape :', topic_vec.shape)    
    return norm_scale_topic_vec


#------------------------------
def get_wv_from_sentence(sentence_str, 
                         wv_DF, 
                         wv_stats, 
                         collection_token_list = ['a', 'b', 'c'], 
                         filter_stopwords = False, 
                         n2grams_DF = None, 
                         n3grams_DF = None, 
                         test=False):
    
    import time
    import numpy as np
    from functions_TEXTS import filter_tokens
    from functions_TOKENS import replace_Ngrams_in_tokens
    import spacy
    nlp = spacy.load('en_core_web_sm')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')
    
    #print(sentence_str)
    
    sent_token_wvs = []
    #splitando a sentença em tokens
    sent_tokens = [ token.lower() for token in filter_tokens(list(nlp(sentence_str))) ]
    #print('\nTOKENS SENT: ', sent_tokens)
    sent_tokens_filtered = [ token for token in sent_tokens if token in collection_token_list]
    
    #substituindo os Ngrams
    if (n2grams_DF is not None) or (n3grams_DF is not None):
        sent_tokens_filtered = replace_Ngrams_in_tokens(sent_tokens_filtered, n2grams_DF = n2grams_DF, n3grams_DF = n3grams_DF)
    
    #caso o filtro de stopwords seja usado
    if (filter_stopwords is True):
        sent_tokens_filtered = [ token for token in sent_tokens_filtered if token not in stopwords_list]
    #print('FILTERED TOKENS SENT: ', sent_tokens_filtered, '\n')
    
    #gerando data para o treino
    for token in sent_tokens_filtered:
        
        scaled_wv = scaling_vector( wv_DF.loc[token].values, wv_stats, scaler_type = 'min_max' )
        
        if test is True:
            sent_token_wvs.append(scaled_wv[-5: ])
            print(sent_token_wvs)
            time.sleep(0.5)
        else:
            sent_token_wvs.append(scaled_wv)

    sent_token_wvs = np.array(sent_token_wvs)
    #print('( sent_tokens_wv.shape: ', sent_token_wvs.shape , ' )')
    
    return sent_token_wvs


#------------------------------
def normalize_1dvector(vector):
    
    import numpy as np
    
    #print('Normalizing vector... ')
    if ( vector.max() - vector.min() ) != 0:
        scaled_wv = ( vector - vector.min() ) / ( vector.max() - vector.min() )
        return ( scaled_wv / np.linalg.norm(scaled_wv, ord = 1) )
    else:
        return None


#------------------------------
def scaling_vector(vector, matrix_vector_stats, scaler_type = 'min_max'):
    
    vector_min = matrix_vector_stats['min']
    vector_max = matrix_vector_stats['max']
    
    if scaler_type == 'min_max':
        vector_norm = (vector - vector_min) / (vector_max - vector_min)
    
    #print( vector_norm.min() , vector_norm.max() )
    return vector_norm