#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#------------------------------
def find_LSA_sent_similarity(self, sent_index = 0, n_similar_sents = 5, replace_Ngrams = False, diretorio = None):

    from FUNCTIONS import get_topic_vec_from_sent_index    
    from FUNCTIONS import get_sent_from_index
    from FUNCTIONS import get_close_vecs
    import pandas as pd
    import h5py
    
    #carregando a lista com o index de sentenças
    sent_indexes = pd.read_csv(diretorio + '/Outputs/LOG/SENT_INDEX.csv', index_col = 0)

    #carregando o topic_vector para a sentença inserida
    topic_vec = get_topic_vec_from_sent_index(sent_index, replace_Ngrams = replace_Ngrams)
    
    for filename in sent_indexes.index:
        
        #carregando a matriz DOC_TOPIC
        h5 = h5py.File(diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{replace_Ngrams}.h5', 'r')
        DOC_TOPIC_matrix = h5['data']
        
        index_first_sent = sent_indexes.loc[filename, 'initial_sent']
        index_last_sent = sent_indexes.loc[filename, 'last_sent']
        
        #calculando a semelhança pela similaridade de cosseno
        indexes_closest_vecs = get_close_vecs(topic_vec, 
                                              DOC_TOPIC_matrix[ index_first_sent : index_last_sent ], 
                                              first_index = index_first_sent , 
                                              n_close_vecs = n_similar_sents)
        h5.close()
        
        print('Filename:', filename)
        counter = 1
        for index in indexes_closest_vecs:
            closest_sent = get_sent_from_index(index)
            print(f'Closest Sent ({counter}): ', closest_sent[1])
            counter += 1
            
    del h5
    del DOC_TOPIC_matrix
    del indexes_closest_vecs


#------------------------------
def find_min_max_token_sent_len(sent_batch_size = 4, machine_type = 'conv1d', sent_tokens_stats = {}, filter_stopwords = False):
  
    if machine_type in ('conv1d', 'logreg'):
        #para conv1d, o tamanho máximo permitido de tokens para as sentenças concatenadas será a média somada do desvio padrão (1sigma) vezes o sent_batch
        sent_max_token_len = int( (sent_tokens_stats[f'SW_{filter_stopwords}']['avg'] + (sent_tokens_stats[f'SW_{filter_stopwords}']['std']) ) * sent_batch_size )
        #para conv1d, o tamanho mínimo permitido de tokens para as sentenças concatenadas será a média subtraida do desvio padrão (1sigma) vezes o sent_batch
        sent_min_token_len = int( (sent_tokens_stats[f'SW_{filter_stopwords}']['avg'] - (sent_tokens_stats[f'SW_{filter_stopwords}']['std']) ) * sent_batch_size )
        #o token_len mínimo para uma senteça é o 0.05 quantile
        single_sent_min_token_len = int(sent_tokens_stats[f'SW_{filter_stopwords}']['0.05_quantile'])
    elif machine_type == 'conv2d':
        #para conv2d, o tamanho máximo permitido de tokens para as sentenças concatenadas será o quantile de 0.95
        sent_max_token_len = int(sent_tokens_stats[f'SW_{filter_stopwords}']['0.95_quantile'])
        #para conv2d, o tamanho mínimo permitido de tokens para as sentenças concatenadas será o quantile de 0.05
        sent_min_token_len = int(sent_tokens_stats[f'SW_{filter_stopwords}']['0.05_quantile'])
        #o token_len mínimo para uma senteça é o 0.05 quantile
        single_sent_min_token_len = int(sent_tokens_stats[f'SW_{filter_stopwords}']['0.05_quantile'])
        
    return sent_max_token_len , sent_min_token_len , single_sent_min_token_len


#------------------------------
def from_token_indexed_csv_to_DICT(filepath):
    
    import pandas as pd
    
    try:
        dic = pd.read_csv(filepath, index_col=0).to_dict(orient='index')
        
    except ValueError:
        df = pd.read_csv(filepath)
        df.dropna(inplace=True)
        df.set_index('index', inplace=True)
        dic = df.to_dict(orient='index')
        
    return dic


#------------------------------
def get_nGrams_list(term, ngrams = 0, min_ngram_appearence = 5, diretorio = None):
    
    import regex as re
    import pandas as pd
    
    term_list = []
    
    if ngrams == 1:

        try:
            prob_term_list = []
            n1gram_DF = pd.read_csv(diretorio + f'/Outputs/nGrams/Semantic/n1gram_{term}.csv', index_col = 0)
            n1gram_DF = n1gram_DF[ n1gram_DF['Sem_App_Counter'] >= min_ngram_appearence ]
            n1gram_DF.dropna(inplace = True)

            #filtrando os plurais (ex: nutshells -> nutshell)
            n1gram_list_to_exclude1 = [n1gram+'s' for n1gram in n1gram_DF.index.to_list() if (n1gram+'s' in n1gram_DF.index.to_list())] 
            n1gram_list_to_exclude2 = [n1gram+'es' for n1gram in n1gram_DF.index.to_list() if (n1gram+'es' in n1gram_DF.index.to_list())]
            n1gram_list_to_exclude = n1gram_list_to_exclude1 + n1gram_list_to_exclude2
            term_list = [n1gram for n1gram in n1gram_DF.index.to_list() if n1gram not in n1gram_list_to_exclude]

            #cálculo da probabilidade de aparecer o n1gram
            max_counts = n1gram_DF['Sem_App_Counter'].values[0]
            for n1gram in term_list:
                prob_term_list.append( n1gram_DF.loc[n1gram, 'Sem_App_Counter'] / max_counts)
            print(f'Term_list montada com n1grams (termo: {term})')
        except FileNotFoundError:
            print('Erro! Nenhum arquivo n1gram com termos semânticos encontrado para esse termo.')
            return
        
    elif ngrams == 2:

        try:
            prob_term_list = []
            n2gram_DF = pd.read_csv(diretorio + f'/Outputs/nGrams/Semantic/n2gram_{term}.csv', index_col = 0)
            
            #filtrando os plurais (ex: cocoa_nutshells -> cocoa_nutshell)
            n2gram_list_to_exclude1 = [n2gram+'s' for n2gram in n2gram_DF.index.to_list() if (n2gram+'s' in n2gram_DF.index.to_list())]
            n2gram_list_to_exclude2 = [n2gram+'es' for n2gram in n2gram_DF.index.to_list() if (n2gram+'es' in n2gram_DF.index.to_list())]
            n2gram_list_to_exclude = n2gram_list_to_exclude1 + n2gram_list_to_exclude2
            term_list = [n2gram for n2gram in n2gram_DF.index.to_list() if n2gram not in n2gram_list_to_exclude]
            
            #cálculo da probabilidade de aparecer o n2gram (é 1 sempre)
            n_n2grams = len(term_list)
            prob_term_list = n_n2grams * [1]
            print(f'Term_list montada com n2grams (termo: {term})')                
        except FileNotFoundError:
            print(f'Erro! Nenhum arquivo n2gram com termos semânticos encontrado para o termo: {term}')
            return
        
    elif ngrams == 0:

        try:
            prob_term_list_n1gram = []
            n1gram_DF = pd.read_csv(diretorio + f'/Outputs/nGrams/Semantic/n1gram_{term}.csv', index_col = 0)
            n1gram_DF = n1gram_DF[ n1gram_DF['Sem_App_Counter'] >= min_ngram_appearence ]
            n1gram_DF.dropna(inplace = True)
            
            #filtrando os plurais (ex: nutshells -> nutshell)
            n1gram_list_to_exclude1 = [n1gram+'s' for n1gram in n1gram_DF.index.to_list() if (n1gram+'s' in n1gram_DF.index.to_list())] 
            n1gram_list_to_exclude2 = [n1gram+'es' for n1gram in n1gram_DF.index.to_list() if (n1gram+'es' in n1gram_DF.index.to_list())]
            n1gram_list_to_exclude = n1gram_list_to_exclude1 + n1gram_list_to_exclude2
            n1gram_list = [n1gram for n1gram in n1gram_DF.index.to_list() if n1gram not in n1gram_list_to_exclude]
            
            #cálculo da probabilidade de aparecer o n1gram
            max_counts = n1gram_DF['Sem_App_Counter'].values[0]
            for n1gram in n1gram_list:
                prob_term_list_n1gram.append( n1gram_DF.loc[n1gram, 'Sem_App_Counter'] / max_counts)
        except FileNotFoundError:
            print(f'Erro! Nenhum arquivo n1gram com termos semânticos encontrado para o termo: {term}')
            return

        try:
            prob_term_list_n2gram = []
            n2gram_DF = pd.read_csv(diretorio + f'/Outputs/nGrams/Semantic/n2gram_{term}.csv', index_col = 0)                
            #filtrando os plurais (ex: cocoa_nutshells -> cocoa_nutshell)
            #filtrando os plurais (ex: cocoa_nutshells -> cocoa_nutshell)
            n2gram_list_to_exclude1 = [n2gram+'s' for n2gram in n2gram_DF.index.to_list() if (n2gram+'s' in n2gram_DF.index.to_list())]
            n2gram_list_to_exclude2 = [n2gram+'es' for n2gram in n2gram_DF.index.to_list() if (n2gram+'es' in n2gram_DF.index.to_list())]
            n2gram_list_to_exclude = n2gram_list_to_exclude1 + n2gram_list_to_exclude2            
            n2gram_list = [n2gram for n2gram in n2gram_DF.index.to_list() if n2gram not in n2gram_list_to_exclude]

            #cálculo da probabilidade de aparecer o n2gram (= 1)
            n_n2grams = len(n2gram_list)
            prob_term_list_n2gram = n_n2grams * [1]                

            #filtrando os n1grams (só entram os que não estão no n2grams)
            n1gram_DF_list_filtered = [n1gram for n1gram in n1gram_list if (n1gram not in n2gram_DF['token_0'].values) and (n1gram not in n2gram_DF['token_1'].values)]
            
            #concatenando os n1grams + n2grams
            term_list = n2gram_list + n1gram_DF_list_filtered
            
            #a lista de probabilidade de n1gram é mudada pois parte dos termos foram filtrados
            prob_term_list_n1gram = len(n1gram_DF_list_filtered) * [1]
            
            #concatenando as probabilidades de n1grams + n2grams
            prob_term_list = prob_term_list_n2gram + prob_term_list_n1gram
            print(f'Term_list montada com n1grams + n2grams (termo: {term})')
        
        except FileNotFoundError:
            term_list = n1gram_list
            prob_term_list = prob_term_list_n1gram
            print(f'Term_list montada com n1grams (termo: {term})')
    
    term_list_modified = [ re.sub('_', ' ' , token) for token in term_list ]
    
    #print(sorted(term_list_modified))
    #print(len(term_list_modified), len(prob_term_list))
    return term_list_modified, prob_term_list


#------------------------------
def get_tokens_from_sent(sent, tokenizer_func = None, tokens_list = None, stopwords_list = None, n2grams_DF = None, filter_stopwords = False, replace_Ngrams = False):
    
    import time
    from functions_TEXTS import filter_tokens
    from functions_TOKENS import replace_Ngrams_in_tokens
    
    #splitando a sentença em tokens
    if tokenizer_func in (None, 'python'):
        sent_tokens = [ token.lower() for token in filter_tokens(sent.split()) ]
    else:
        sent_tokens = [ token.lower() for token in filter_tokens(list(tokenizer_func(sent))) ]
    
    #print('TOKENS SENT: ', sent_tokens)
    
    #substituindo os Ngrams
    if (replace_Ngrams is True):
        sent_tokens = replace_Ngrams_in_tokens(sent_tokens, n2grams_DF)
    
    #filtrando os tokens considerando o que está na lista
    if tokens_list is None:
        sent_tokens_filtered = sent_tokens
    else:
        sent_tokens_filtered = [ token for token in sent_tokens if token in tokens_list]
    
    #caso o filtro de stopwords seja usado
    if (filter_stopwords is True):
        sent_tokens_filtered = [ token for token in sent_tokens_filtered if token not in stopwords_list]
    #print('FILTERED TOKENS SENT: ', sent_tokens_filtered)
    #time.sleep(0.1)

    return sent_tokens_filtered

#------------------------------
def get_tokens_len(sentence_str, collection_token_list = ['a', 'b', 'c'], filter_stopwords = False, n2grams_DF = None, n3grams_DF = None):
    
    from functions_TOKENS import replace_Ngrams_in_tokens
    from functions_TEXTS import filter_tokens
    import spacy
    nlp = spacy.load('en_core_web_sm')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')
    
    #print(sentence_str)
    
    #splitando a sentença em tokens
    sent_tokens_raw = [ token.lower() for token in filter_tokens(list(nlp(sentence_str))) ]
    sent_tokens_filtered = [ token for token in sent_tokens_raw if token in collection_token_list]
    
    #substituindo os Ngrams
    if (n2grams_DF is not None) or (n3grams_DF is not None):
        sent_tokens_filtered = replace_Ngrams_in_tokens(sent_tokens_filtered, n2grams_DF = n2grams_DF, n3grams_DF = n3grams_DF)
    
    #caso o filtro de stopwords seja usado
    if (filter_stopwords is True):
        sent_tokens_filtered = [ token for token in sent_tokens_filtered if token not in stopwords_list]

    #print('TOKENS SENT: ', sent_tokens_filtered)    
    print('N_TOKENS: ', len(sent_tokens_filtered))
    return len(sent_tokens_filtered)


#------------------------------
def replace_Ngrams_in_tokens(sent_tokens, n2grams_DF = None, n3grams_DF = None, check = False):

    
    if (n3grams_DF is not None):
        replaced_token = 0
        modified_sent_tokens_1 = []
        for index in range(len(sent_tokens)):
            try:
                token_0 = sent_tokens[index]
                token_1 = sent_tokens[index + 1]  
                token_2 = sent_tokens[index + 2]
                n3gram = token_0 + '_' + token_1 + '_' + token_2
                cond1 = n3gram in n3grams_DF.index
                if (cond1 is True):
                    modified_sent_tokens_1.append(n3gram)
                    replaced_token = 2
                    
                elif (replaced_token > 0):
                    replaced_token -= 1

                else:
                    modified_sent_tokens_1.append(token_0)
                    
            except IndexError:
                if (replaced_token is True):
                    continue
                else:
                    modified_sent_tokens_1.append(token_0)

    else:
        modified_sent_tokens_1 = sent_tokens


    if (n2grams_DF is not None):
        replaced_token = False
        modified_sent_tokens_2 = []
        for index in range(len(modified_sent_tokens_1)):
            try:
                token_0 = modified_sent_tokens_1[index]
                token_1 = modified_sent_tokens_1[index + 1]            
                n2gram = token_0 + '_' + token_1
                cond1 = n2gram in n2grams_DF.index
                if (replaced_token is True):
                    replaced_token = False
                    continue
                
                elif (cond1 is True):
                    modified_sent_tokens_2.append(n2gram)
                    replaced_token = True
                
                else:
                    modified_sent_tokens_2.append(token_0)
                    
            except IndexError:
                if (replaced_token is True):
                    continue
                else:
                    modified_sent_tokens_2.append(token_0)
                    
    else:
        modified_sent_tokens_2 = modified_sent_tokens_1
    
    if check is True:
        if len(sent_tokens) > len(modified_sent_tokens_2):
            print(sent_tokens)
            print(modified_sent_tokens_2)
        
    return modified_sent_tokens_2


#------------------------------
def save_tokens_to_csv(tokens_list, filename, diretorio = None):
    
    import os
    import pandas as pd
    
    DF = pd.DataFrame([],columns=['Token'], dtype=object)
    counter = 0
    for token in tokens_list:
        DF.loc[counter] = token
        counter += 1
    #caso não haja o diretorio ~/Outputs/
    if not os.path.exists(diretorio + '/Outputs'):
        os.makedirs(diretorio + '/Outputs')
    DF.to_csv(diretorio + '/Outputs/' + filename)
    print('Salvando os tokens extraidos em ~/Outputs/' + filename)
