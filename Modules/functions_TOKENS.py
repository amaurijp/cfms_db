#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#------------------------------
def find_min_max_token_sent_len(sent_batch_size = 4, machine_type = 'conv1d', sent_tokens_stats = {}, with_SW = False):
  
    if machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'lstm'):
        #para conv1d, o tamanho máximo permitido de tokens para as sentenças concatenadas será a média somada do desvio padrão (1sigma) vezes o sent_batch
        sent_max_token_len = int( (sent_tokens_stats[f'with_SW_{with_SW}']['0.95_quantile'] ) * sent_batch_size )
        #para conv1d, o tamanho mínimo permitido de tokens para as sentenças concatenadas será a média subtraida do desvio padrão (1sigma) vezes o sent_batch
        sent_min_token_len = int( (sent_tokens_stats[f'with_SW_{with_SW}']['0.05_quantile'] ) * sent_batch_size )
        #o token_len mínimo para uma senteça é o 0.05 quantile
        single_sent_min_token_len = int(sent_tokens_stats[f'with_SW_{with_SW}']['0.05_quantile'])
    elif machine_type in ('conv2d', 'conv2d_conv1d', 'conv2d_lstm'):
        #para conv2d, o tamanho máximo permitido de tokens para as sentenças concatenadas será o quantile de 0.95
        sent_max_token_len = int(sent_tokens_stats[f'with_SW_{with_SW}']['0.95_quantile'])
        #para conv2d, o tamanho mínimo permitido de tokens para as sentenças concatenadas será o quantile de 0.05
        sent_min_token_len = int(sent_tokens_stats[f'with_SW_{with_SW}']['0.05_quantile'])
        #o token_len mínimo para uma senteça é o 0.05 quantile
        single_sent_min_token_len = int(sent_tokens_stats[f'with_SW_{with_SW}']['0.05_quantile'])
        
    return sent_max_token_len , sent_min_token_len , single_sent_min_token_len


#------------------------------
def from_token_indexed_csv_to_dict(filepath):
    
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
def get_nGrams_list(terms_list: list, ngrams: int = None, min_ngram_appearence: int = 5, diretorio: str = None):

    import regex as re
    import pandas as pd
    
    term_list = []

    for term in terms_list:
        
        if str(ngrams) == '1':

            try:
                n1gram_DF = pd.read_csv(diretorio + f'/Outputs/ngrams/semantic/n1gram_{term}.csv', index_col = 0)
                n1gram_DF = n1gram_DF[ n1gram_DF['Sem_App_Counter'] >= min_ngram_appearence ]
                n1gram_DF.dropna(inplace = True)

                #filtrando os plurais (ex: nutshells -> nutshell)
                n1gram_list_to_exclude1 = [n1gram+'s' for n1gram in n1gram_DF.index.tolist() if (n1gram+'s' in n1gram_DF.index.tolist())] 
                n1gram_list_to_exclude2 = [n1gram+'es' for n1gram in n1gram_DF.index.tolist() if (n1gram+'es' in n1gram_DF.index.tolist())]
                n1gram_list_to_exclude = n1gram_list_to_exclude1 + n1gram_list_to_exclude2
                term_list.extend( [n1gram for n1gram in n1gram_DF.index.tolist() if n1gram not in n1gram_list_to_exclude] )
                print(f'Term_list montada com n1grams (termo: {term})')
            
            except FileNotFoundError:
                print(f'Erro! Nenhum arquivo n1gram com termos semânticos encontrado para o termo: {term}')
                return
            
        elif str(ngrams) == '2':

            try:
                n2gram_DF = pd.read_csv(diretorio + f'/Outputs/ngrams/semantic/n2gram_{term}.csv', index_col = 0)
                
                #filtrando os plurais (ex: cocoa_nutshells -> cocoa_nutshell)
                n2gram_list_to_exclude1 = [n2gram+'s' for n2gram in n2gram_DF.index.tolist() if (n2gram+'s' in n2gram_DF.index.tolist())]
                n2gram_list_to_exclude2 = [n2gram+'es' for n2gram in n2gram_DF.index.tolist() if (n2gram+'es' in n2gram_DF.index.tolist())]
                n2gram_list_to_exclude = n2gram_list_to_exclude1 + n2gram_list_to_exclude2
                term_list.extend( [n2gram for n2gram in n2gram_DF.index.tolist() if n2gram not in n2gram_list_to_exclude] )
                print(f'Term_list montada com n2grams (termo: {term})')                
            except FileNotFoundError:
                print(f'Erro! Nenhum arquivo n2gram com termos semânticos encontrado para o termo: {term}')
                return
            
        elif str(ngrams) == '0':

            try:
                n1gram_DF = pd.read_csv(diretorio + f'/Outputs/ngrams/semantic/n1gram_{term}.csv', index_col = 0)
                n1gram_DF = n1gram_DF[ n1gram_DF['Sem_App_Counter'] >= min_ngram_appearence ]
                n1gram_DF.dropna(inplace = True)
                
                #filtrando os plurais (ex: nutshells -> nutshell)
                n1gram_list_to_exclude1 = [n1gram +'s' for n1gram in n1gram_DF.index.tolist() if (n1gram+'s' in n1gram_DF.index.tolist())] 
                n1gram_list_to_exclude2 = [n1gram +'es' for n1gram in n1gram_DF.index.tolist() if (n1gram+'es' in n1gram_DF.index.tolist())]
                n1gram_list_to_exclude = n1gram_list_to_exclude1 + n1gram_list_to_exclude2
                n1gram_list = [n1gram for n1gram in n1gram_DF.index.tolist() if n1gram not in n1gram_list_to_exclude]

            except FileNotFoundError:
                print(f'Erro! Nenhum arquivo n1gram com termos semânticos encontrado para o termo: {term}')
                return

            try:
                n2gram_DF = pd.read_csv(diretorio + f'/Outputs/ngrams/semantic/n2gram_{term}.csv', index_col = 0)
                #filtrando os plurais (ex: cocoa_nutshells -> cocoa_nutshell)
                #filtrando os plurais (ex: cocoa_nutshells -> cocoa_nutshell)
                n2gram_list_to_exclude1 = [n2gram +'s' for n2gram in n2gram_DF.index.tolist() if (n2gram+'s' in n2gram_DF.index.tolist())]
                n2gram_list_to_exclude2 = [n2gram +'es' for n2gram in n2gram_DF.index.tolist() if (n2gram+'es' in n2gram_DF.index.tolist())]
                n2gram_list_to_exclude = n2gram_list_to_exclude1 + n2gram_list_to_exclude2            
                n2gram_list = [n2gram for n2gram in n2gram_DF.index.tolist() if n2gram not in n2gram_list_to_exclude]

                #filtrando os n1grams (só entram os que não estão no n2grams)
                n1gram_DF_list_filtered = [n1gram for n1gram in n1gram_list if (n1gram not in n2gram_DF['token_0'].values) and (n1gram not in n2gram_DF['token_1'].values)]
                
                #concatenando os n1grams + n2grams
                term_list.extend( n2gram_list + n1gram_DF_list_filtered )
                
                print(f'Term_list montada com n1grams + n2grams (termo: {term})')
            
            except FileNotFoundError:
                term_list.extend( n1gram_list )
                print(f'Term_list montada com n1grams (termo: {term})')
        
        elif str(ngrams).lower() == 'x':

            try:
                nxgram_DF = pd.read_csv(diretorio + f'/Outputs/ngrams/semantic/nxgram_{term}.csv', index_col = 0)
                term_list.extend( nxgram_DF.index.tolist() )
            
            except FileNotFoundError:
                print(f'Erro! Nenhum arquivo nxgram com termos semânticos encontrado para o termo: {term}')
                return

    term_list_modified = [ re.sub('_', ' ' , term) for term in term_list ]

    #print(sorted(term_list_modified))
    #print(len(term_list_modified), len(prob_term_list))
    return term_list_modified


#------------------------------
def get_tokens_indexes(tokens_list, all_token_array = None):

    import numpy as np

    tokens_array = np.array(tokens_list, dtype=object)
    sorter = np.argsort(all_token_array)
    return sorter[ np.searchsorted(all_token_array, tokens_array, sorter = sorter) ]


#------------------------------
def get_tokens_from_sent(sent, tokens_list_to_filter = None, stopwords_list_to_remove = None, get_number_tokens = False, spacy_tokenizer = None, process_with_spacy = False):
    
    import time
    
    if spacy_tokenizer is not None and process_with_spacy is True:
        sent_tokens = [ token.lemma_ for token in spacy_tokenizer(sent) ]
    
    else:
        def regTokenize(text):
            import re
            WORD = re.compile(r'\w+')
            words = WORD.findall(text)
            return words
        
        #splitando a sentença em tokens
        sent_tokens = [ token.lower() for token in regTokenize(sent) ]
    
    #qunado não se quer pegar os tokens numéricos
    if get_number_tokens is False:
        sent_tokens = [ token for token in sent_tokens if token[0] in 'abcdefghijklmnopqrstuvwxyz' ]

    #print('TOKENS SENT: ', sent_tokens)
    
    #filtrando os tokens considerando o que está na lista
    if (tokens_list_to_filter is not None) and len(tokens_list_to_filter) > 0:
        sent_tokens_filtered = [ token for token in sent_tokens if token in tokens_list_to_filter]
    else:
        sent_tokens_filtered = sent_tokens
    
    #caso o filtro de stopwords seja usado
    if (stopwords_list_to_remove is not None) and len(stopwords_list_to_remove) > 0:
        sent_tokens_filtered = [ token for token in sent_tokens_filtered if token not in stopwords_list_to_remove]
    #print('FILTERED TOKENS SENT: ', sent_tokens_filtered)
    #time.sleep(0.1)

    return sent_tokens_filtered


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
