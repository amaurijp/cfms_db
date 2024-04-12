#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import math
import pandas as pd
import numpy as np
import regex as re
import random
import h5py
from joblib import dump, load
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords

from NNs import NN_model
from NNs import complete_NN_parameters_dic

from FUNCTIONS import get_filenames_from_folder
from FUNCTIONS import generate_PR_results
from FUNCTIONS import save_dic_to_json
from FUNCTIONS import update_log
from FUNCTIONS import load_log_info
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import error_print_abort_class
from FUNCTIONS import error_incompatible_strings_input
from FUNCTIONS import get_tag_name

from functions_TEXTS import concat_DF_sent_indexes
from functions_TOKENS import get_tokens_from_sent

from functions_TOKENS import find_min_max_token_sent_len

from functions_VECTORS import get_tv_from_sent_index
from functions_VECTORS import get_wv_from_sentence

import tensorflow.keras.backend as backend
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from tensorflow.keras.models import load_model
        
        


class stats(object):
    
    def __init__(self, diretorio = None):

        print('\n( Class: stats )')
        
        self.diretorio = diretorio

        #carregando os tokens do IDF
        print('Carregando os tokens do IDF')
        self.tokens_DF = pd.read_csv(self.diretorio + f'/Outputs/tfidf/IDF.csv', index_col = 0)
        self.all_tokens_array = self.tokens_DF.index.values
        self.n_tokens = len(self.all_tokens_array)
        self.stopwords_list = stopwords.words('english')
        print('n_tokens = ', self.n_tokens)
        
        #checando a pasta /Outputs/models/
        if not os.path.exists(self.diretorio + '/Outputs/models'):
            os.makedirs(self.diretorio + '/Outputs/models')


    def get_topic_prob_func(self, input_DF_file_names = ['pyrolysis_temperature_extracted_mode_match_topic_vector'], feature = 'pyrolysis_temperature'):

        print('\n( Function: get_topic_prob_func )')

        #criando o dicionário para os DFs
        DFs_concat_dic = {}
        input_DF_dic = {}

        #criando o DF da função prob
        topic_func_prob_DF = pd.DataFrame([], index = self.all_tokens_array)
        topic_func_prob_DF['token_index'] = np.arange(0, len(self.all_tokens_array), 1)        

        for input_DF_file_name in input_DF_file_names:        
            print('\ninput_DF_file_name: ', input_DF_file_name)
            #filtrando a input_DF_dic caso já haja na pasta ~/DataFrames a DF filtrada
            filter_input_DF = False
            #determinando o base name do input_DF_dic
            input_DF_basename = re.search(r'.*(?=extracted_)', input_DF_file_name).captures()[0]
            
            #obtendo os arquivos .csv na pasta /DataFrames
            files = get_filenames_from_folder(self.diretorio + '/Outputs/DataFrames', file_type = 'csv')
            for f in files:
                if input_DF_basename in f and 'FULL_DF' not in f:
                    #coletando os indices das sentenças cujos valores foram extraídos com sucesso das DFs extraídas (~/Extracted) para os DFs (~/DFs)
                    input_DF_dic_filtered = pd.read_csv(self.diretorio + f'/Outputs/DataFrames/{f}.csv', index_col=0)
                    input_DF_filtered_indexes = np.unique(input_DF_dic_filtered.iloc[ : , 2].values)
                    filter_input_DF = True
                    print('DF encontrada em: ', f'/Outputs/DataFrames/{f}.csv')
                    print('n indexes encontrados na DF filtrada: ', len(input_DF_filtered_indexes))
        
            #encontrando a função de probabilidade para as duas classes (sentenças desejadas e sentenças não desejadas)
            for filename in (f'{input_DF_file_name}', f'{input_DF_file_name}_random'):
                
                if filename[-7:] == '_random':
                    DF_type = 'random'
                else:
                    DF_type = 'target'
                
                #importando o DF com as sentenças encontradas
                print(f'Abrindo DF ~/Outputs/Extracted/{filename}.csv')
                print('DF_type: ', DF_type)
                try:
                    input_DF_dic[DF_type] = pd.read_csv(self.diretorio + f'/Outputs/Extracted/{filename}.csv', index_col=0)
                    print('n indexes:', len(input_DF_dic[DF_type].index))
                except FileNotFoundError:                    
                    print('Erro! DF não encontrada em: ', self.diretorio + f'/Outputs/Extracted/{filename}.csv')

                #manipulando a DF
                if DF_type == 'random':
                    print('renaming column: "rand_sent_index" -> "index"')                    
                    input_DF_dic[DF_type].rename(columns={ 'rand_sent_index' : 'index'}, inplace=True)
                    print('set index para: "index"')
                    input_DF_dic[DF_type] = input_DF_dic[DF_type].reset_index().set_index('index')
                    input_DF_dic[DF_type] = input_DF_dic[DF_type][['rand_sent']].copy()
                    
                else:
                    print('renaming column: "', input_DF_basename[:-1] + '_index" -> "index"')
                    print('renaming column: "', input_DF_basename[:-1] + '" -> "sent"')
                    input_DF_dic[DF_type].rename(columns={ input_DF_basename[:-1] + '_index' : 'index'}, inplace=True)
                    input_DF_dic[DF_type].rename(columns={ input_DF_basename[:-1] : 'sent'}, inplace=True)
                    print('set index para: "index"')
                    input_DF_dic[DF_type] = input_DF_dic[DF_type].reset_index().set_index('index')
                    input_DF_dic[DF_type] = input_DF_dic[DF_type][['sent']].copy()

                #filtrando o input_DF_dic
                if filter_input_DF is True and filename == f'{input_DF_file_name}':
                    #filtrando só as sentenças que foram extraídas para o DF na pasta ~/DataFrames            
                    input_DF_dic[DF_type] = input_DF_dic[DF_type].loc[input_DF_filtered_indexes]                    
                    print(f'Filtrando a DF "{input_DF_file_name}" a partir da DF montada em ~/Outputs/DataFrames')
                    print('n indexes:', len(input_DF_dic[DF_type].index))

                #dropando duplicatas
                input_DF_dic[DF_type].drop_duplicates(inplace=True)
                print('drop duplicates...')
                print('n indexes:', len(input_DF_dic[DF_type].index))
                                
                #definindo um input_DF_dic concatenado
                try:
                    DFs_concat_dic[DF_type] = pd.concat([DFs_concat_dic[DF_type], input_DF_dic[DF_type]], axis=0).drop_duplicates(keep='last')
                    DFs_concat_dic[DF_type].sort_index(inplace=True)
                except KeyError:
                    DFs_concat_dic[DF_type] = input_DF_dic[DF_type]

        for DF_type in ('target', 'random'):        
            print('* concat DF ', DF_type)
            print('* n indexes:', len(DFs_concat_dic[DF_type].index))

        #encontrando a função de probabilidade para as duas classes (sentenças desejadas e sentenças não desejadas)
        print('Processando as sentenças...')
        for DF_type in DFs_concat_dic.keys():
            
            if DF_type == 'random':
                column_name = 'counts_target_0'
            elif DF_type == 'target':
                column_name = 'counts_target_1'
            
            #definindo a coluna de count
            topic_func_prob_DF[column_name] = 0
            
            #contagem de tokens
            token_counting = 0
            token_counting_unique = []
            
            #contador de sentença
            sent_counter = 0
            
            #varrendo as sentenças
            for i in range(len(DFs_concat_dic[DF_type].iloc[ : , 0].values)):
                
                #analisando cada sentença
                sent = DFs_concat_dic[DF_type].iloc[i, 0]

                #splitando a sentença em tokens
                sent_tokens_filtered = get_tokens_from_sent(sent, tokens_list_to_filter = self.all_tokens_array, stopwords_list_to_remove = self.stopwords_list, spacy_tokenizer = nlp)
                
                #número total de tokens
                token_counting += len(sent_tokens_filtered)
                
                for token in sent_tokens_filtered:
                    #coletando tokens únicos
                    if token not in token_counting_unique:
                        token_counting_unique.append(token)
                    #fazendo a contagem
                    topic_func_prob_DF.loc[ token , column_name ] +=  1
                
                sent_counter += 1
                if sent_counter % 200 == 0:
                    print('n_sent processadas: ', sent_counter)
                                
            #encontrando a função de probabilidade para a classe
            topic_func_prob_DF[f'prob_func_{column_name}'] = ( topic_func_prob_DF[column_name].values / np.linalg.norm(topic_func_prob_DF[column_name].values, ord=1) )
            
            #fazendo o laplace smoothing
            # theta = (Xi + alpha) / (N + (alpha*d))
            #onde Xi = contagem do token, alpha = cte, N = contagem total de tokens, d=contagem de tokens únicos
            alpha = 1
            N = token_counting
            d = len(token_counting_unique)
            topic_func_prob_DF[f'theta_{column_name}'] = ( topic_func_prob_DF[column_name].values + 1 ) / ( N + (alpha*d) )                        
        
        #salvando a função de probabilidade em .csv
        print(f'Salvando DF ~/Outputs/models/prob_func_{feature}.csv')
        topic_func_prob_DF.to_csv(self.diretorio + f'/Outputs/models/prob_func_{feature}.csv')



    def set_bayes_classifier(self, prob_func_filename = 'pyrolysis_temperature'):
    
            
    
        #abrindo a função de probabilidade para o tópico
        self.prob_func_DF = pd.read_csv(self.diretorio + f'/Outputs/models/prob_func_{prob_func_filename}.csv', index_col=0)    



    def use_bayes_classifier(self, sent):        
        
        #abrindo a estatística de article_sent
        article_sent_stats = load_dic_from_json(self.diretorio + '/Outputs/log/stats_article_SENT.json')
        
        #print('\nBayesian classifier for sent:')
        #print(sent)
        
        #coletando os tokens da sentença                        
        sent_tokens = get_tokens_from_sent(sent, tokens_list_to_filter = self.all_tokens_array, stopwords_list_to_remove = self.stopwords_list, spacy_tokenizer=nlp)
                
        #definindo o valor de prior_Ck (probabilidade da sentença ser da classe; é aproximadamente 1 por artigo)
        #usamos como valor inicial = 1 / número médio de sentença por artigo
        prior_Ck = 1 / article_sent_stats['avg']
        post_Ck = prior_Ck
        #definindo o valor de prior_not_Ck 
        prior_not_Ck = ( 1 - prior_Ck )
        post_not_Ck = prior_not_Ck
        #print('( prior_Ck: ', post_Ck, ' , prior_not_Ck: ', post_not_Ck, ' )')
        try:
            for token in sent_tokens:
                #print(token)
                if token in self.all_tokens_array:
                    #calculando a probabilidade de ser da classe
                    #definindo o valor de support
                    support_Ck = self.prob_func_DF.loc[token , 'theta_counts_target_1' ]
                    #atualizando o valor de post
                    log_post_Ck = math.log( post_Ck ) + math.log( support_Ck )
                    post_Ck = math.exp( log_post_Ck )
    
                    #calculando a probabilidade de não ser da classe
                    #definindo o valor de support
                    support_not_Ck = self.prob_func_DF.loc[token , 'theta_counts_target_0' ]
                    #atualizando o valor de post
                    log_post_not_Ck = math.log( post_not_Ck ) + math.log( support_not_Ck )
                    post_not_Ck = math.exp( log_post_not_Ck )                
                    
                    norm_factor = np.linalg.norm((post_Ck, post_not_Ck), ord=1)
                    #print( '( post_Ck: ', post_Ck / norm_factor , ', post_not_Ck: ', post_not_Ck / norm_factor, ' )')
                    #time.sleep(0.5)        
            
            return post_Ck / norm_factor
        
        except (UnboundLocalError, ValueError):
            return 0



class train_machine(object):
    
    def __init__(self, machine_type = 'LSTM', wv_matrix_name = 'w2vec_xx', diretorio=None):

        print('\n( Class: train machine )')
        
        
        
        

        self.diretorio = diretorio
        self.machine_type = machine_type.lower()
        self.class_name = 'machine'
        self.wv_matrix_name = wv_matrix_name
        self.stopwords_list = stopwords.words('english')

        #testando os erros de inputs
        self.abort_class = error_incompatible_strings_input('machine_type', 
                                                            machine_type, ('conv1d',
                                                                           'conv1d_conv1d',
                                                                           'conv1d_lstm',
                                                                           'conv2d',
                                                                           'conv2d_conv1d',
                                                                           'conv2d_lstm',
                                                                           'logreg', 
                                                                           'lstm', 
                                                                           'randomforest', 
                                                                           'svm'), class_name = self.class_name)

        #caso seja uma NN
        if self.machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'conv2d', 'conv2d_conv1d', 'conv2d_lstm', 'lstm'):
            #carregando o WV
            try:
                self.wv = pd.read_csv(self.diretorio + f'/Outputs/wv/{self.wv_matrix_name}.csv', index_col = 0)
            
            except FileNotFoundError:
                print(f'Erro para a entrada wv_name: {self.wv_matrix_name}')
                print('Entradas disponíveis: olhar os arquivos .csv na pasta ~/Outputs/wv/')
                print('> Abortando a classe: machine')
                self.abort_class = True
                return        
            
            self.wv_emb_dim = len(self.wv.columns)
            
            #carregando a estatísticas dos word_vectors
            self.wv_stats = load_dic_from_json(self.diretorio + '/Outputs/wv/wv_stats.json')['w2vec'][self.wv_matrix_name]
            
            #carregando os tokens do IDF
            self.IDF = pd.read_csv(self.diretorio + f'/Outputs/tfidf/idf.csv', index_col = 0)
            self.all_tokens_array = self.IDF.index.values

        
        #caso use os topic vectors
        if self.machine_type in ('conv1d_conv1d', 'conv1d_lstm', 'conv2d_conv1d', 'conv2d_lstm', 'logreg', 'randomforest', 'svm'):
            #carregando a matriz DOC_TOPIC
            h5 = h5py.File(self.diretorio + f'/Outputs/models/sent_topic_full_matrix.h5', 'r')
            self.doc_topic_matrix = h5['data']

            #carregando a estatísticas dos topic_vectors
            self.tv_stats = load_dic_from_json(self.diretorio + '/Outputs/models/sent_topic_matrix_stats.json')



    def set_train_sections(self, 
                           section_name = 'methodolody',
                           sent_batch_size = 5,
                           article_batch_size = 5,
                           n_epochs = 5,
                           sent_stride = 1,
                           load_architecture_number = False):


        self.section_name = section_name
        self.sent_batch_size = sent_batch_size
        self.article_batch_size = article_batch_size
        self.n_epochs = n_epochs
        self.sent_stride = sent_stride

        print('\n( Function: set_train_sections )')

        #check para redes neurais
        self.nn_check = False

        #caso já exista a arquitetura desejada a ser treinada
        if load_architecture_number is not False:
            models_architectures = load_dic_from_json(self.diretorio + f'/Outputs/models/models_architectures.json')
            self.n_architecture = get_tag_name( load_architecture_number , prefix = '')
            self.models_parameters_dic = models_architectures[ self.n_architecture ]

            #caso seja NN
            if self.machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'conv2d', 'conv2d_conv1d', 'conv2d_lstm', 'lstm'):
                
                self.nn_check = True
                
                #definido o maior valor permitido de sent_token_len (quantidade de tokens por sentença) o qual será o input da NN
                sent_tokens_stats = load_dic_from_json(self.diretorio + '/Outputs/log/stats_sents_tokens_len_filtered.json')
                self.sent_max_token_len , self.sent_min_token_len , self.single_sent_min_token_len = find_min_max_token_sent_len(sent_batch_size = self.sent_batch_size, 
                                                                                                                                 machine_type = self.machine_type, 
                                                                                                                                 sent_tokens_stats = sent_tokens_stats)            

        #caso a arquitetura não seja definida
        else:

            #definindo os parâmetros da NN
            self.models_parameters_dic = {}        
            
            #general parameters
            self.models_parameters_dic['machine_type'] = self.machine_type
            self.models_parameters_dic['feature'] = None
            self.models_parameters_dic['section_name'] = self.section_name
            self.models_parameters_dic['batch_size'] = 50
            self.models_parameters_dic['n_epochs'] = self.n_epochs
            self.models_parameters_dic['sent_batch_size'] = self.sent_batch_size

            #caso seja NN
            if self.machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'conv2d', 'conv2d_conv1d', 'conv2d_lstm', 'lstm'):
                
                self.nn_check = True
                
                #definido o maior valor permitido de sent_token_len (quantidade de tokens por sentença) o qual será o input da NN
                sent_tokens_stats = load_dic_from_json(self.diretorio + '/Outputs/log/stats_sents_tokens_len_filtered.json')
                self.sent_max_token_len , self.sent_min_token_len , self.single_sent_min_token_len = find_min_max_token_sent_len(sent_batch_size = self.sent_batch_size, 
                                                                                                                                 machine_type = self.machine_type, 
                                                                                                                                 sent_tokens_stats = sent_tokens_stats)

                #NN inputs
                self.models_parameters_dic['input_shape'] = {}
                
                #definindo o input de word_vector
                if self.machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'lstm'):
                    self.models_parameters_dic['input_shape']['wv'] = [ self.sent_max_token_len, self.wv_emb_dim ] #word vector

                elif self.machine_type in ('conv2d', 'conv2d_conv1d', 'conv2d_lstm'): 
                    self.models_parameters_dic['input_shape']['wv'] = [ self.sent_max_token_len, self.wv_emb_dim, self.sent_batch_size ] #word vector
                
                #caso as NNs vão usar os tvs
                if self.machine_type in ('conv1d_conv1d', 'conv1d_lstm', 'conv2d_conv1d', 'conv2d_lstm'):
                    #definindo o input de topic_vector para as NNs
                    self.models_parameters_dic['input_shape']['tv'] = [ self.sent_batch_size * self.doc_topic_matrix.shape[1], 1 ] #topic vector
                
                #completar os parametros que faltam
                self.models_parameters_dic = complete_NN_parameters_dic(self.models_parameters_dic, machine_type = self.machine_type)

            #procurando setups de NNs já criadas
            self.n_architecture = None
            if os.path.exists(self.diretorio + f'/Outputs/models/models_architectures.json'):
                models_architectures = load_dic_from_json(self.diretorio + f'/Outputs/models/models_architectures.json')
                for key in models_architectures:
                    if self.models_parameters_dic == models_architectures[key]:
                        self.n_architecture = key

                #caso a arquitetura seja nova,cria-se um número subsequente
                if self.n_architecture is None:
                    self.n_architecture = get_tag_name( int( sorted(models_architectures.keys())[-1] ) + 1 , prefix = '')
                    models_architectures[ str(self.n_architecture) ] = self.models_parameters_dic
                    save_dic_to_json(self.diretorio + f'/Outputs/models/models_architectures.json', models_architectures)

            
            #caso não haja o arquivo dic com nenhuma arquitetura de modelo
            else:
                self.n_architecture = "00001"
                models_architectures = {}
                models_architectures[ "00001" ] = self.models_parameters_dic
                save_dic_to_json(self.diretorio + f'/Outputs/models/models_architectures.json', models_architectures)



    def train_on_sections(self):
        
        print('\n( Function: train_on_sections )')

        #checando erros de instanciação/inputs
        abort_class = error_incompatible_strings_input('machine_type', self.machine_type, ('conv1d',
                                                                                           'conv1d_conv1d',
                                                                                           'conv1d_lstm',
                                                                                           'conv2d',
                                                                                           'conv2d_conv1d',
                                                                                           'conv2d_lstm',
                                                                                           'logreg', 
                                                                                           'lstm', 
                                                                                           'randomforest', 
                                                                                           'svm'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
        
        

        print(f'\nModel: sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}\n')
        
        #checando a pasta /Outputs/sections/
        if not os.path.exists(self.diretorio + '/Outputs/sections'):
            os.makedirs(self.diretorio + '/Outputs/sections')
        
        #coletando os filenames para treinar as seções
        filenames = get_filenames_from_folder(self.diretorio + '/Outputs/sections', file_type = 'csv')
        if len(filenames) == 0:    
            print('Erro! Não há arquivos para treinar na pasta ~/Outputs/sections/')
            print('> Abortando função: train_machine.train_on_sections')
            return        
        
        #contador de arquivos (para parar e resumir o fit)
        try:
            file_name = load_log_info(log_name = f'counter_file_index_sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}', 
                                      logpath = self.diretorio + '/Outputs/log/train_sections.json')
            file_name = file_name if file_name is not None else filenames[0]
            print('Training_counter_file_index: ', file_name)
            file_index = filenames.index(file_name) + 1
        except FileNotFoundError:
            file_index = 0
                    
        #definindo as listas para coleta dos dados para processamento de cada batch de arquivos
        X_wv_train = []
        X_tv_train = []
        Y_train = []
        X_wv_test = []
        X_tv_test = []
        Y_test = []
        
        #analisando quais vetores serão usados
        get_wv = False
        get_tv = False
        if self.machine_type in ('conv1d', 'conv2d', 'lstm'):
            get_wv = True
            get_tv = False                    
        elif self.machine_type in ('logreg', 'randomforest', 'svm'):
            get_wv = False
            get_tv = True                    
        else:
            get_wv = True
            get_tv = True

        #treinado os modelos com batch de sentenças
        go_to_train = False
        got_article_counter = 0
        batch_counter= 1
        print('Preparing data to train/test sections filter...')
        for filename in filenames[ file_index : ]:

            print('Processing ', filename, '...')
            section_counter = 0
            sentDF = pd.read_csv(self.diretorio + f'/Outputs/sections/{filename}.csv', index_col=0)

            #limites de indexes            
            index_first = sentDF.index.values[0]
            index_last = sentDF.index.values[-1]
            #print('(first, last) sent indexes: ', index_first, index_last)            

            #encontrando os index para as sentenças do documento (artigo)
            sent_indexes = range( index_first, ( index_last + 2) - self.sent_batch_size, self.sent_stride)            
            #print('Sent indexes: ', list(sent_indexes) )

            try:
                #varrendo as sentenças  
                for start_index in sent_indexes:                
                    
                    #definindo uma lista para coletar somente os targets
                    Y_target_section = []
                    
                    #determinando os indexes para cada conjunto de sentenças (section)
                    section_indexes = range( start_index , start_index + self.sent_batch_size )
                    #print('Sent indexes to collect: ', list(section_indexes))
                    #varrendo os indexes para pegar um conjunto de sentenças
                    min_max_section_token_len_check = True
                    min_sent_token_len_check = True

                    for sent_index in section_indexes:
                        #print('Getting sentence vectors - sent index: ', sent_index)
                        #sentença
                        sent = sentDF.loc[sent_index, 'Sentence']
                        #target
                        target = sentDF.loc[sent_index, self.section_name]

                        #coletando os wvs
                        if get_wv is True:

                            #before = time.time()
                            #colentando os word vectors (scaled) da sentença (section)
                            sent_word_vectors = get_wv_from_sentence(sent,
                                                                     self.wv,
                                                                     self.wv_stats,
                                                                     stopwords_list = self.stopwords_list,
                                                                     spacy_tokenizer = nlp)
                            #print('get_wv_from_sentence  time spent: ', time.time() - before)
                            #caso o token_sent_len for menor que o determinado no SENT_TOKEN_STATS
                            if sent_word_vectors is None or sent_word_vectors.shape[0] < self.single_sent_min_token_len:
                                #print('Ignorando sentence - sent len:', sent_word_vectors.shape[0], ' (min_sent_token_len allowed: ', single_sent_min_token_len, ')')
                                min_sent_token_len_check = False
                                break
                            
                            #coletando o wv e o target
                            if self.machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'lstm'):
                                #before = time.time()
                                try:
                                    X_wv_section = np.vstack((X_wv_section, sent_word_vectors))
                                except NameError:
                                    X_wv_section = sent_word_vectors
                                
                                #print('vstack time spent: ', time.time() - before)
                                #print('X_wv_section.shape = ', X_wv_section.shape)
                            
                            elif self.machine_type in ('conv2d', 'conv2d_conv1d', 'conv2d_lstm'):
                                
                                #caso o número de tokens na sentença seja maior que o permitido
                                if (self.sent_min_token_len <= sent_word_vectors.shape[0] <= self.sent_max_token_len):
                                    zero_m = np.zeros((self.sent_max_token_len - sent_word_vectors.shape[0], self.wv_emb_dim))
                                    sent_word_vectors = np.vstack((sent_word_vectors, zero_m))                            
                                else:
                                    #print('Ignorando section - concat sent len:', sent_word_vectors.shape[0], ' (min_sent_token_len, ', sent_min_token_len, '; max_sent_token_len: ', sent_max_token_len, ')')
                                    min_max_section_token_len_check = False
                                    break
                                
                                #reshaping
                                sent_word_vectors = sent_word_vectors.reshape(1, sent_word_vectors.shape[0], sent_word_vectors.shape[1])
                                
                                try:
                                    X_wv_section = np.vstack((X_wv_section, sent_word_vectors))                                
                                except NameError:
                                    X_wv_section = sent_word_vectors
                                
                                #print('X_wv_section.shape = ', X_wv_section.shape)
            
                        #coletando os tvs
                        if get_tv is True:

                            #coletando o vector topico da sentença
                            sent_topic_vectors = get_tv_from_sent_index(sent_index,
                                                                        tv_stats = self.tv_stats, 
                                                                        scaling = False,
                                                                        normalize = True,
                                                                        doc_topic_matrix = self.doc_topic_matrix)
                            
                            #caso seja uma sentença com nenhum token presente na IDF
                            if sent_topic_vectors is None:
                                min_sent_token_len_check = False
                                break
                            
                            try:
                                X_tv_section = np.hstack((X_tv_section, sent_topic_vectors))
                            except NameError:
                                X_tv_section = sent_topic_vectors
                            
                            #print('X_tv_section.shape = ', X_tv_section.shape)                        
                        
                        Y_target_section.append(target)
                        #print('len(Y_target_section) = ', len(Y_target_section))
                        #X_test_section.append(sent)
                    
                    #caso todos os target sejam iguais (ou tudo 0; ou tudo 1) e se todas as sentenças tiverem o sent_token_len mínimo (determinado pelo SENT_TOKEN_STATS)
                    if len(set(Y_target_section)) == 1 and min_sent_token_len_check is True:                    

                        #padding dos conjunto de dados de word-vectors das sentenças                    
                        if self.machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'lstm'):
                            if (self.sent_min_token_len <= X_wv_section.shape[0] <= self.sent_max_token_len):
                                zero_m = np.zeros((self.sent_max_token_len - X_wv_section.shape[0], self.wv_emb_dim))
                                X_wv_section = np.vstack((X_wv_section, zero_m))
                                #print('Final X_wv_section.shape = ', X_wv_section.shape)
                            
                            #caso o número de tokens nas sentenças concatenadas seja maior que o permitido
                            else:
                                #print('Ignorando section - concat sent len:', X_wv_section.shape[0], ' (min_sent_token_len, ', sent_min_token_len, '; max_sent_token_len: ', sent_max_token_len, ')')
                                min_max_section_token_len_check = False
                                                
                        elif self.machine_type in ('conv2d', 'conv2d_conv1d', 'conv2d_lstm'):
                            #fazendo o reshape da matrix de seção para encontrar os channels
                            #(self.sent_batch_size , max_token_len , wv_dim) -> (max_token_len , wv_dim, self.sent_batch_size)
                            X_wv_section = np.dstack(X_wv_section)
                            #print('Final X_wv_section.shape = ', X_wv_section.shape)
                        
                        #imprimindo o shape final para os topic vectors
                        if self.machine_type in ('conv1d_conv1d', 'conv1d_lstm', 'conv2d_conv1d', 'conv2d_lstm', 'logreg', 'randomforest', 'svm'):
                            #print('Final X_tv_section.shape = ', X_tv_section.shape)
                            pass

                        if min_max_section_token_len_check is True:

                            #determinando uma probabilidade
                            r_prob = random.random()
                        
                            if r_prob > 0.2:
                                Y_train.append( list(set(Y_target_section))[0] )
                                
                            else:
                                Y_test.append( list(set(Y_target_section))[0] )

                            #separando entre train e test                
                            if get_wv is True:

                                #print('X_wv_section.shape: ', np.array(X_wv_section).shape )

                                if r_prob > 0.2:
                                    X_wv_train.append( X_wv_section )
                                    
                                else:
                                    X_wv_test.append( X_wv_section )
                                    
                            if get_tv is True:

                                #print('X_tv_section.shape: ', np.array(X_tv_section).shape )   

                                if r_prob > 0.2:
                                    X_tv_train.append( X_tv_section )
                                    
                                else:
                                    X_tv_test.append( X_tv_section )
                                
                            section_counter += 1
                            #print('( X_wv_train.shape ; X_tv_train.shape ) ( ', np.array(X_wv_train).shape, ' ; ', np.array(X_tv_train).shape, ' ) Last Y_train: ', Y_train[-5 : ] )
                            #print('( X_wv_test.shape  ; X_tv_test.shape  ) ( ', np.array(X_wv_test).shape, ' ; ', np.array(X_tv_test).shape, ' ) Last Y_test: ', Y_test[-5 : ] )
                            #print('( len(Y_train)  ; len(Y_test)  ) ( ', len(Y_train), ' ; ',  len(Y_test), ' )' )
                            #print('Section counter: ', section_counter)                        
                        
                        del Y_target_section
                        if get_wv is True:
                            del X_wv_section
                        if get_tv is True:
                            del X_tv_section

                    else:
                        #deletando as arrays da seção
                        del Y_target_section
                        try:                    
                            if get_wv is True:
                                del X_wv_section
                            if get_tv is True:
                                del X_tv_section
                        except UnboundLocalError:
                            pass
            
            #caso haja alguma seção que não foi encontrada no processo de identificação
            except KeyError:
                continue


            #contador de artigoss processados    
            got_article_counter += 1            
            
            #caso o número de artigos processados seja igual ao batch size inserido ou caso seja o último arquivo da pasta
            if got_article_counter == self.article_batch_size or filename == filenames[-1]:
                go_to_train = True
                    
            if go_to_train is True:
        
                #caso a máquina seja uma NN
                if self.nn_check is True:
                    print('(Input) Informações dos inputs das NNs')
                    #definindo a NN
                    if get_wv is True:
                        print('Número de instâncias de wvs para o treino: ', len(X_wv_train))
                        print('Shape de cada instância de wvs: ', X_wv_train[0].shape)

                    if get_tv is True:
                        print('Número de instâncias de tvs para o treino: ', len(X_tv_train))
                        print('Shape de cada instância de tv: ', X_tv_train[0].shape)

                    NN = NN_model()
                    NN.set_parameters(self.models_parameters_dic, architecture = self.n_architecture)
                    model = NN.get_model()
                
                #caso a máquina seja de logistic regression
                elif self.machine_type == 'logreg':
                    model_save_folder = self.diretorio + f'/Outputs/models/sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.joblib'
                    if os.path.exists(model_save_folder):
                        print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
                        print('Carregando o arquivo joblib com o modelo...')
                        model = load(model_save_folder)
                    else:
                        print('Criando modelo ', self.machine_type)
                        model = LogisticRegression(warm_start=False, solver='lbfgs')
                
                #caso a máquina seja de svm
                elif self.machine_type == 'svm':
                    model_save_folder = self.diretorio + f'/Outputs/models/sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.joblib'
                    if os.path.exists(model_save_folder):
                        print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
                        print('Carregando o arquivo joblib com o modelo...')
                        model = load(model_save_folder)
                    else:
                        print('Criando modelo ', self.machine_type)
                        model = LinearSVC()
                        
                #caso a máquina seja de random forest
                elif self.machine_type == 'randomforest':
                    model_save_folder = self.diretorio + f'/Outputs/models/sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.joblib'
                    if os.path.exists(model_save_folder):
                        print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
                        print('Carregando o arquivo joblib com o modelo...')
                        model = load(model_save_folder)
                    else:
                        print('Criando modelo ', self.machine_type)
                        model = RandomForestClassifier()
                    
                #dicionário com os resultados do test set
                if os.path.exists(self.diretorio + f'/Outputs/models/sections_{self.section_name}_training_results.json'): 
                    test_results = load_dic_from_json(self.diretorio + f'/Outputs/models/sections_{self.section_name}_training_results.json')
                    try:
                        test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']
                    except KeyError:
                        test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}'] = {}
                        test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['acc'] = []
                        test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['loss'] = []
                        test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['precision'] = []
                        test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['recall'] = []
                else:
                    test_results = {}
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}'] = {}
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['acc'] = []
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['loss'] = []
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['precision'] = []
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['recall'] = []


                print(f'\nSentence batch: {batch_counter}')
                print('Shuffling...')
                #gerando indeces embaralhados para o treinamento
                index_list = list(range(len(Y_train)))
                random.shuffle(index_list)

                #coletando o Y
                Y_train_shuffled = []
                for index in index_list:
                    Y_train_shuffled.append( Y_train[ index ] )
                
                #convertendo para array
                Y_train = np.array(Y_train_shuffled)                
                Y_test = np.array(Y_test)
                print('Y_train: ', Y_train.shape)
                print('Y_test: ', Y_test.shape)

                #coletando o X para wv
                if get_wv is True:
                    
                    X_wv_train_shuffled = []                    
                    for index in index_list:
                        X_wv_train_shuffled.append( X_wv_train[index] )
                    
                    print('Converting to array...')
                    #array dos conjunto de dados de word-vectors e topic-vectors das sentenças
                    X_wv_train = np.array(X_wv_train_shuffled)
                    X_wv_test = np.array(X_wv_test)

                    print('X_wv_train: ', X_wv_train.shape)
                    print('X_wv_test: ', X_wv_test.shape)
                    
                #coletando o X para tv
                if get_tv is True:
                    
                    X_tv_train_shuffled = []
                    for index in index_list:                    
                        X_tv_train_shuffled.append(X_tv_train[index])
                    
                    print('Converting to array...')
                    #array dos conjunto de dados de word-vectors e topic-vectors das sentenças
                    X_tv_train = np.array(X_tv_train_shuffled)
                    X_tv_test = np.array(X_tv_test)

                    print('X_tv_train: ', X_tv_train.shape)
                    print('X_tv_test: ', X_tv_test.shape)


                print('Training model...')                
                #training                
                if self.machine_type in ('conv1d', 'lstm'):

                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_train.shape)
                    print('Y_train shape (n_samples, ):', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = X_wv_train , y = Y_train, epochs=self.n_epochs, batch_size = self.models_parameters_dic['batch_size'], verbose=1)

                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    results = model.evaluate(x = X_wv_test , y = Y_test, batch_size = self.models_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {results[0]}, Acc: {results[1]}')

                elif self.machine_type == 'conv2d':

                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim, self.sent_batch_size): ', X_wv_train.shape)
                    print('Y_train shape (n_samples, ):', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = X_wv_train , y = Y_train, epochs=self.n_epochs, batch_size = self.models_parameters_dic['batch_size'], verbose=1)

                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim, self.sent_batch_size): ', X_wv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    results = model.evaluate(x = X_wv_test , y = Y_test, batch_size = self.models_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {results[0]}, Acc: {results[1]}')

                elif self.machine_type == 'conv1d_lstm':
                    pass

                elif self.machine_type == 'conv1d_conv1d':

                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_train.shape)
                    print('Y_train shape (n_samples, ):', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = ( X_wv_train, X_tv_train ) , y = Y_train, epochs=self.n_epochs, batch_size = self.models_parameters_dic['batch_size'], verbose=1)

                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    results = model.evaluate(x = ( X_wv_test, X_tv_test ) , y = Y_test, batch_size = self.models_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {results[0]}, Acc: {results[1]}')
            
                elif self.machine_type == 'conv2d_conv1d':
                    
                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_train.shape)
                    print('Y_train shape (n_samples, ):', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = ( X_wv_train, X_tv_train ) , y = Y_train, epochs=self.n_epochs, batch_size = self.models_parameters_dic['batch_size'], verbose=1)

                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    results = model.evaluate(x = ( X_wv_test, X_tv_test ) , y = Y_test, batch_size = self.models_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {results[0]}, Acc: {results[1]}')                    

                elif self.machine_type == 'conv2d_lstm':
                    pass

                elif self.machine_type in ('logreg', 'randomforest'):
                    
                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_tv_train shape (n_samples, tv_dim): ', X_tv_train.shape)
                    print('Y_train shape: (n_samples, )', Y_train.shape)
                    print('Fitting...')
                    model.fit(X_tv_train, Y_train)

                    print('X_tv_test shape (n_samples, tv_dim): ', X_tv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    #for i in range(len(X_tv_test)):
                    #sample = X_tv_test[i].reshape(1,-1)
                    prediction_results = model.predict_proba(X_tv_test)
                    proba_threshold = 0.5
                    results = [0, 0]
                    results[0], results[1] = generate_PR_results(prediction_results, Y_test, proba_threshold = proba_threshold)
                    print(f'Evaluate results - Precision: {results[0]} ; Recall: {results[1]} ; Proba_thr: {proba_threshold}')

                elif self.machine_type == 'svm':
                    
                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_tv_train shape (n_samples, tv_dim): ', X_tv_train.shape)
                    print('Y_train shape: (n_samples, )', Y_train.shape)
                    print('Fitting...')
                    model.fit(X_tv_train, Y_train)

                    print('X_tv_test shape (n_samples, tv_dim): ', X_tv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    #for i in range(len(X_tv_test)):
                    #sample = X_tv_test[i].reshape(1,-1)
                    prediction_results = model.predict(X_tv_test)
                    proba_threshold = 0.5
                    results = [0, 0]
                    results[0], results[1] = generate_PR_results(prediction_results, Y_test, proba_threshold = proba_threshold)
                    print(f'Evaluate results - Precision: {results[0]} ; Recall: {results[1]} ; Proba_thr: {proba_threshold}')

                else:
                    print('Erro! Entrada errada para a variável "machine_type"')
                    print('> Abortando função: train_machine.train_on_sentences')
                    return
                
                if self.nn_check is True:
                    #salvando o modelo h5            
                    print('Salvando o modelo de ', self.machine_type)
                    model_save_folder = self.diretorio + f'/Outputs/models/sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
                    model.save(model_save_folder)
                
                elif self.machine_type in ('logreg', 'randomforest', 'svm'):
                    #salvando o modelo joblib    
                    print('Salvando o modelo de ', self.machine_type)
                    model_save_folder = self.diretorio + f'/Outputs/models/sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.joblib'
                    dump(model, model_save_folder)
                
                #salvando o número do último arquivo processado
                print('Salvando o file_index...')
                
                update_log(log_names = [f'counter_file_index_sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}'], 
                           entries = [filename],
                           logpath = self.diretorio + '/Outputs/log/train_sections.json')

                #salvando os resultados
                print('Salvando o test results...')
                if self.nn_check is True:
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['loss'].append( results[0] )
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['acc'].append( results[1] )
                    save_dic_to_json(self.diretorio + f'/Outputs/models/sections_{self.section_name}_training_results.json',
                                     test_results)
                    
                    self.plot_line(test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['acc'], plot_title = 'accuracy')
                    self.plot_line(test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['loss'], plot_title = 'loss')
                
                elif self.machine_type in ('logreg', 'randomforest', 'svm'):
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['precision'].append( results[0] )
                    test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['recall'].append( results[1] )
                    save_dic_to_json(self.diretorio + f'/Outputs/models/sections_{self.section_name}_training_results.json',
                                     test_results)

                    self.plot_line(test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['precision'], plot_title = 'precision')
                    self.plot_line(test_results[f'sections_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}']['recall'], plot_title = 'recall')
                

                #limpando as listas para os novos batchs    
                X_wv_train=[]
                X_wv_test=[]
                X_tv_train=[]
                X_tv_test=[]
                Y_train = []
                Y_test = []
                got_article_counter = 0
                batch_counter += 1
                go_to_train = False
                del test_results
                
                if self.nn_check is True:
                    #deletando o graph da última sessão
                    backend.clear_session()



    def plot_line(self, value_list: list, plot_title: str = None):

        fig, axes = plt.subplots(1, 1, figsize=(10,10), dpi=300)

        axes.set_title(plot_title)
        axes.plot( range(1, len(value_list) + 1), value_list, linewidth=2, color='blue', alpha=0.5, label='values')
        axes.axhline( sum(value_list) / len(value_list), linewidth=3, color='red', alpha=0.5, label='m_avg')
        axes.set_xlabel('batch')
        axes.set_ylabel('%')
        axes.legend()

        fig.savefig(self.diretorio + f'/Outputs/models/sections_{plot_title}_{self.section_name}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.png' )



class use_machine(object):
    
    def __init__(self, model_name = '', diretorio = None):

        print('\n( Class: use_machine )')

        
        
        
        
        self.diretorio = diretorio
        self.model_name = model_name

        #checando se há algum modelo a ser carregado
        self.model_found = False
        if os.path.exists(self.diretorio + f'/Outputs/models/{self.model_name}.h5') == True or os.path.exists(self.diretorio + f'/Outputs/models/{self.model_name}.joblib'):                        
            self.sent_batch_size = int( re.search(r'ch_[0-9]', self.model_name).captures()[0][ -1 : ] )
            self.machine_type = re.findall(r'(conv1d_conv1d|conv1d_lstm|conv1d|conv2d_conv1d|conv2d_lstm|conv2d|lstm|logreg|randomforest|svm)', self.model_name)[0]    
            self.stopwords_list = stopwords.words('english')
            self.model_found = True



    def set_machine_parameters_to_use(self, tokens_list = None, wv_type = None, wv_matrix_name = None, wv_matrix = None, doc_topic_matrix = None):

        print('\n( Function: set_machine_parameters_to_use )')

        self.wv = wv_matrix
        self.wv_emb_dim = len(self.wv.columns)
        self.doc_topic_matrix = doc_topic_matrix
        self.all_tokens_array = tokens_list

        #carregando a estatísticas dos topic_vectors
        self.tv_stats = load_dic_from_json(self.diretorio + '/Outputs/models/sent_topic_matrix_stats.json')

        #carregando a estatísticas dos word_vectors
        self.wv_stats = load_dic_from_json(self.diretorio + '/Outputs/wv/wv_stats.json')[wv_type][wv_matrix_name]

        #caso a máquina seja uma rede neural
        if self.machine_type in ('conv1d', 'conv1d_conv1d', 'conv1d_lstm', 'conv2d', 'conv2d_conv1d', 'conv2d_lstm', 'lstm'):
            
            #deletando o graph da última sessão
            backend.clear_session()

            #carregando as estatísticas de sentenças
            self.sent_tokens_stats = load_dic_from_json(self.diretorio + '/Outputs/log/stats_sents_tokens_len_filtered.json')

            #self.session = tf.Session()
            #self.graph = tf.get_default_graph()
            #set_session(self.session)

            #definido o maior valor permitido de sent_token_len (quantidade de tokens por sentença) o qual será o input da NN
            self.sent_max_token_len , self.sent_min_token_len , self.single_sent_min_token_len = find_min_max_token_sent_len(sent_batch_size = self.sent_batch_size, 
                                                                                                                             machine_type = self.machine_type, 
                                                                                                                             sent_tokens_stats = self.sent_tokens_stats)

            #carregando o modelo
            self.model = load_model(self.diretorio + f'/Outputs/models/{self.model_name}.h5')
            print('Section filter encontrado: ', self.diretorio + f'/Outputs/models/{self.model_name}')
            time.sleep(1)
        
        #caso a máquina seja um modelo do sklearn (não está sendo usado)
        if self.machine_type in ('logred', 'randomforest', 'svm'):
            
            #carregando o modelo sklearn
            model_folder = self.diretorio + f'/Outputs/models/{self.model_name}.joblib'
            self.model = load(model_folder)
            print('Section filter encontrado: ', self.model_name)
            time.sleep(1)



    def check_sent_in_section_for_ML(self, sent_index, DF = 'sentDF'): #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXfalta colocar aqui um for loop para concatenar as sentenças adjacentes
        
        
        #carregando a estatísticas dos topic_vectors
        self.tv_stats = load_dic_from_json(self.diretorio + '/Outputs/tfidf/H5/DOC_TOPIC_MATRIX_STATS.json')
        
        self.text_file_DF = DF        
        
        #print('\nML analyzing sent...') 
        #print('sent_index: ', sent_index)
        #print('Sent: ', self.text_file_DF.loc[ sent_index , 'Sentence'])

        #carregando a matriz DOC_TOPIC
        h5 = h5py.File(self.diretorio + f'/Outputs/tfidf/H5/doc_topic_full.h5', 'r')
        doc_topic_matrix = h5['data']

        #coletando o vector topico da sentença
        topic_vector = get_tv_from_sent_index(sent_index, 
                                              sent_indexes_slices = None, 
                                              tv_stats = self.tv_stats, 
                                              doc_topic_matrix = doc_topic_matrix)
        
        result_proba = self.model.predict_proba(topic_vector)
        
        return result_proba



    def check_sent_in_section_for_NN(self, sent_index, DF = 'sentDF', n2grams_DF = None, n3grams_DF = None):

        #carregando a estatísticas dos word_vectors
        self.text_file_DF = DF        
        
        print('NN analyzing sent index ', sent_index, ' in ', self.model_name)
        #print('Sent: ', self.text_file_DF.loc[ sent_index , 'Sentence'])
    
        #obtendo os indexes das sentenças concatenadas (adjacentes ao sent_index)
        concat_sent_indexes = concat_DF_sent_indexes(sent_index, self.sent_batch_size)
    
        #lista para coletar os resultados
        results = []
    
        #varrendo as sentenças concatenadas
        for i in concat_sent_indexes:
            try:
                #concatenando os word vectors das sentenças
                concat_wvs = self.get_concat_sents_wvs(i, 
                                                       n2grams_DF = n2grams_DF, 
                                                       n3grams_DF = n3grams_DF)
                
                #checando o sent_token_len
                if (self.sent_min_token_len <= concat_wvs.shape[0] <= self.sent_max_token_len):
                    #fazendo o padding
                    if concat_wvs.shape[0] < self.sent_max_token_len:
                        zero_m = np.zeros((self.sent_max_token_len - concat_wvs.shape[0], 300))
                        concat_wvs = np.vstack((concat_wvs, zero_m))
                    else:
                        pass
                else:
                    print('concatenated sents outsized (continue)...')
                    continue
                
                #fazendo o reshape
                concat_wvs = concat_wvs.reshape(1, concat_wvs.shape[0], concat_wvs.shape[1])
                
                #caso o modelo use os topic vectors
                if self.machine_type in ('conv1d_conv1d', 'conv1d_lstm', 'conv2d_conv1d', 'conv2d_lstm'):
                    
                    #concatenando os word vectors das sentenças
                    concat_tvs = self.get_concat_sents_tvs(i)

                    #caso nenhum dos dois vetores concateandos seja None
                    if concat_wvs is not None and concat_tvs is not None:
                
                        #fazendo o reshape
                        concat_tvs = concat_tvs.reshape(1, concat_tvs.shape[0], 1)  
                        
                        #fazendo o predict
                        #print('sent_wv_shape: ', concat_wvs.shape)
                        #print('sent_tv_shape: ', concat_tvs.shape)
                        result = self.model.predict([concat_wvs, concat_tvs])
                        results.append(result[0][0])
                
                #caso só use os wvs
                else:
                    if concat_wvs is not None:
                        #fazendo o predict
                        #print('sent_wv_shape: ', concat_wvs.shape)
                        result = self.model.predict(concat_wvs)
                        results.append(result[0][0])
            
            #caso algum index esteja fora de range
            except KeyError:
                continue
        
        #print('Results: ', results)
        try:
            return max(results)
        except ValueError:
            return 0



    def get_concat_sents_wvs(self, sent_indexes, n2grams_DF = None, n3grams_DF = None):    
             
                
        for sent_index in sent_indexes:
            #print('Getting sentence vectors - sent index: ', i)
            #sentença
            sent = self.text_file_DF.loc[sent_index, 'Sentence']
            
            #colentando os word vectors (scaled) da sentença (section)
            sent_word_vectors = get_wv_from_sentence(sent, 
                                                     self.wv,
                                                     self.wv_stats,
                                                     stopwords_list = self.stopwords_list,
                                                     spacy_tokenizer = nlp)
            
            #concatenando os word vectors
            #caso haja token na sentença
            if sent_word_vectors is not None and sent_word_vectors.shape[0] > 0:
                try:
                    concat_wvs = np.vstack(( concat_wvs , sent_word_vectors ))
                except NameError:
                    concat_wvs = sent_word_vectors

        return concat_wvs



    def get_concat_sents_tvs(self, sent_indexes):
        

        for sent_index in sent_indexes:
            
            #coletando os vectores topicos concatenados da sentença
            sent_topic_vectors = get_tv_from_sent_index(sent_index,
                                                        tv_stats = self.tv_stats, 
                                                        scaling = False,
                                                        normalize = True,
                                                        doc_topic_matrix = self.doc_topic_matrix)
            
            #caso seja uma sentença com nenhum token presente na IDF
            if sent_topic_vectors is None:
                concat_tvs = None
                break
            
            try:
                concat_tvs = np.hstack((concat_tvs, sent_topic_vectors))
            except NameError:
                concat_tvs = sent_topic_vectors
        
        return concat_tvs