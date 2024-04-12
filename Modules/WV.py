#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import os
import numpy as np
import time
import random
import h5py
import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
from scipy import sparse
from joblib import dump
import matplotlib.pyplot as plt

import tensorflow.keras.backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
#from tensorflow.config import experimental
#from tensorflow.config import list_physical_devices
#gpus = list_physical_devices('GPU')

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from FUNCTIONS import get_filenames_from_folder
from FUNCTIONS import saving_acc_to_CSV
from FUNCTIONS import update_log
from FUNCTIONS import load_log_info
from FUNCTIONS import save_dic_to_json
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import error_incompatible_strings_input
from FUNCTIONS import error_print_abort_class
from FUNCTIONS import get_tag_name

from functions_TOKENS import get_tokens_from_sent

from functions_TEXTS import get_term_list_from_TXT

from functions_VECTORS import get_close_vecs
from functions_VECTORS import scale_normalize_1dvector
from functions_VECTORS import get_largest_topic_value_indexes



class WV(object):
    
    def __init__(self, wv_model = 'svd', diretorio = None):
        
        print('\n( Class: WV )')
        
        self.stopwords_list = stopwords.words('english')        
        self.diretorio = diretorio
        self.abort_class = False
        self.wv_model = wv_model.lower()
        self.class_name = 'WV'
                
        #testando os erros de inputs
        self.abort_class = error_incompatible_strings_input('wv_model', wv_model, ('svd', 'w2vec', 'gensim'), class_name = self.class_name)

        #carregando os tokens do IDF
        print('Carregando os tokens do IDF')
        self.IDF = pd.read_csv(self.diretorio + '/Outputs/tfidf/idf.csv', index_col = 0)
        self.all_tokens_array = self.IDF.index.values
        self.n_tokens = len(self.all_tokens_array)
        print('n_tokens = ', self.n_tokens)
                   
        if self.wv_model == 'svd':

            #carregando o número de sentences (=número de documentos)
            print('Model: svd')
            print('Carregando o número de documentos (sentenças)')
            self.n_sents = load_log_info(log_name = 'sentence_counter', logpath = self.diretorio + '/Outputs/log/filtered_sents.json')
            self.proc_documents = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')
            self.n_articles = len(self.proc_documents)

        elif self.wv_model == 'w2vec':
    
            #carregando os tokens do IDF
            print('Model: w2vec')
            
        elif self.wv_model == 'gensim':
            
            #carregando os tokens do IDF
            print('Model: gemsim')
            

        #checando a pasta /Outputs/wv/
        if not os.path.exists(self.diretorio + '/Outputs/wv'):
            os.makedirs(self.diretorio + '/Outputs/wv')
            
        #checando a pasta /Outputs/models/
        if not os.path.exists(self.diretorio + '/Outputs/models'):
            os.makedirs(self.diretorio + '/Outputs/models')



    def set_w2vec_parameters(self, mode = 'cbow', words_window_size = 3, subsampling = False, sub_sampling_thresold = 3e-4):
    
        print('\n( Function: set_w2vec_parameters )')    
    
        #checando erros de instanciação/inputs
        abort_class = error_incompatible_strings_input('mode', mode, ('skip-gram', 'cbow'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
    
        self.mode = mode.lower()
        self.subsampling = subsampling
        self.words_window_size = words_window_size
        self.sub_sampling_thresold = sub_sampling_thresold
        self.wv_matrix_name = f'{self.wv_model}_{self.mode}_WS_{self.words_window_size}'
        
        print('Set conditions:')
        print('mode = ', self.mode, ', words_window_size = ', self.words_window_size)



    def set_svd_parameters(self, mode = 'truncated_svd'):
    
        print('\n( Function: set_svd_parameters )')    
    
        #checando erros de instanciação/inputs
        abort_class = error_incompatible_strings_input('mode', mode, ('truncated_svd'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
    
        self.mode = mode.lower()
        self.wv_matrix_name = f'{self.wv_model}_{self.mode}'
        
        print('Set conditions:')
        print('mode = ', self.mode)

                
        
    def get_LSA_wv_matrixes(self, n_dimensions = 300, word_list_to_test = None):

        print('\n( Function: get_LSA_wv )')

        #checando erros de instanciação/inputs
        abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('svd'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
        abort_class = error_incompatible_strings_input('LSA_mode', self.mode, ('truncated_svd'), class_name = self.class_name)
        if abort_class is True:
            error_print_abort_class(self.class_name)
            return
                
        print('mode:', self.mode)
                
        if os.path.exists(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv') and os.path.exists(self.diretorio + f'/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv'):
            print(f'Já existem os svd wv df (~/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv) e (~/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv)')
            print('> Abortando função: WV.get_LSA_WV')
            
            #plotando os treinamentos
            self.plot_2DWV(words = word_list_to_test, 
                           path_to_wv_matrix = self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv', 
                           filepath_to_save = f'/Outputs/wv/PlotQTest_wv_sents_{self.wv_matrix_name}.png')
            
            #plotando os treinamentos
            self.plot_2DWV(words=word_list_to_test, 
                           path_to_wv_matrix = self.diretorio + f'/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv',
                           filepath_to_save = f'/Outputs/wv/PlotQTest_wv_articles_{self.wv_matrix_name}.png')
            
            return
        

        #montando a lista de arquivos npz com as sparse matrices
        NPZ_documents = get_filenames_from_folder(self.diretorio + '/Outputs/tfidf/tfidf_sents_npz', file_type = 'npz')

        #fiting a data
        print('Running SVD...')
        
        if (self.mode == 'truncated_svd'):
            
            model = TruncatedSVD(n_components = n_dimensions,  algorithm = 'arpack', n_iter=2000)

            #--------------------------------------------
            #gerando a token_topic matrix para as sentenças
            sparse_list = []
            for filename in NPZ_documents:
                print(f'Opening: {filename}...')                
                m = sparse.load_npz(self.diretorio + '/Outputs/tfidf/tfidf_sents_npz/' + filename + '.npz')
                sparse_list.append(m)
            
            X = sparse.vstack(sparse_list, dtype=np.float64)
            
            #normalizando e centralizando
            X.data /= X.data.max()
            X.data -= X.data.mean()

            print('Term-Sent Sparse matrix shape: ', X.T.shape)
            print('Processing Term-Sent matrix via Truncated SVD...')
            U = model.fit_transform(X.T)
            print('Fit-transformed matrix (U) shape: ', U.shape)
            
            #definindo os word vectors
            wv_df = pd.DataFrame(U, index = self.all_tokens_array)
            wv_df.to_csv(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv')
            print(f'Saving the sent_wv_svd model (~/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv)')

            #salvando o modelo
            dump(model, self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.joblib')
                
            #plotando os treinamentos
            self.plot_2DWV(words = word_list_to_test, 
                           path_to_wv_matrix = self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv', 
                           filepath_to_save = f'/Outputs/wv/PlotQTest_wv_sents_{self.wv_matrix_name}.png')


            del sparse_list
            del X
            del model
            del wv_df

            #--------------------------------------------
            #gerando a token_topic matrix para os artigos

            model = TruncatedSVD(n_components = n_dimensions,  algorithm = 'arpack', n_iter=2000)

            X = sparse.load_npz(self.diretorio + '/Outputs/tfidf/tfidf_articles_sparse_csr.npz')
            
            #normalizando e centralizando
            X.data /= X.data.max()
            X.data -= X.data.mean()

            print('Term-Article H5 matrix shape: ', X.T.shape)
            print('Processing Term-Article matrix via Truncated SVD...')
            U = model.fit_transform(X.T)
            print('Fit-transformed matrix (U) shape: ', U.shape)
            
            #definindo os word vectors
            wv_df = pd.DataFrame(U, index = self.all_tokens_array)
            wv_df.to_csv(self.diretorio + f'/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv')
            print(f'Saving the wv_svd model (~/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv)')

            #salvando o modelo
            dump(model, self.diretorio + f'/Outputs/wv/wv_articles_{self.wv_matrix_name}.joblib')
                
            #plotando os treinamentos
            self.plot_2DWV(words=word_list_to_test, 
                           path_to_wv_matrix = self.diretorio + f'/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv',
                           filepath_to_save = f'/Outputs/wv/PlotQTest_wv_articles_{self.wv_matrix_name}.png')
            
            del X
            del model
            del wv_df



    def get_LSA_sent_topic_matrix(self):

        print('\n( Function: get_LSA_sent_topic_matrix)')

        #checando erros de instanciação/inputs
        abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('svd'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return

        #abrindo o svd_wv (sent_token_topic matrix)
        svd_wv = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv', index_col = 0)        

        #carregando o dicinário com as informações sobre os batches        
        tfidf_batches_log = load_dic_from_json(self.diretorio + f'/Outputs/log/tfidf_batches_log.json')
        last_batch_to_be_processed = int(sorted(list(tfidf_batches_log.keys()))[-1])
        
        #criando a matriz sent_token_topic
        if os.path.exists(self.diretorio + f'/Outputs/models/lsa_sents_topic_full_matrix.h5'):
            #contador de arquivos
            last_batch_processed = load_log_info(log_name = 'last_batch_processed_for_sents', logpath = self.diretorio + '/Outputs/log/lsa_matrices.json')
            last_batch_processed = last_batch_processed if last_batch_processed is not None else int(sorted(list(tfidf_batches_log.keys()))[0])
            next_batch = int(last_batch_processed) + 1
            print('next TFIDF batch to process: ', next_batch)

            if last_batch_to_be_processed == last_batch_processed:
                print('A matriz sent_topics já foi calculada.')
                print('> Abortando função: WV.get_LSA_doc_topic_matrixes')
                return
            
        else:
            print('Criando a matriz sent_topics (H5 file)...')
            
            h5_sent_topic = h5py.File(self.diretorio + f'/Outputs/models/lsa_sents_topic_full_matrix.h5', 'w')
            wv_dim = len(svd_wv.columns)
            h5_sent_topic.create_dataset('data', shape=(self.n_sents, wv_dim), dtype=np.float64)
            h5_sent_topic.close()        
            
            update_log(log_names = ['last_batch_processed_for_sents'], entries = [0], logpath = self.diretorio + '/Outputs/log/lsa_matrices.json')
            next_batch = 1
            print('next TFIDF batch to process: ', next_batch)
            del h5_sent_topic

        #abringo os indices das sentenças
        sent_indexes = pd.read_csv(self.diretorio + '/Outputs/log/sents_index.csv', index_col = 0)    
                
        #varrendo os batches estabelecidos no LOG
        for batch in range(next_batch, last_batch_to_be_processed + 1):

            print('\nProcessando batch: ', batch)            

            #gerando o nome do arquivo para o batch
            c_batch_number = get_tag_name(batch, prefix = '')            

            #carregando os index do primeiro e último documentos do batch
            first_file_index, last_file_index = tfidf_batches_log[c_batch_number]['first_file_index'], tfidf_batches_log[c_batch_number]['last_file_index']
            
            #determinando os nomes do arquivos com esses file_indexes
            first_article_filename , last_article_filename = self.proc_documents[first_file_index] , self.proc_documents[last_file_index]

            #determinando o index do primeiro documento (sent) do batch
            initial_sent_index = sent_indexes.loc[first_article_filename, 'initial_sent']            
            #determinando o index do último documento (sent) do batch
            last_sent_index = sent_indexes.loc[last_article_filename, 'last_sent']

            #carregando o arquivo TFIDF.npz do batch
            m = sparse.load_npz(self.diretorio + f'/Outputs/tfidf/tfidf_sents_npz/sparse_csr_{c_batch_number}.npz')
            print('Carregando a matrix sent TFIDF do batch. Shape: ', m.shape)
            print('Carregando a matrix term_topic. Shape: ', svd_wv.values.shape)
                        
            #carregando a matriz DOC_TOPIC
            h5_sent_topic = h5py.File(self.diretorio + f'/Outputs/models/lsa_sents_topic_full_matrix.h5', 'a')
            sent_topic_m = h5_sent_topic['data']
            print('Carregando a matrix sent_topic. Shape: ', sent_topic_m.shape)
                        
            print('Filenames - first: ', first_article_filename, ' ; last: ', last_article_filename)
            print('Sent slice: ', initial_sent_index, ' a ',  last_sent_index)
            
            #Dot product das matrizes: sent_token . token_topic
            sent_topic_m[ initial_sent_index : last_sent_index + 1] = np.dot( m.todense(), svd_wv.values )
            
            #salvando o número total de sentença (documents)
            update_log(log_names = ['last_batch_processed_for_sents'], entries = [batch], logpath = self.diretorio + '/Outputs/log/lsa_matrices.json')

            h5_sent_topic.close()
            del h5_sent_topic            
            del sent_topic_m



    def get_LSA_article_topic_matrix(self):
                
        print('\n( Function: get_LSA_article_topic_matrix )')

        #checando erros de instanciação/inputs
        abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('svd'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return

        #abrindo o svd_wv (article_token_topic matrix)
        svd_wv = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv', index_col = 0)
        wv_dim = len(svd_wv.columns)

        #criando a matriz article_topic
        if os.path.exists(self.diretorio + f'/Outputs/models/lsa_articles_topic_full_matrix.h5'):
                print('A matriz article_topics já foi calculada.')
                print('> Abortando função: WV.get_LSA_doc_topic_matrixes')
                return
            
        else:
            print('Criando a matriz article_topics (H5 file)...')            
            h5_article_topic = h5py.File(self.diretorio + f'/Outputs/models/lsa_articles_topic_full_matrix.h5', 'w')
            h5_article_topic.create_dataset('data', shape=(self.n_articles, wv_dim), dtype=np.float64)
            article_topic_m = h5_article_topic['data']
            print('article_topic matrix shape: ', article_topic_m.shape)

        #carregando o arquivo TFIDF.npz dos artigos
        m = sparse.load_npz(self.diretorio + f'/Outputs/tfidf/tfidf_articles_sparse_csr.npz')
        print('Carregando a matrix article TFIDF. Shape: ', m.shape)
        print('Carregando a matrix article term_topic. Shape: ', svd_wv.values.shape)
        
        #Dot product das matrizes: article_token . token_topic
        article_topic_m[ : ] = np.dot( m.todense(), svd_wv.values )
        h5_article_topic.close()

        del h5_article_topic            
        del article_topic_m



    def get_LSA_topic_vector_stats(self):

        print('\n( Function: get_LSA_topic_vector_stats )')

        #checando se os stats da matrix doc_topic já foram encontrados
        if os.path.exists(self.diretorio + '/Outputs/models/lsa_sents_topic_matrix_stats.json') and os.path.exists(self.diretorio + '/Outputs/models/lsa_articles_topic_matrix_stats.json'):
            print('Arquivo "sent_topic_matrix_stats" encontrado em ~/Outputs/models/lsa_sents_topic_matrix_stats.json')
            print('Arquivo "article_topic_matrix_stats" encontrado em ~/Outputs/models/lsa_articles_topic_matrix_stats.json')
            print('> Abortando função: WV.get_LSA_topic_vector_stats')
            return
        else:
            sent_tv_stats = {}
            article_tv_stats = {}

        print('Finding min/max values in Doc_Topic_matrixes:')
        #carregando as matrizes DOC_TOPIC
        h5_sent_topic = h5py.File(self.diretorio + f'/Outputs/models/lsa_sents_topic_full_matrix.h5', 'r')
        sent_topic_m = h5_sent_topic['data']

        sent_tv_stats['min'] = sent_topic_m[:].min()
        sent_tv_stats['max'] = sent_topic_m[:].max()
        
        h5_sent_topic.close()

        h5_article_topic = h5py.File(self.diretorio + f'/Outputs/models/lsa_articles_topic_full_matrix.h5', 'r')
        article_topic_m = h5_article_topic['data']

        article_tv_stats['min'] = article_topic_m[:].min()
        article_tv_stats['max'] = article_topic_m[:].max()

        h5_article_topic.close()
                
        print('Salvando o sent_tv_stats em ~/Outputs/models/lsa_sents_topic_matrix_stats.json')
        save_dic_to_json(self.diretorio + '/Outputs/models/lsa_sents_topic_matrix_stats.json', sent_tv_stats)

        print('Salvando o article_tv_stats em ~/Outputs/models/lsa_articles_topic_matrix_stats.json')
        save_dic_to_json(self.diretorio + '/Outputs/models/lsa_articles_topic_matrix_stats.json', article_tv_stats)
        

        
    def get_W2Vec(self, wv_dim = 300, n_epochs = 1, batch_size = 50, article_batch_size = 10, word_list_to_test = None):

        print('\n( Function: get_W2Vec )')

        #checando erros de instanciação/inputs
        abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('w2vec'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
            

        #determinando os one-hot vectors para todas os tokens encontrados no IDF para o modo skip-gram 
        if self.mode == 'skip-gram':
            OHV_array = np.zeros([ self.n_tokens , self.n_tokens ], dtype = np.int8)
            line_counter = 0
            for i in range(self.n_tokens):
               OHV_array[ line_counter , i ] = 1
               line_counter +=1
            OHV_DF = pd.DataFrame(OHV_array, index=self.all_tokens_array)
            #print(OHV_DF)        
        
        #montando a lista de documentos já processados
        proc_documents = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv') #lista de arquivos com as textos já extraídos

        #contador de arquivos (para parar e resumir o fit)
        filename = load_log_info(log_name = 'last_filename_processed', logpath = self.diretorio + '/Outputs/log/w2vec_matrices.json')
        filename = filename if filename is not None else proc_documents[0]
        print('w2vec_counter_file_index: ', filename)
        fileindex = proc_documents.index(filename)

        if os.path.exists(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.h5'):
            model = load_model(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.h5')
            
        else:
            #criando a rede neural
            #Create Sequential model with Dense layers, using the add method
            model = Sequential()
            #Creating the hidden layers
            n_input = self.n_tokens
            n_output = n_input
            n_neurons = wv_dim
            
            model.add(Dense(units=n_neurons,
                            activation='elu',
                            kernel_initializer='he_uniform',
                            input_shape=(n_input, ),
                            name='layer1'))
                      
            model.add(Dense(units=n_output, 
                            kernel_initializer='he_uniform',
                            activation='softmax',
                            name='layer2'))
            
            #The compile method configures the model’s learning process
            opt = SGD(learning_rate=0.1, nesterov=True, momentum=0.9)
            model.compile(loss = 'categorical_crossentropy',
                          optimizer = opt,
                          metrics = ['accuracy'])
            
            #imprime o sumário do modelo
            model.summary()
            model_save_folder = self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.h5'
            model.save(model_save_folder)                    
        
        
        #contador do batch de PDFs
        article_batch_counter = 0

        #lista com todos os dados usados no treino
        data = []        
        #abrindo os arquivos com as sentenças
        for filename in proc_documents[ fileindex + 1 : ]:    
    
            #abrindo o csv com as sentenças do artigo
            sentDF = pd.read_csv(self.diretorio + '/Outputs/sents_filtered/' + f'{filename}.csv', index_col = 0)
            article_batch_counter += 1
            print(f'Processando CSV (~/Outputs/sents_filtered/{filename}.csv)')
            for index in sentDF.index:
                #print(f'Procurando tokens na sentença {file_tag}_sent{index}')
                #analisando cada sentença
                sent = sentDF.loc[index, 'Sentence']                
                                
                #splitando a sentença em tokens
                sent_tokens_filtered = get_tokens_from_sent(sent, tokens_list_to_filter = self.all_tokens_array, stopwords_list_to_remove = self.stopwords_list, spacy_tokenizer = nlp)                                                
                
                #gerando data para o treino                
                #usando a abordagem skip-gram
                if self.mode == 'skip-gram':
                    for idx, word in enumerate(sent_tokens_filtered):
                        for neighbor in sent_tokens_filtered[max(idx - self.words_window_size, 0) : min(idx + self.words_window_size + 1, len(sent_tokens_filtered))]:
                            if neighbor != word:
                                if (self.subsampling is True):
                                    term_freq = self.IDF.loc[neighbor, 'TF_TOKEN_NORM']
                                    take_probality_THR = 1 - ( self.sub_sampling_thresold / term_freq )**(1/2)                                    
                                    random_prob = random.uniform(0,1)
                                    if random_prob > take_probality_THR:
                                        #print('token-neighbor ignorados: ', word, ' ', neighbor )
                                        continue
                                    else:
                                        data.append([ OHV_DF.loc[word].values , OHV_DF.loc[neighbor].values ])
                                        #print('word: ', word, '; neighbor: ', neighbor)
                                        #time.sleep(1)                                        
                                else:
                                    data.append([ OHV_DF.loc[word].values , OHV_DF.loc[neighbor].values ])
                                    #print('word: ', word, '; neighbor: ', neighbor)
                                    #time.sleep(1)
                    
                #usando a abordagem do continuous bag of words (CBOW)
                elif self.mode == 'cbow':
                    for idx, word in enumerate(sent_tokens_filtered):                        
                        #listas para testar
                        test_list_X = []
                        test_list_Y = []
                        #definindo o vector para a janela da sentença (com zeros)
                        cbow_vec_X = np.zeros(self.n_tokens, dtype = np.int8) #TRAIN
                        cbow_vec_Y = np.zeros(self.n_tokens, dtype = np.int8) #TARGET
                        #definindo somente o index da palavra como = 1
                        word_index = np.where(self.all_tokens_array == word)[0][0]
                        cbow_vec_Y[ word_index ] = 1
                        test_list_Y.append(word)
                        #print('word: ', self.all_tokens_array[ word_index ])
                        
                        list_p1 = sent_tokens_filtered[ max(idx - self.words_window_size, 0) : idx ]
                        list_p2 = sent_tokens_filtered[ idx + 1: min(idx + self.words_window_size + 1, len(sent_tokens_filtered)) ]
                        #print('list_p1 i1: ', max(idx - self.words_window_size, 0), 'i2: ', idx )
                        #print('list_p1: ', list_p1)
                        #print('list_p2 i1: ', idx + 1, 'i2: ', min(idx + self.words_window_size, len(sent_tokens_filtered)))
                        #print('list_p2: ', list_p2)
                        #time.sleep(1)
                        word_window_list = list_p1 + list_p2
                        for neighbor in word_window_list:
                            #definindo os indeces do vizinho da palabra como sendo 1 (a palavra é = 0)
                            neighbor_index = np.where(self.all_tokens_array == neighbor)[0][0]
                            cbow_vec_X[ neighbor_index ] = 1
                            test_list_X.append(neighbor)
                            #print('neighbor: ', self.all_tokens_array[ neighbor_index ])
                            
                        data.append([ cbow_vec_X , cbow_vec_Y ])                                
                        #time.sleep(1)
                        #print('palavra target: ', test_list_Y, '; neighbors: ', test_list_X)
                                                
            if article_batch_counter % article_batch_size == 0 or filename == proc_documents[-1]:
                            
                #carregando a rede neural
                model = load_model(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.h5')
                
                #The fit method does the training in batches
                # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
                # data[: , 0] = X ; data[: , 1] = Y
                data = np.array(data)
                #print(data[: , 0].shape, data[: , 1].shape)
                
                #Treinando
                history = model.fit(x = data[: , 0], y = data[: , 1], epochs=n_epochs, batch_size=batch_size)
                
                #caso tenha havido treino
                if history is not None:
                
                    #salvando o modelo h5            
                    model_save_folder = self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.h5'
                    model.save(model_save_folder)
                    
                    #carregando o histórico
                    avg_acc = round( history.history['accuracy'][0] , 2)
                        
                    saving_acc_to_CSV(last_article_file = filename, 
                                      settings = f'wv_sents_{self.wv_matrix_name}', 
                                      acc = avg_acc, 
                                      folder = '/Outputs/wv/',
                                      diretorio = self.diretorio)
                                
                    #salvando os weights em .csv
                    l1_weights = model.get_layer(name='layer1').get_weights()[0] #0 para os weights e 1 para os bias
                    #print(l1_weights.shape, self.n_tokens)
                    wv_df = pd.DataFrame(l1_weights, index = self.all_tokens_array)
                    wv_df.to_csv(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv')
                
                    #salvando o número do último arquivo processado
                    update_log(log_names = ['last_filename_processed'], entries = [filename], logpath = self.diretorio + '/Outputs/log/w2vec_matrices.json')

                    del history
                    del wv_df
                
                #apagando os dados do treino
                del model
                data = []
                
                #deletando o graph da última sessão
                backend.clear_session()
        
        #plotando os treinamentos
        self.plot_2DWV(words=word_list_to_test,
                        path_to_wv_matrix = self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv',
                        filepath_to_save = f'/Outputs/wv/PlotQTest_wv_sents_{self.wv_matrix_name}.png')
                
        

    def get_wv_stats(self):

        print('\n( Function: get_matrix_vector_stats )')
        
        #checando erros de instanciação/inputs
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        
        if os.path.exists(self.diretorio + '/Outputs/wv/wv_stats.json'):
            print('Arquivo matrix_vectors_stats encontrado em ~/Outputs/wv/wv_stats.json')
            wv_stats = load_dic_from_json(self.diretorio + '/Outputs/wv/wv_stats.json')
        else:
            wv_stats = {}        
       
        if self.wv_model == 'w2vec':
            
            sent_wv_matrix = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv', index_col = 0)            
            
            try:
                if self.wv_matrix_name in wv_stats[self.wv_model].keys():
                    print(f'Stats já encontrada para o WV wv_sents_{self.wv_matrix_name}')
                    print('> Abortando função: WV.get_matrix_vector_stats')
                    return
            except KeyError:
                pass
            
            try:
                wv_stats[self.wv_model]
            except KeyError:
                wv_stats[self.wv_model] = {}

            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}'] = {}
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['min'] = round(np.min(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['max'] = round(np.max(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['avg'] = round(np.mean(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['std'] = round(np.std(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['median'] = round(np.median(sent_wv_matrix.values), 10)

            print('Salvando o wv_stats em ~/Outputs/wv/wv_stats.json')
            save_dic_to_json(self.diretorio + '/Outputs/wv/wv_stats.json', wv_stats)
            
        
        elif self.wv_model == 'svd':
            
            sent_wv_matrix = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv', index_col = 0)
            article_wv_matrix = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_articles_{self.wv_matrix_name}.csv', index_col = 0)
            
            try:
                if f'wv_sents_{self.wv_matrix_name}.csv' in wv_stats[self.wv_model].keys() and f'wv_articles_{self.wv_matrix_name}.csv' in wv_stats[self.wv_model].keys():
                    print(f'Stats já encontrada para o WV wv_sents_{self.wv_matrix_name} e para o WV wv_articles_{self.wv_matrix_name}.')
                    print('> Abortando função: WV.get_matrix_vector_stats')
                    return
            except KeyError:
                pass
        
            try:
                wv_stats[self.wv_model]
            except KeyError:
                wv_stats[self.wv_model] = {}
        
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}'] = {}
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['min'] = round(np.min(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['max'] = round(np.max(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['avg'] = round(np.mean(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['std'] = round(np.std(sent_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_sents_{self.wv_matrix_name}']['median'] = round(np.median(sent_wv_matrix.values), 10)

            wv_stats[self.wv_model][f'wv_articles_{self.wv_matrix_name}'] = {}
            wv_stats[self.wv_model][f'wv_articles_{self.wv_matrix_name}']['min'] = round(np.min(article_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_articles_{self.wv_matrix_name}']['max'] = round(np.max(article_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_articles_{self.wv_matrix_name}']['avg'] = round(np.mean(article_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_articles_{self.wv_matrix_name}']['std'] = round(np.std(article_wv_matrix.values), 10)
            wv_stats[self.wv_model][f'wv_articles_{self.wv_matrix_name}']['median'] = round(np.median(article_wv_matrix.values), 10)

            print('Salvando o wv_stats em ~/Outputs/wv/wv_stats.json')
            save_dic_to_json(self.diretorio + '/Outputs/wv/wv_stats.json', wv_stats)



    def find_terms_sem_similarity(self, n_similar_terms_to_overlap = 30):

        print('\n( Function: find_terms_sem_similarity )')
        
        #checando erros de instanciação/inputs
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return

        if not os.path.exists(self.diretorio + '/Outputs/ngrams/semantic'):
            os.makedirs(self.diretorio + '/Outputs/ngrams/semantic')

        #carregando o modelo WV
        if self.wv_model == 'gensim':
            import gensim.downloader as api
            wv = api.load('word2vec-google-news-300')
        else:
            wv = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv', index_col = 0)
            print('Carregando: ', f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv')

        #coletando os n1grams determinados com o term_semantic_similarity        
        filenames = get_filenames_from_folder(self.diretorio + '/Settings', file_type = 'txt')
        filtered_filenames = [filename for filename in filenames if filename[ : len('input_topic_') ] == 'input_topic_']
    
        for filename in filtered_filenames:
            
            #carregando a lista de termos a serem pesquisados
            terms_to_find_similatiry = get_term_list_from_TXT(self.diretorio + f'/Settings/{filename}.txt')
            #definindo o nome do tópico (semantic meaning)
            input_topic  = filename[ len('input_topic_') : ]
            
            #varrendo cada termo
            for term in terms_to_find_similatiry:
                
                print('\nSemantic meaning: ', input_topic)
                print(f'Finding similar terms for * {term} *')
        
                #checar se o termo existe
                try:
                    word_vec = wv.loc[term].values
                except KeyError:
                    print('Um dos termos inseridos não está presente na DF de Word Vectors.')
                    print('Termo: ', term)
                    continue
        
                #calculando a semelhança pela similaridade de cosseno
                indexes_closest_vecs = get_close_vecs(word_vec,
                                                      wv.values, 
                                                      first_index = 0 , 
                                                      n_close_vecs = n_similar_terms_to_overlap)
                
                #calculando a semelhança pela distância entre os dois vetores
                #indexes_closest_vecs = get_neighbor_vecs(word_vec,
                #                                         wv.values,
                #                                         first_index = 0,
                #                                         n_close_vecs = n_similar_terms_to_overlap)
        
                #coletando os termos próximos
                similar_terms = []
                for i in indexes_closest_vecs:
                    token = self.all_tokens_array[i]
                    similar_terms.append(token)
                    #print(token)
                    
                #criando a DF de termos (n1grams) similares
                if not os.path.exists(self.diretorio + f'/Outputs/ngrams/semantic/n1gram_{input_topic}.csv'):
                    n1gram_sim_terms_DF = pd.DataFrame(columns=['Sem_App_Counter', 'sim_check'], index=wv.index.values)            
                    n1gram_sim_terms_DF.index.name = 'index'
                    #setando as condições inicias da DF
                    for token in n1gram_sim_terms_DF.index:
                        n1gram_sim_terms_DF.loc[token] = ( 0 , False )
                    n1gram_sim_terms_DF.to_csv(self.diretorio + f'/Outputs/ngrams/semantic/n1gram_{input_topic}.csv')
                    print(f'Criando a DF ~/Outputs/ngrams/semantic/n1gram_{input_topic}.csv')
                    
                #carregando a DF de termos (n1grams) similares
                else:
                    n1gram_sim_terms_DF = pd.read_csv(self.diretorio + f'/Outputs/ngrams/semantic/n1gram_{input_topic}.csv', index_col=0)
                    n1gram_sim_terms_DF.index.name = 'index'
                        
                #adicionando os termos similares ao DF
                if n1gram_sim_terms_DF.loc[ term, 'sim_check' ] == False:
                    #mudando a condição do termo da procura
                    n1gram_sim_terms_DF.loc[ term, 'Sem_App_Counter' ] = int(n1gram_sim_terms_DF.loc[ term, 'Sem_App_Counter' ]) + 1
                    n1gram_sim_terms_DF.loc[ term, 'sim_check' ] = True
                    #mudando a condição dos tokens encontrados por similaridade semântica
                    for token in similar_terms:
                        n1gram_sim_terms_DF.loc[ token, 'Sem_App_Counter' ] = int(n1gram_sim_terms_DF.loc[ token, 'Sem_App_Counter' ]) + 1
                else:
                    print('Termo já teve a similaridade semântica procurada.')
                    print('Termo: ', term)
                    continue
                
                #listando os tokens mais encontrados
                n1gram_sim_terms_DF.sort_values(by = ['Sem_App_Counter'], ascending = False, inplace = True)
                n1gram_sim_terms_DF.index.name = 'index'
                #print(n1gram_sim_terms_DF.head(10))
                n1gram_sim_terms_DF.to_csv(self.diretorio + f'/Outputs/ngrams/semantic/n1gram_{input_topic}.csv')                        
                
            
        #consolidando os n2grams_DF
        for filename in filtered_filenames:
            #definindo o nome do tópico (semantic meaning)
            input_topic  = filename[ len('input_topic_') : ]
            print('\nSemantic meaning: ', input_topic)
            print('Finding similar n2grams terms')
            
            #carregando o n1gram_DF
            n1gram_sim_terms_DF = pd.read_csv(self.diretorio + f'/Outputs/ngrams/semantic/n1gram_{input_topic}.csv', index_col=0)
            n1gram_sim_terms_DF.index.name = 'index'
            
            #colocando os bigrams similares ao termo em uma DF
            if not os.path.exists(self.diretorio + f'/Outputs/ngrams/semantic/n2gram_{input_topic}.csv'):
                n2gram_sim_terms_DF = pd.DataFrame(columns=['token_1', 'token_2', 'total', 'sent','article'])
                n2gram_sim_terms_DF.index.name = 'index'
                print(f'Criando a DF ~/Outputs/ngrams/semantic/n2gram_{input_topic}.csv')
                
            else:
                print(f'CSV file ~/Outputs/ngrams/semantic/n2gram_{input_topic}.csv encontrado...')
                print(f'Abortando a função para o tópico {input_topic}...')
                return
                #n2gram_sim_terms_DF = pd.read_csv(self.diretorio + f'/Outputs/ngrams/semantic/n2gram_{input_topic}.csv', index_col=0)
                #n2gram_sim_terms_DF.index.name = 'index'    
                        
            #carregando o n2gram DF filtrada (por score e delta de contagem)
            n2grams_filtered = pd.read_csv(self.diretorio + '/Outputs/ngrams/filtered_scores/n2grams_filtered.csv', index_col = 0)
            
            #encontrando as combinações de tokens (n2grams)
            for token in n1gram_sim_terms_DF.index:
                if n1gram_sim_terms_DF.loc[token, 'Sem_App_Counter'] >= 1:
                    for neighbor in n1gram_sim_terms_DF.index:
                        if n1gram_sim_terms_DF.loc[neighbor, 'Sem_App_Counter'] >= 1:
                            if token != neighbor:
                                #só a cond 4 está sendo usada
                                if token + '_' + neighbor in n2grams_filtered.index:
                                    try:                                        
                                        #essa DF é para análise e plotagem dos resultados
                                        n2gram_sim_terms_DF.loc[token + '_' + neighbor] = (token, 
                                                                                          neighbor ,                                                                                     
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'total' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'sent' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'article' ])
                                    except KeyError:
                                        continue
                                    
            #salvando as DF
            n2gram_sim_terms_DF.sort_values(by=['article'], ascending=False, inplace=True)
            n2gram_sim_terms_DF.to_csv(self.diretorio + f'/Outputs/ngrams/semantic/n2gram_{input_topic}.csv')
    
        del n1gram_sim_terms_DF
        del n2grams_filtered
        del n2gram_sim_terms_DF



    def combine_topic_vectors_by_sem_similarity(self, n_largest_topic_vals = 30, min_sem_app_count_to_get_topic = 4):

        print('\n( Function: combine_topic_vectors_by_sem_similarity )')        
        
        #checando erros de instanciação/inputs
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        if self.wv_model != 'svd' or self.mode != 'truncated_svd':
            print('Erro! A função precisa ser usada com wv_model = svd e mode = truncated_svd.')
            print('> Abortando função: WV.combine_topic_vectors_by_sem_similarity')
            return

        #carregando o modelo WV
        wv = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_{self.wv_matrix_name}.csv', index_col = 0)
        #carregando o wv_stats
        #wv_stats_dic = load_dic_from_json(self.diretorio + '/Outputs/wv/wv_stats.json')
        #wv_stats = wv_stats_dic[f'{self.wv_model}'][f'wv_sents_{self.wv_matrix_name}']
        #print('wv_stats usada: ', f'{self.wv_model} ', f'wv_sents_{self.wv_matrix_name}')

        #coletando os n1grams determinados com o term_semantic_similarity        
        filenames = get_filenames_from_folder(self.diretorio + '/Outputs/ngrams/semantic', file_type = 'csv')
        filtered_filenames = [filename for filename in filenames if (filename[ : 6 ] == 'n1gram') and (filename[ -len('regex') : ] != 'regex')]
                    
        for filename in filtered_filenames:
            #carregando a lista de termos a serem pesquisados
            n1gram_DF = pd.read_csv(self.diretorio + f'/Outputs/ngrams/semantic/{filename}.csv', index_col=0)
            words_to_find_topics = [index for index in n1gram_DF.index if n1gram_DF.loc[index, 'Sem_App_Counter'] >= min_sem_app_count_to_get_topic]

            #determinando o signifado semântico desses termos
            semantic_meaning = filename[ len('n1gram_') : ]
        
            #array para concatenar todos os índices    
            larg_val_index_concat = np.array([])    
        
            #gerando um vetor para colocar as somas e overlap
            wv_dim = len(wv.columns)
            sum_topic_vector = np.zeros(wv_dim)
            overlap_topic_vector = np.zeros(wv_dim)
        
            #caso haja pelo menos um termo para encontrar os topic vectors
            if len(words_to_find_topics) > 1:
                #varrendo cada termo
                for term in words_to_find_topics:
                    
                    print('Semantic meaning: ', semantic_meaning)
                    print(f'Combining topic vectors for * {term} *')        
                    
                    try:
                        #carregando o topic_vector para a sentença inserida
                        topic_vec = wv.loc[term].values
                    except KeyError:
                        continue
                    
                    #somando os vetores tópicos
                    sum_topic_vector = sum_topic_vector + topic_vec
                    
                    #determinando os  indexes do topicos com maiores valores
                    larg_topic_val_index = get_largest_topic_value_indexes(topic_vec, n_topics = n_largest_topic_vals, get_only_positive=True)
                    #concatenando os resultados no axis = 1 para cada termo presente no n1gram
                    larg_val_index_concat = np.r_[larg_val_index_concat , larg_topic_val_index ]
    
                #construindo os vetores de probabilidade
                #contando os indexes para o vector overlap de máximos valores de tópico
                unique_indexes, unique_vals  = np.unique(larg_val_index_concat.astype(int), return_counts=True)
                overlap_topic_vector[unique_indexes] = unique_vals
                            
                #normalizando os vetores
                overlap_topic_prob_vector = scale_normalize_1dvector(overlap_topic_vector)
                sum_topic_prob_vector = scale_normalize_1dvector(sum_topic_vector)
                
                fig, axs = plt.subplots(1, 2, tight_layout=True)
                #plotando os overlaps dos vetores
                axs[0].bar(np.arange(len(overlap_topic_prob_vector)), overlap_topic_prob_vector)
                axs[0].set_xlabel('Topic index')
                axs[0].set_ylabel('Overlap occurrenes')                
                
                #plotando a soma dos vetores
                axs[1].bar(np.arange(len(sum_topic_prob_vector)), sum_topic_prob_vector)
                axs[1].set_xlabel('Topic index')
                axs[1].set_ylabel('Topic values sum')   
       
                plt.savefig(self.diretorio + f'/Outputs/models/LSA_topic_{semantic_meaning}.png', dpi=200)
                
                #salvando o dic em json    
                if os.path.exists(self.diretorio + '/Outputs/models/sem_sent_topic_vectors.json'):
                    dic = load_dic_from_json(self.diretorio + '/Outputs/models/sem_sent_topic_vectors.json')
                    dic[semantic_meaning] = {}
                    dic[semantic_meaning]['topic_overlap'] = list(overlap_topic_prob_vector)
                    dic[semantic_meaning]['topic_sum'] = list(sum_topic_prob_vector)
                    save_dic_to_json(self.diretorio + '/Outputs/models/sem_sent_topic_vectors.json', dic)
                else:
                    dic = {}
                    dic[semantic_meaning] = {}
                    dic[semantic_meaning]['topic_overlap'] = list(overlap_topic_prob_vector)
                    dic[semantic_meaning]['topic_sum'] = list(sum_topic_prob_vector)
                    save_dic_to_json(self.diretorio + '/Outputs/models/sem_sent_topic_vectors.json', dic)
                    
                #avaliando a semelhança entre os tópicos
                #self.plot_2DWV(words=words_to_find_topics,
                #               matrix_name = self.wv_matrix_name, filepath_to_save = f'/Outputs/models/Topics_{semantic_meaning}.png')

            else:
                print('ERRO na combinação de Topic vectors...')
                print(f'Os termos em "{filename}" com valores "Sem_App_Counter" abaixo do threshold ("min_sem_app_count_to_get_topic" {min_sem_app_count_to_get_topic})')
                time.sleep(3)



    def plot_2DWV(self, words=['aaa', 'bbb', 'ccc', 'ddd'], path_to_wv_matrix = None, filepath_to_save = None):

        print('\n( Function: plot_2DWV )')
                
        #checando erros de instanciação/inputs
        if self.abort_class is True:
            error_print_abort_class(self.class_name)
            return

        if self.wv_model == 'gensim':
            import gensim.downloader as api
            wv = api.load('word2vec-google-news-300')
        else:
            #carregando o modelo WV
            wv = pd.read_csv(path_to_wv_matrix, index_col = 0)
        
        #definindo os WVs para as palavras desejadas
        labels = []
        wordvecs = []
        for word in words:
            try:
                if self.wv_model == 'gensim':
                    wordvecs.append(wv[word])
                    labels.append(word)                
                else:
                    wordvecs.append(wv.loc[word])
                    labels.append(word)
            
            except KeyError:
                print('Um dos termos inseridos não está presente na DF de Word Vectors.')
                print('Termo: ', word)
                print('> Abortando função: WV.plot_2DWV')
                return
                    
                
        #definindo a função para plotagem
        tsne_model = TSNE(perplexity=3, n_components=2, init='pca', random_state=42)
        coordinates = tsne_model.fit_transform(wordvecs)
        
        x = []
        y = []        
        for values in coordinates:
            x.append(values[0])
            y.append(values[1])
        
        plt.figure(figsize=(8,8))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy = (x[i], y[i]),
                         xytext = (2,2),
                         textcoords = 'offset points',
                         ha = 'right',
                         va = 'bottom')

        plt.savefig(self.diretorio + f'{filepath_to_save}', dpi=200)
        #plt.show()

        del wv