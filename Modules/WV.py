#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class WV(object):
    
    def __init__(self, wv_model = 'svd', replace_Ngrams = False, diretorio = None):
        
        import os
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import load_index_from_TXT
        import pandas as pd
        
        self.diretorio = diretorio
        self.abort_class = False
        self.wv_model = wv_model.lower()
        self.replace_Ngrams = replace_Ngrams
        self.class_name = 'WV'
        import spacy
        self.tokenizer_func = spacy.load('en_core_web_sm')
        
        print('\n=============================\nWORD VECTOR...\n=============================\n')
        print('( Class: WV )')
                
        #testando os erros de inputs
        from FUNCTIONS import error_incompatible_strings_input        
        self.abort_class = error_incompatible_strings_input('wv_model', wv_model, ('svd', 'w2vec', 'gensim'), class_name = self.class_name)

        #carregando os tokens do IDF
        print('Carregando os tokens do IDF')
        self.IDF = pd.read_csv(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv', index_col = 0)
        self.tokens_list = self.IDF.index.values
        self.n_tokens = len(self.tokens_list)
        print('n_tokens = ', self.n_tokens)

        #caso o replace nGrams esteja ativado
        self.n2grams_DF = None
        if replace_Ngrams is True:
            #carregando o DF de bigrams    
            self.n2grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')
            self.n2grams_DF.dropna(inplace=True)
            self.n2grams_DF.set_index('index', inplace=True)
                   
        if self.wv_model == 'svd':

            #carregando o número de sentences (=número de documentos)
            print('Model: svd')
            print('Carregando o número de documentos (sentenças)')
            self.n_documents = load_index_from_TXT('Sentence_counts', '/Outputs/LOG', diretorio = self.diretorio) #número de documentos
            self.proc_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', file_type = 'csv')

        elif self.wv_model == 'w2vec':
    
            #carregando os tokens do IDF
            print('Model: w2vec')
            
        elif self.wv_model == 'gensim':
            
            #carregando os tokens do IDF
            print('Model: gemsim')
            

        #checando a pasta /Outputs/WV/
        if not os.path.exists(self.diretorio + '/Outputs/WV'):
            os.makedirs(self.diretorio + '/Outputs/WV')
            
        #checando a pasta /Outputs/Models/
        if not os.path.exists(self.diretorio + '/Outputs/Models'):
            os.makedirs(self.diretorio + '/Outputs/Models')



    def set_w2vec_parameters(self, mode = 'cbow', filter_stopwords = False, subsampling = True, words_window_size = 4, sub_sampling_thresold = 3e-4):
    
        print('( Function: set_w2vec_parameters )')    
    
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import error_incompatible_strings_input
        abort_class = error_incompatible_strings_input('mode', mode, ('skip-gram', 'cbow'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
    
        self.mode = mode.lower()
        self.filter_stopwords = filter_stopwords
        self.subsampling = subsampling
        self.words_window_size = words_window_size
        self.sub_sampling_thresold = sub_sampling_thresold
        self.matrix_name = f'{self.wv_model}_{self.mode}_SW_{self.filter_stopwords}_WS_{self.words_window_size}_rNgram_{self.replace_Ngrams}'
        
        print('Set conditions:')
        print('mode = ', self.mode,
              ', replace Ngrams = ', self.replace_Ngrams,
              ', filter_stopwords = ', self.filter_stopwords, 
              ', words_window_size = ', self.words_window_size)



    def set_svd_parameters(self, mode = 'truncated_svd', filter_stopwords = False):
    
        print('( Function: set_svd_parameters )')    
    
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import error_incompatible_strings_input
        abort_class = error_incompatible_strings_input('mode', mode, ('pca', 'truncated_svd'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
    
        self.mode = mode.lower()
        self.filter_stopwords = filter_stopwords
        self.matrix_name = f'{self.wv_model}_{self.mode}_rNgram_{self.replace_Ngrams}'
        
        print('Set conditions:')
        print('mode = ', self.mode,
              ', replace Ngrams = ', self.replace_Ngrams,
              ', filter_stopwords = ', self.filter_stopwords)                

                
        
    def get_LSA_wv(self, n_dimensions = 300):

        print('( Function: get_LSA_WV )')

        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import error_incompatible_strings_input
        abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('svd'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
        abort_class = error_incompatible_strings_input('LSA_mode', self.mode, ('truncated_svd', 'pca'), class_name = self.class_name)
        if abort_class is True:
            error_print_abort_class(self.class_name)
            return
                
        import numpy as np
        import pandas as pd
        from scipy import sparse
        from sklearn.decomposition import PCA, TruncatedSVD
        from scipy.linalg import svd
        import json
        from joblib import dump, load
        import os
                
        print('mode:', self.mode)
                
        if os.path.exists(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv'):
            print(f'Já existe um SVD WV (~/Outputs/WV/{self.matrix_name}.csv)')
            print('> Abortando função: WV.get_LSA_WV')
            return
        
        #montando a lista de arquivos npz com as sparse matrices
        file_list = os.listdir(self.diretorio + '/Outputs/TFIDF/npz') #lista de arquivos
        NPZ_documents = []
        for filename in file_list:
            if filename[ -3 :  ].lower() == 'npz':
                NPZ_documents.append(filename)
        NPZ_documents = sorted(NPZ_documents)

        #fiting a data
        print('Running SVD...')            
        
        if self.mode == 'pca':
            model = PCA(n_components = n_dimensions, svd_solver='arpack')
        elif self.mode == 'truncated_svd':
            model = TruncatedSVD(n_components = n_dimensions,  algorithm = 'arpack', n_iter=2000)
        
        
        if (self.mode == 'truncated_svd'):
            sparse_list = []
            for filename in [filename for filename in NPZ_documents if (f'_rNgram_{self.replace_Ngrams}' in filename)]:
                print(f'Opening: {filename}...')                
                m = sparse.load_npz(self.diretorio + '/Outputs/TFIDF/npz/' + filename)
                sparse_list.append(m)
            
            M = sparse.vstack(sparse_list, dtype=np.float64)

            #fazendo o scaling da matriz
            M /= M.max()
            
            #centralizando a matriz
            M.data -= M.mean()
            
            print('Term-Document Sparse matrix shape: ', M.T.shape)
            print('Processing Term-Document matrix via Truncated SVD...')
            U = model.fit_transform(M.T)
            #definindo os word vectors
            wv_df = pd.DataFrame(U, index = self.tokens_list)
            wv_df.to_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv')
            print(f'Saving the wv_svd model (~/Outputs/WV/{self.matrix_name}.csv)')

            #salvando o modelo
            dump(model, self.diretorio + f'/Outputs/WV/{self.matrix_name}.joblib')
                
            #plotando os treinamentos
            self.plot_2DWV(words=['carbon', 'chicken', 'coconut', 'shell', 'cattle', 'manure',
                                  'sugarcane', 'bagasse', 'graphene', 'oxide', 'go'],                                       
                                   plot_batch = 0)
            
            del sparse_list
            del M
            del model
            del wv_df
                
        else:                
            batch_counter = 0
            for filename in [filename for filename in NPZ_documents if (f'_rNgram_{self.replace_Ngrams}' in filename)]:
                print(f'Opening: {filename}...')
                M = sparse.load_npz(self.diretorio + '/Outputs/TFIDF/npz/' + filename)
                if self.mode == 'pca':
                    print('Coverting SPARSE to np.array...')
                    M = M.toarray()
                    print('Term-Document Matrix Shape: ', M.T.shape)
                    print('Processing Term-Document matrix via PCA...')
                    U = model.fit_transform(M.T)
    
                #definindo os word vectors
                wv_df = pd.DataFrame(U, index = self.tokens_list)
                wv_df.to_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv')
    
                #salvando o modelo
                dump(model, self.diretorio + f'/Outputs/WV/model_{self.matrix_name}.joblib')
                    
                #plotando os treinamentos
                self.plot_2DWV(words=['antioxidant', 'anticancer', 'antibacterial', 'leishmania', 'cancer', 'heart', 'liver',
                                       'heterocyclic', 'mutans','candida', 'albicans', 'streptococcus'],
                               plot_batch = batch_counter)
                batch_counter += 1
                
                del M
                del model
                del wv_df



    def get_LSA_doc_topic_matrix(self):

        print('( Function: get_LSA_doc_topic_matrix )')

        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import error_incompatible_strings_input
        abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('svd'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return
        
        from FUNCTIONS import save_dic_to_json
        from FUNCTIONS import save_index_to_TXT
        from FUNCTIONS import load_index_from_TXT
        import numpy as np
        import h5py
        import pandas as pd
        import os
        import sys

        #abrindo o svd_wv (TERM_TOPIC matrix)
        svd_wv = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv', index_col = 0)
        
        #criando a matriz DOC_TOPIC
        if os.path.exists(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5'):
            #contador de arquivos
            file_name = load_index_from_TXT(f'counter_File_index_DOC_TOPIC_rNgram_{self.replace_Ngrams}', '/Outputs/LOG', diretorio = self.diretorio)
            print('DOC_TOPIC_counter_File_name: ', file_name)
            file_index = self.proc_documents.index(file_name) + 1
            
        else:         
            print('Criando a matriz DOCUMENTS_TOPICS (H5 file)...')
            h5 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'w')
            wv_dim = len(svd_wv.columns)
            h5.create_dataset('data', shape=(self.n_documents, wv_dim), dtype=np.float64)
            h5.close()
            file_name = 'PDF00000'
            print('DOC_TOPIC_counter_File_name: ', file_name)
            file_index = 0
            del h5 
        
        if self.proc_documents[-1] == file_name:
            print('A matriz DOCUMENTS_TOPIC já foi calculada.')
            print('> Abortando função: WV.get_LSA_doc_topic_matrix')
            return
        
        #abringo os indices das sentenças
        sent_indexes = pd.read_csv(self.diretorio + '/Outputs/LOG/SENT_INDEX.csv', index_col = 0)    
        
        counter = 1
        for pdf_filename in sent_indexes.index.values[ file_index : ]:
            
            initial_sent = sent_indexes.loc[pdf_filename, 'initial_sent']
            last_sent = sent_indexes.loc[pdf_filename, 'last_sent']
            
            print('Processando: ', pdf_filename)
            print('Processando indexes: ', initial_sent, ' : ', last_sent)
            
            #carregando a matriz DOC_TERM (matriz TFIDF)
            h5_1 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5', 'r')
            TFIDF_matrix = h5_1['data']
            TFIDF_slice = TFIDF_matrix[ initial_sent : last_sent + 1 ]
                        
            #Dot product das matrizes: DOC_TERM . TERM_TOPIC
            document_topic_m_slice = np.dot(TFIDF_slice, svd_wv.values)
            
            #carregando a matriz DOC_TOPIC
            h5_2 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'a')
            document_topic_m = h5_2['data']
            
            #salvando o slice da matrix DOC_TOPIC
            document_topic_m[ initial_sent : last_sent + 1 ] = document_topic_m_slice        

            #print('TFIDF M.SHAPE: ', TFIDF_matrix.shape)
            #print('DOC_TOPIX M.SHAPE: ', document_topic_m.shape)
            #print('WORD_TOPIC M.SHAPE: ', svd_wv.shape)
            
            #salvando o número total de sentença (documents)
            save_index_to_TXT(pdf_filename, f'counter_File_index_DOC_TOPIC_rNgram_{self.replace_Ngrams}', '/Outputs/LOG', diretorio = self.diretorio)

            
            if pdf_filename == sent_indexes.index.values[-1]:
                LOG_DIC = {}
                LOG_DIC['status'] = 'finished'
                save_dic_to_json(self.diretorio + f'/Outputs/LOG/DOC_TOPIC_log_rNgram_{self.replace_Ngrams}.json', LOG_DIC)            
                
                h5_1.close()
                h5_2.close()
                
                del h5_1
                del h5_2
                del TFIDF_matrix
                del TFIDF_slice
                del document_topic_m
                del document_topic_m_slice
                del svd_wv
                
            elif counter % 200 == 0:
                #print('Doc_Topic matrix process:', pdf_filename)
                h5_1.close()
                h5_2.close()
                
                del h5_1
                del h5_2
                del TFIDF_matrix
                del TFIDF_slice
                del document_topic_m
                del document_topic_m_slice
                #sys.exit()
            
            counter += 1                    



    def get_LSA_topic_vector_stats(self):

        import os
        import h5py
        from FUNCTIONS import save_dic_to_json

        print('( Function: get_LSA_topic_vector_stats )')

        #checando se os stats da matrix doc_topic já foram encontrados
        if os.path.exists(self.diretorio + '/Outputs/TFIDF/H5/DOC_TOPIC_MATRIX_STATS.json'):
            print('Arquivo DOC_TOPIC_MATRIX_STATS encontrado em ~/Outputs/TFIDF/H5/DOC_TOPIC_MATRIX_STATS.json')
            print('> Abortando função: WV.get_LSA_topic_vector_stats')
            return
        else:
            tv_stats = {}

        #carregando a matriz DOC_TOPIC
        h5 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'r')
        document_topic_m = h5['data']

        print('Finding min/max values in Doc_Topic_matrix:')
        tv_stats['min'] = document_topic_m[:].min()
        tv_stats['max'] = document_topic_m[:].max()
        
        
        print('Salvando o tv_stats em ~/Outputs/TFIDF/H5/DOC_TOPIC_MATRIX_STATS.json')
        save_dic_to_json(self.diretorio + '/Outputs/TFIDF/H5/DOC_TOPIC_MATRIX_STATS.json', tv_stats)
        
        h5.close()
        
    
        
    def get_W2Vec(self, wv_dim = 300, n_epochs = 5, batch_size = 50, pdf_batch_size = 10):

        print('( Function: get_W2Vec )')

        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import error_incompatible_strings_input
        abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('w2vec'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return

        from nltk.corpus import stopwords
        stopwords_list = stopwords.words('english')
        import keras.backend as backend
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.models import load_model
        from keras.optimizers import SGD
        from FUNCTIONS import load_index_from_TXT
        from FUNCTIONS import save_index_to_TXT
        from FUNCTIONS import saving_acc_to_CSV
        from FUNCTIONS import get_filenames_from_folder
        from functions_TEXTS import filter_tokens
        from functions_TOKENS import get_tokens_from_sent
        from functions_TOKENS import replace_Ngrams_in_tokens
        import pandas as pd
        import numpy as np
        import time
        import random
        import os    
                        
        #determinando os one-hot vectors para todas os tokens encontrados no IDF para o modo skip-gram 
        if self.mode == 'skip-gram':
            OHV_array = np.zeros([ self.n_tokens , self.n_tokens ], dtype = np.int8)
            line_counter = 0
            for i in range(self.n_tokens):
               OHV_array[ line_counter , i ] = 1
               line_counter +=1
            OHV_DF = pd.DataFrame(OHV_array, index=self.tokens_list)
            #print(OHV_DF)        
        
        #montando a lista de documentos já processados
        proc_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', file_type = 'csv') #lista de arquivos com as textos já extraídos

        #contador de arquivos (para parar e resumir o fit)
        try:
            file_name = load_index_from_TXT(f'counter_File_index_{self.matrix_name}', '/Outputs/LOG', diretorio = self.diretorio)
            print('W2Vec_counter_File_index: ', file_name)
            file_index = proc_documents.index(file_name) + 1
        except FileNotFoundError:
            file_index = 0

        if os.path.exists(self.diretorio + f'/Outputs/WV/{self.matrix_name}.h5'):
            model = load_model(self.diretorio + f'/Outputs/WV/{self.matrix_name}.h5')
            
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
            model_save_folder = self.diretorio + f'/Outputs/WV/{self.matrix_name}.h5'
            model.save(model_save_folder)                    
        
        
        #contador do batch de PDFs
        pdf_batch_counter = 0

        #lista com todos os dados usados no treino
        data = []        
        #abrindo os arquivos com as sentenças
        for file_tag in proc_documents[ file_index : ]:    
    
            #abrindo o csv com as sentenças do artigo
            sentDF = pd.read_csv(self.diretorio + '/Outputs/Sentences_filtered/' + f'{file_tag}.csv', index_col = 0)
            pdf_batch_counter += 1
            print(f'Processando CSV (~/Outputs/Sentences_filtered/{file_tag}.csv)')
            for index in sentDF.index:
                #print(f'Procurando tokens na sentença {file_tag}_sent{index}')
                #analisando cada sentença
                sent = sentDF.loc[index, 'Sentence']                
                                
                #splitando a sentença em tokens
                sent_tokens_filtered = get_tokens_from_sent(sent,
                                                            tokenizer_func = self.tokenizer_func, 
                                                            tokens_list = self.tokens_list, 
                                                            filter_stopwords = self.filter_stopwords, 
                                                            replace_Ngrams = self.replace_Ngrams, 
                                                            stopwords_list = stopwords_list, 
                                                            n2grams_DF = self.n2grams_DF)                                                
                
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
                        word_index = np.where(self.tokens_list == word)[0][0]
                        cbow_vec_Y[ word_index ] = 1
                        test_list_Y.append(word)
                        #print('word: ', self.tokens_list[ word_index ])
                        
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
                            neighbor_index = np.where(self.tokens_list == neighbor)[0][0]
                            cbow_vec_X[ neighbor_index ] = 1
                            test_list_X.append(neighbor)
                            #print('neighbor: ', self.tokens_list[ neighbor_index ])
                            
                        data.append([ cbow_vec_X , cbow_vec_Y ])                                
                        #time.sleep(1)
                        #print('palavra target: ', test_list_Y, '; neighbors: ', test_list_X)
                                                
                            
            if pdf_batch_counter % pdf_batch_size == 0 or file_tag == proc_documents[-1]:
                            
                #carregando a rede neural
                model = load_model(self.diretorio + f'/Outputs/WV/{self.matrix_name}.h5')
                
                #The fit method does the training in batches
                # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
                # data[: , 0] = X ; data[: , 1] = Y
                data = np.array(data)
                #print(data[: , 0].shape, data[: , 1].shape)
                history = model.fit(x = data[: , 0], y = data[: , 1], epochs=n_epochs, batch_size=batch_size)
                
                #salvando o modelo h5            
                model_save_folder = self.diretorio + f'/Outputs/WV/{self.matrix_name}.h5'
                model.save(model_save_folder)
                
                #carregando o histórico
                avg_acc = round( history.history['accuracy'][0] , 2)
                    
                saving_acc_to_CSV(last_PDF_file = file_tag, 
                                  settings = f'{self.matrix_name}', 
                                  acc = avg_acc, 
                                  folder = '/Outputs/WV/',
                                  diretorio = self.diretorio)
                            
                #salvando os weights em .csv
                l1_weights = model.get_layer(name='layer1').get_weights()[0] #0 para os weights e 1 para os bias
                #print(l1_weights.shape, self.n_tokens)
                wv_df = pd.DataFrame(l1_weights, index = self.tokens_list)
                wv_df.to_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv')
                
                #salvando o número do último arquivo processado
                save_index_to_TXT(file_tag, f'counter_File_index_{self.matrix_name}', '/Outputs/LOG', diretorio = self.diretorio)
                
                #plotando os treinamentos
                self.plot_2DWV( words=['antioxidant', 'anticancer', 'antibacterial', 'leishmania', 'cancer', 'heart', 'liver',
                                       'heterocyclic', 'mutans','candida', 'albicans', 'streptococcus'],
                                       plot_batch = pdf_batch_counter)
                
                #deletando o graph da última sessão
                backend.clear_session()
            
                del model
                del history
                del wv_df
                
                #apagando os dados do treino
                data = []
        
                
                
    def get_matrix_vector_stats(self):

        print('( Function: get_matrix_vector_stats )')
        print('wv_stats: ', f'( {self.wv_model}', ' ; ', f'{self.matrix_name} )')
        
        #checando erros de instanciação/inputs
        import numpy as np
        from FUNCTIONS import save_dic_to_json
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import error_print_abort_class
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        
        import pandas as pd
        import os
        
        if os.path.exists(self.diretorio + '/Outputs/WV/MATRIX_VECTORS_STATS.json'):
            print('Arquivo matrix_vectors_stats encontrado em ~/Outputs/WV/MATRIX_VECTORS_STATS.json')
            wv_stats = load_dic_from_json(self.diretorio + '/Outputs/WV/MATRIX_VECTORS_STATS.json')
        else:
            wv_stats = {}        
       
        if self.wv_model == 'w2vec':
            vector_matrix = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv', index_col = 0)            
            try:
                if self.matrix_name in wv_stats[self.wv_model].keys():
                    print(f'Stats já encontrada para o WV {self.matrix_name}')
                    print('> Abortando função: WV.get_matrix_vector_stats')
                    return
            except KeyError:
                pass
            
        elif self.wv_model == 'svd':
            vector_matrix = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv', index_col = 0)
            try:
                if self.matrix_name in wv_stats[self.wv_model].keys():
                    print(f'Stats já encontrada para o WV {self.matrix_name}')
                    print('> Abortando função: WV.get_matrix_vector_stats')
                    return
            except KeyError:
                pass
        
        try:
            wv_stats[self.wv_model]
        except KeyError:
            wv_stats[self.wv_model] = {}            
        
        wv_stats[self.wv_model][self.matrix_name] = {}
        wv_stats[self.wv_model][self.matrix_name]['min'] = round(np.min(vector_matrix.values), 10)
        wv_stats[self.wv_model][self.matrix_name]['max'] = round(np.max(vector_matrix.values), 10)
        wv_stats[self.wv_model][self.matrix_name]['avg'] = round(np.mean(vector_matrix.values), 10)
        wv_stats[self.wv_model][self.matrix_name]['std'] = round(np.std(vector_matrix.values), 10)
        wv_stats[self.wv_model][self.matrix_name]['median'] = round(np.median(vector_matrix.values), 10)
        wv_stats[self.wv_model][self.matrix_name]['1_quartile'] = round(np.quantile(vector_matrix.values, 0.25), 10)
        wv_stats[self.wv_model][self.matrix_name]['3_quartile'] = round(np.quantile(vector_matrix.values, 0.75), 10)
        wv_stats[self.wv_model][self.matrix_name]['0.05_quantile'] = round(np.quantile(vector_matrix.values, 0.05), 10)
        wv_stats[self.wv_model][self.matrix_name]['0.95_quantile'] = round(np.quantile(vector_matrix.values, 0.95), 10)

        print('Salvando o wv_stats em ~/Outputs/WV/MATRIX_VECTORS_STATS.json')
        save_dic_to_json(self.diretorio + '/Outputs/WV/MATRIX_VECTORS_STATS.json', wv_stats)



    def find_terms_sem_similarity(self, n_similar_terms_to_overlap = 5, minimum_app_count_to_put_into_2grams = 1):

        print('( Function: find_terms_sem_similarity )')
        
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        
        import time
        import numpy as np
        import pandas as pd
        from FUNCTIONS import get_filenames_from_folder
        from functions_TEXTS import get_term_list_from_TXT
        from functions_VECTORS import get_close_vecs
        import spacy
        nlp = spacy.load('en_core_web_sm')
        import os

        #carregando o modelo WV
        if self.wv_model == 'gensim':
            import gensim.downloader as api
            wv = api.load('word2vec-google-news-300')
        else:
            wv = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv', index_col = 0)
            print('Carregando: ', f'/Outputs/WV/{self.matrix_name}.csv')

        #coletando os n1grams determinados com o term_semantic_similarity        
        filenames = get_filenames_from_folder(self.diretorio + '/Settings', file_type = 'txt')
        filtered_filenames = [filename for filename in filenames if filename[ : len('input_topic_') ] == 'input_topic_']
    
        for filename in filtered_filenames:
            #carregando a lista de termos a serem pesquisados
            words_to_find_similatiry = get_term_list_from_TXT(self.diretorio + f'/Settings/{filename}.txt')
            #definindo o nome do tópico (semantic meaning)
            input_topic  = filename[ len('input_topic_') : ]
            
            #varrendo cada termo
            for term in words_to_find_similatiry:
                
                print('\nSemantic meaning: ', input_topic)
                print(f'Finding similar terms for * {term} *')
        
                if not os.path.exists(self.diretorio + '/Outputs/nGrams/Semantic'):
                    os.makedirs(self.diretorio + '/Outputs/nGrams/Semantic')    
        
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
        
                #coletando os termos próximos
                similar_terms = []
                for i in indexes_closest_vecs:
                    token = self.tokens_list[i]
                    similar_terms.append(token)
                    #print(token)
                    
                #criando a DF de termos (n1grams) similares
                if not os.path.exists(self.diretorio + f'/Outputs/nGrams/Semantic/n1gram_{input_topic}.csv'):
                    n1gram_sim_terms_DF = pd.DataFrame(columns=['Sem_App_Counter', 'sim_check'], index=wv.index.values)            
                    n1gram_sim_terms_DF.index.name = 'index'
                    #setando as condições inicias da DF
                    for token in n1gram_sim_terms_DF.index:
                        n1gram_sim_terms_DF.loc[token] = ( 0 , False )
                    n1gram_sim_terms_DF.to_csv(self.diretorio + f'/Outputs/nGrams/Semantic/n1gram_{input_topic}.csv')
                    print('Criando a DF 'f'~/Outputs/nGrams/Semantic/n1gram_{input_topic}.csv')
                    
                #carregando a DF de termos (n1grams) similares
                else:
                    n1gram_sim_terms_DF = pd.read_csv(self.diretorio + f'/Outputs/nGrams/Semantic/n1gram_{input_topic}.csv', index_col=0)
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
                n1gram_sim_terms_DF.to_csv(self.diretorio + f'/Outputs/nGrams/Semantic/n1gram_{input_topic}.csv')                        
                
            
        #consolidando os n2grams_DF
        for filename in filtered_filenames:
            #definindo o nome do tópico (semantic meaning)
            input_topic  = filename[ len('input_topic_') : ]
            print('\nSemantic meaning: ', input_topic)
            print('Finding similar n2grams terms')
            
            #carregando o n1gram_DF
            n1gram_sim_terms_DF = pd.read_csv(self.diretorio + f'/Outputs/nGrams/Semantic/n1gram_{input_topic}.csv', index_col=0)
            n1gram_sim_terms_DF.index.name = 'index'
            
            #colocando os bigrams similares ao termo em uma DF
            if not os.path.exists(self.diretorio + f'/Outputs/nGrams/Semantic/n2gram_{input_topic}.csv'):
                n2gram_sim_terms_DF = pd.DataFrame(columns=['token_0', 'token_1', 'TOTAL', 'SENT','PDF'])
                n2gram_sim_terms_DF.index.name = 'index'
                print('Criando a DF 'f'~/Outputs/nGrams/Semantic/n2gram_{input_topic}.csv')
                
            else:
                print(f'CSV file ~/Outputs/nGrams/Semantic/n2gram_{input_topic}.csv encontrado...')
                print(f'Abortando a função para o tópico {input_topic}...')
                return
                #n2gram_sim_terms_DF = pd.read_csv(self.diretorio + f'/Outputs/nGrams/Semantic/n2gram_{input_topic}.csv', index_col=0)
                #n2gram_sim_terms_DF.index.name = 'index'
            
            
            #checando se a DF de n2gram replace já existe
            if os.path.exists(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv'):
                n2grams_to_replace = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv', index_col = 0)
            else:
                n2grams_to_replace = pd.DataFrame(columns=['token_0', 'token_1', 'Sem_App_Counter_t0', 'Sem_App_Counter_t1', 'Score', 'min_delta',
                                                           'TOTAL', 'SENT','PDF', 'check_PDF_pres', 'check_SENT_pres'])
                n2grams_to_replace.index.name = 'index'
            
            
            #carregand os termos que não deverão entrar na procura por bigrams
            terms_not_to_include = get_term_list_from_TXT(self.diretorio + '/Settings/input_terms_not_to_include_in_bigrams.txt')
                        
            #carregando o n2gram DF filtrada (por score e delta de contagem)
            n2grams_filtered = pd.read_csv(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n2grams_filtered.csv', index_col = 0)
            #encontrando as combinações de tokens (n2grams)
            for token in n1gram_sim_terms_DF.index:
                if n1gram_sim_terms_DF.loc[token, 'Sem_App_Counter'] >= minimum_app_count_to_put_into_2grams:
                    for neighbor in n1gram_sim_terms_DF.index:
                        if n1gram_sim_terms_DF.loc[neighbor, 'Sem_App_Counter'] >= minimum_app_count_to_put_into_2grams:
                            if token != neighbor:
                                #só a cond 4 está sendo usada
                                cond1 = True #token + '_' + neighbor in n2grams_filtered.index
                                cond2 = True #nlp(token)[0].pos_.lower() != 'verb'
                                cond3 = True #nlp(neighbor)[0].pos_.lower() != 'verb'
                                cond4 = {}
                                #o token e o neighbor não devem estar na lista a ser excluida
                                for t in terms_not_to_include:
                                    if t not in (token, neighbor):
                                        cond4[t] = True
                                    else:
                                        cond4[t] = False
                                
                                if False not in [cond1, cond2, cond3] + list(cond4.values()):
                                    try:
                                        #essa DF é para substituição dos bigrams no texto
                                        n2grams_to_replace.loc[token + '_' + neighbor] = (token, 
                                                                                          neighbor ,                                                                                     
                                                                                          n1gram_sim_terms_DF.loc[token, 'Sem_App_Counter'], 
                                                                                          n1gram_sim_terms_DF.loc[neighbor, 'Sem_App_Counter'],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'Score' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'min_delta' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'TOTAL' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'SENT' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'PDF' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'check_PDF_pres' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'check_SENT_pres' ])
                                        
                                        #essa DF é para análise e plotagem dos resultados
                                        n2gram_sim_terms_DF.loc[token + '_' + neighbor] = (token, 
                                                                                          neighbor ,                                                                                     
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'TOTAL' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'SENT' ],
                                                                                          n2grams_filtered.loc[ token + '_' + neighbor , 'PDF' ])
                                    except KeyError:
                                        continue
                                    
            #salvando as DF
            #DF para substituição de bigrams no text
            n2grams_to_replace.sort_values(by=['min_delta'], ascending=True, inplace=True)
            n2grams_to_replace.index.name = 'index'
            n2grams_to_replace.to_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')
            #DF para análise e plotagem
            n2gram_sim_terms_DF.sort_values(by=['PDF'], ascending=False, inplace=True)
            n2gram_sim_terms_DF.to_csv(self.diretorio + f'/Outputs/nGrams/Semantic/n2gram_{input_topic}.csv')
    
        del n1gram_sim_terms_DF
        del n2grams_filtered
        del n2gram_sim_terms_DF
        del n2grams_to_replace



    def combine_topic_vectors_by_sem_similarity(self, n_largest_topic_vals = 10, min_sem_app_count_to_get_topic = 4):

        print('( Function: combine_topic_vectors_by_sem_similarity )')        
        
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        if self.wv_model != 'svd' or self.mode != 'truncated_svd':
            print('Erro! A função precisa ser usada com wv_model = svd e mode = truncated_svd.')
            print('> Abortando função: WV.combine_topic_vectors_by_sem_similarity')
            return
        
        from functions_VECTORS import normalize_1dvector
        from functions_VECTORS import get_largest_topic_value_indexes
        from FUNCTIONS import save_dic_to_json
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import get_filenames_from_folder
        import time
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np            

        #carregando o modelo WV
        wv = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv', index_col = 0)
        #carregando o wv_stats
        #wv_stats_dic = load_dic_from_json(self.diretorio + '/Outputs/WV/MATRIX_VECTORS_STATS.json')
        #wv_stats = wv_stats_dic[f'{self.wv_model}'][f'{self.matrix_name}']
        #print('wv_stats usada: ', f'{self.wv_model} ', f'{self.matrix_name}')

        #coletando os n1grams determinados com o term_semantic_similarity        
        filenames = get_filenames_from_folder(self.diretorio + '/Outputs/nGrams/Semantic', file_type = 'csv')
        filtered_filenames = [filename for filename in filenames if (filename[ : 6 ] == 'n1gram') and (filename[ -len('regex') : ] != 'regex')]
                    
        for filename in filtered_filenames:
            #carregando a lista de termos a serem pesquisados
            n1gram_DF = pd.read_csv(self.diretorio + f'/Outputs/nGrams/Semantic/{filename}.csv', index_col=0)
            words_to_find_similatiry = [index for index in n1gram_DF.index if n1gram_DF.loc[index, 'Sem_App_Counter'] >= min_sem_app_count_to_get_topic]

            #determinando o signifado semântico desses termos
            semantic_meaning = filename[ len('n1gram_') : ]
        
            #array para concatenar todos os índices    
            larg_val_index_concat = np.array([])    
        
            #gerando um vetor para colocar as somas
            wv_dim = len(wv.columns)
            sum_topic_vector = np.zeros(wv_dim)
        
            #definindo os vetores finais de probabilidade
            overlap_topic_vector = np.zeros(wv_dim)
        
            #caso haja pelo menos um termo para encontrar os topic vectors
            if len(words_to_find_similatiry) > 1:
                #varrendo cada termo
                for term in words_to_find_similatiry:
                    
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
                overlap_topic_prob_vector = normalize_1dvector(overlap_topic_vector)
                sum_topic_prob_vector = normalize_1dvector(sum_topic_vector)
                
                fig, axs = plt.subplots(1, 2, tight_layout=True)
                #plotando os overlaps dos vetores
                axs[0].bar(np.arange(len(overlap_topic_prob_vector)), overlap_topic_prob_vector)
                axs[0].set_xlabel('Topic index')
                axs[0].set_ylabel('Overlap occurrenes')                
                
                #plotando a soma dos vetores
                axs[1].bar(np.arange(len(sum_topic_prob_vector)), sum_topic_prob_vector)
                axs[1].set_xlabel('Topic index')
                axs[1].set_ylabel('Topic values sum')   
       
                plt.savefig(self.diretorio + f'/Outputs/Models/LSA_topic_{semantic_meaning}.png', dpi=200)
                
                #salvando o dic em json    
                if os.path.exists(self.diretorio + '/Outputs/Models/sem_topic_vectors.json'):
                    dic = load_dic_from_json(self.diretorio + '/Outputs/Models/sem_topic_vectors.json')
                    dic[semantic_meaning] = {}
                    dic[semantic_meaning]['topic_overlap'] = list(overlap_topic_prob_vector)
                    dic[semantic_meaning]['topic_sum'] = list(sum_topic_prob_vector)
                    save_dic_to_json(self.diretorio + '/Outputs/Models/sem_topic_vectors.json', dic)
                else:
                    dic = {}
                    dic[semantic_meaning] = {}
                    dic[semantic_meaning]['topic_overlap'] = list(overlap_topic_prob_vector)
                    dic[semantic_meaning]['topic_sum'] = list(sum_topic_prob_vector)
                    save_dic_to_json(self.diretorio + '/Outputs/Models/sem_topic_vectors.json', dic)    

            else:
                print('ERRO na combinação de Topic vectors...')
                print(f'Os termos em "{filename}" com valores "Sem_App_Counter" abaixo do threshold ("min_sem_app_count_to_get_topic" {min_sem_app_count_to_get_topic})')
                time.sleep(3)


    def plot_2DWV(self, words=['aaa', 'bbb', 'ccc', 'ddd'], plot_batch = 0):

        print('( Function: plot_2DWV )')
                
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        
        from matplotlib import pyplot as plt        
        from sklearn.manifold import TSNE
        import pandas as pd

        if self.wv_model == 'gensim':
            import gensim.downloader as api
            wv = api.load('word2vec-google-news-300')
        else:
            #carregando o modelo WV
            wv = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.matrix_name}.csv', index_col = 0)
        
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

        if self.wv_model == 'svd':
            plt.savefig(self.diretorio + f'/Outputs/WV/PlotQTest_{self.matrix_name}_{plot_batch}.png', dpi=200)
            #plt.show()
        else:
            plt.savefig(self.diretorio + f'/Outputs/WV/PlotQTest_{self.matrix_name}_{plot_batch}.png', dpi=200)
            #plt.show()

        del wv



'''
def split_TFIDF_to_SPARSE(self, batch_size = 50000):

    print('( Function: split_TFIDF_to_SPARSE )')

    #checando erros de instanciação/inputs
    from FUNCTIONS import error_print_abort_class
    from FUNCTIONS import error_incompatible_strings_input
    from FUNCTIONS import get_file_batch_index_list
    abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('svd'), class_name = self.class_name)
    if True in (abort_class, self.abort_class):
        error_print_abort_class(self.class_name)
        return

    
    import h5py        
    from FUNCTIONS import get_filenames_from_folder
    from scipy import sparse
    import numpy as np
    import os
    
    if not os.path.exists(self.diretorio + '/Outputs/TFIDF/npz'):
        os.makedirs(self.diretorio + '/Outputs/TFIDF/npz')
                
    #determinando os slices para os batchs
    batch_indexes = get_file_batch_index_list(self.n_documents, batch_size)
    
    print('Converting h5 to SPARSE...')
    print('Determinando os slices para os batches...')
    slice_indexes = list(range(0, self.n_documents, batch_size)) #indexes dos slices
    batch_indexes = []
    for i in range(len(slice_indexes)):
        try:
            batch_indexes.append([ slice_indexes[i] , slice_indexes[i + 1] - 1])
        except IndexError:
            pass
    batch_indexes.append([slice_indexes[-1] , self.n_documents - 1])
    
    #testando se as sparse já foram convertidas        
    try:
        all_sparse_files = get_filenames_from_folder(self.diretorio + '/Outputs/TFIDF/npz', file_type = 'npz')
        n_files = len( [filename for filename in all_sparse_files if (f'_rNgram_{self.replace_Ngrams}' in filename)] )
    except TypeError:
       n_files = 0 
    if n_files == len(batch_indexes):
        print('A matriz TFIDF já foi convertida para SPARSE.')
        print('> Abortando função: WV.split_TFIDF_to_SPARSE')
        return   

    #salvando as sparse matrix
    counter_npz = 0 
    print('Dividindo o processamento do arquivo TFIDF FULL H5 em ', len(batch_indexes), ' batches...')
    for index in batch_indexes:            
        #abrindo a matrix concatenada (FULL)
        print('Abrindo a matrix h5 concatenada (FULL TFIDF)')
        h5_M = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5', 'r')
        M = h5_M['data']                        
        print('TFIDF matrix shape: ', M.shape)
        print('Indexes: ', index[0] , index[1])
        sm = sparse.csr_matrix(M[ index[0] : index[1]+1], dtype = np.float64)
        print('SPARSE TFIDF matrix shape: ', sm.shape)
        #salvando as sparse matrix em arquivos npz separados
        if 0 <= counter_npz < 10:
            counter_npz_to_save = '0' + str(counter_npz)
        else:
            counter_npz_to_save = str(counter_npz)
        print(f'Salvando a sparse ~/Outputs/TFIDF/npz/sparse_csr_rNgram_{self.replace_Ngrams}_{counter_npz_to_save}.npz')
        sparse.save_npz(self.diretorio + f'/Outputs/TFIDF/npz/sparse_csr_rNgram_{self.replace_Ngrams}_{counter_npz_to_save}.npz', sm, compressed=True)
        h5_M.close()        
        counter_npz += 1
        print('Batches processados: ', counter_npz)
    
        del h5_M
        del M
        del sm'''
        

'''
def concat_TFIDF_DFs(self):

    print('( Function: concat_TFIDF_DFs )')

    #checando erros de instanciação/inputs
    from FUNCTIONS import error_print_abort_class
    from FUNCTIONS import error_incompatible_strings_input
    abort_class = error_incompatible_strings_input('wv_model', self.wv_model, ('svd'), class_name = self.class_name)
    if True in (abort_class, self.abort_class):
        error_print_abort_class(self.class_name)
        return
        
    import os
    import h5py
    import numpy as np
    from FUNCTIONS import load_index_from_TXT    

    #contador de todas as sentenças processadas
    total_sentence_processed = load_index_from_TXT('Sentence_counts', '/Outputs/LOG', diretorio = self.diretorio)
    try:
        #contador das sentenças processadas no slice 1
        sentence_counter_s1 = load_index_from_TXT(f'TFIDF_Sentence_counter_s1_rNgram_{self.replace_Ngrams}', '/Outputs/LOG', diretorio = self.diretorio)
    except FileNotFoundError:
        sentence_counter_s1 = '0'
        pass
                    
    #checando se já uma matrix h5 concatenada
    if os.path.exists(self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5'):
        print(f'Já existe uma matriz TFIDF concatenada em ~/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5')
        print('> Abortando função: WV.concat_TFIDF_DFs')
        return
    
    #caso todas as sentenças tenham sido processadas de uma só vez
    elif sentence_counter_s1 == total_sentence_processed:
        os.rename(self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_s1_rNgram_{self.replace_Ngrams}.h5', self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5')
        print('O TFIDF de todos os documentos foi calculado de uma vez só (1 slice)')
        print(f'Renomeando o arquivo H5 de: TFIDF_s1.h5 para TFIDF_full_rNgram_{self.replace_Ngrams}.h5')
        return 
        
    else:
        #criando um matrix h5 para concaternar todos os arquivos
        #número de linhas e colunas da matrix concatenada (FULL)
        n_rows = self.n_documents
        n_columns = len(self.IDF.index)
        h5_M = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5', 'w')
        h5_M.create_dataset('data', shape=(n_rows, n_columns), dtype=np.float64)
        #print(h5_M['data'].shape)
        h5_M.close()
        print('Criando matriz FULL TFIDF...')
        print('n_sentences (n_rows) = ', n_rows)
        print('n_tokens (n_columns) = ', n_columns)

    
    #montando a lista de arquivos h5 com os valores de TFIDF
    file_list = os.listdir(self.diretorio + '/Outputs/TFIDF/H5') #lista de arquivos
    H5_documents = []
    for filename in file_list:
        if filename[ -2 :  ].lower() == 'h5' and filename[ 6 : 7 ].lower() == 's':
            H5_documents.append(filename)
    H5_documents = sorted(H5_documents)

    #abrindo a matrix concatenada (FULL)
    h5_M = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5', 'a')
    M = h5_M['data']
    
    if M.shape[0] == n_rows and M.shape[1] == n_columns:
        pass
    else:
        print('O formato da Matriz concatenada (M.shape) não é igual ao número de rows e columns.')
        print('Encerrando...')
        #fechando o arquivo h5 da matriz concatenada
        h5_M.close()
        return
    
    #concatenando os arquivos h5 em uma matriz
    row_number = 0
    for filename in H5_documents:
        h5_m = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/{filename}', 'r')
        m_to_concat = h5_m['data']
        print(f'{filename} tem {m_to_concat.shape[0]} sentenças')            
        for i in range(m_to_concat.shape[0]):
            M[row_number] = m_to_concat[i]
            #mostrando o andamento da concatenação
            if row_number % 50000 == 0:
                print('Processing row: ', row_number)
                print('Closing and opening h5 file again...')
                #fechando a matrix concatenada
                h5_M.close()
                #abrindo a matrix concatenada (FULL)
                h5_M = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/TFIDF_full_rNgram_{self.replace_Ngrams}.h5', 'a')
                M = h5_M['data']                    
            row_number += 1                
            
        #fechando o arquivo h5 da matriz cortada (sliced)
        h5_m.close()
    
    h5_M.close()
    print('Concat concluido.')
    
    del h5_m
    del h5_M
    del m_to_concat
    del M'''