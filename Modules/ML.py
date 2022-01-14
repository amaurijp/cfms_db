#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class stats(object):
    
    def __init__(self, filter_stopwords = False, replace_Ngrams = False, diretorio = None):

        print('( Class: stats )')
        
        import os
        import pandas as pd
        from nltk.corpus import stopwords
        import spacy
        
        self.filter_stopwords = filter_stopwords
        self.replace_Ngrams = replace_Ngrams
        self.diretorio = diretorio

        #carregando os tokens do IDF
        print('Carregando os tokens do IDF')
        self.tokens_DF = pd.read_csv(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv', index_col = 0)
        self.tokens_list = self.tokens_DF.index.values
        self.n_tokens = len(self.tokens_list)
        print('n_tokens = ', self.n_tokens)
    
        self.tokenizer_func = spacy.load('en_core_web_sm')
        self.stopwords_list = stopwords.words('english')
        
        #checando a pasta /Outputs/Models/
        if not os.path.exists(self.diretorio + '/Outputs/Models'):
            os.makedirs(self.diretorio + '/Outputs/Models')


    def get_topic_prob_func(self, input_DF_file_names = ['pyrolysis_temperature_extracted_mode_match_topic_vector'], feature = 'pyrolysis_temperature'):

        print('( Function: get_topic_prob_func )')
        
        import numpy as np
        import time
        import pandas as pd
        import regex as re
        from functions_TOKENS import get_tokens_from_sent
        from FUNCTIONS import get_filenames_from_folder

        #criando o dicionário para os DFs
        DFs_concat_dic = {}
        input_DF_dic = {}

        #criando o DF da função prob
        topic_func_prob_DF = pd.DataFrame([], index = self.tokens_list)
        topic_func_prob_DF['token_index'] = np.arange(0, len(self.tokens_list), 1)        

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
                sent_tokens_filtered = get_tokens_from_sent(sent,
                                                            tokenizer_func = self.tokenizer_func, 
                                                            tokens_list = self.tokens_list, 
                                                            filter_stopwords = self.filter_stopwords, 
                                                            replace_Ngrams = self.replace_Ngrams, 
                                                            stopwords_list = self.stopwords_list)
                
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
        print(f'Salvando DF ~/Outputs/Models/prob_func_{feature}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.csv')
        topic_func_prob_DF.to_csv(self.diretorio + f'/Outputs/Models/prob_func_{feature}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.csv')



    def set_bayes_classifier(self, prob_func_filename = 'pyrolysis_temperature'):
    
        import pandas as pd    
    
        #abrindo a função de probabilidade para o tópico
        self.prob_func_DF = pd.read_csv(self.diretorio + f'/Outputs/Models/prob_func_{prob_func_filename}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.csv', index_col=0)    



    def use_bayes_classifier(self, sent):
        
        import time
        import math
        import numpy as np
        from FUNCTIONS import load_dic_from_json
        from functions_TOKENS import get_tokens_from_sent        
        
        #abrindo a estatística de pdf_sent
        pdf_sent_stats = load_dic_from_json(self.diretorio + '/Outputs/LOG/stats_PDF_SENT.json')
        
        #print('\nBayesian classifier for sent:')
        #print(sent)
        
        #coletando os tokens da sentença                        
        sent_tokens = get_tokens_from_sent(sent, 
                                           tokenizer_func = self.tokenizer_func, 
                                           tokens_list = self.tokens_list,
                                           stopwords_list = self.stopwords_list, 
                                           filter_stopwords = self.filter_stopwords,
                                           replace_Ngrams = self.replace_Ngrams)
                
        #definindo o valor de prior_Ck (probabilidade da sentença ser da classe; é aproximadamente 1 por PDF)
        #usamos como valor inicial = 1 / número médio de sentença por PDF
        prior_Ck = 1 / pdf_sent_stats['avg']
        post_Ck = prior_Ck
        #definindo o valor de prior_not_Ck 
        prior_not_Ck = ( 1 - prior_Ck )
        post_not_Ck = prior_not_Ck
        #print('( prior_Ck: ', post_Ck, ' , prior_not_Ck: ', post_not_Ck, ' )')
        try:
            for token in sent_tokens:
                #print(token)
                if token in self.tokens_list:
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
    
    def __init__(self, machine_type = 'LSTM', wv_model ='w2vec', wv_matrix_name = 'w2vec_xx',  feature = 'temperature', filter_stopwords = False, replace_Ngrams = False, diretorio=None):

        print('( Class: train machine )')
        
        import pandas as pd
        self.diretorio = diretorio
        self.machine_type = machine_type.lower()
        self.feature = feature.lower()
        self.replace_Ngrams = replace_Ngrams
        self.class_name = 'machine'
        self.wv_model = wv_model
        self.wv_matrix_name = wv_matrix_name
        self.filter_stopwords = filter_stopwords

        #testando os erros de inputs
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import error_incompatible_strings_input        
        self.abort_class = error_incompatible_strings_input('machine_type', 
                                                            machine_type, ('logreg', 'randomforest', 'svm', 'lstm', 'conv1d', 'conv2d','conv1d_lstm'), 
                                                            class_name = self.class_name)

        #caso seja uma NN
        if self.machine_type in ('lstm', 'conv1d', 'conv2d','conv1d_lstm'):
            #carregando o WV
            try:
                self.wv = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.wv_matrix_name}.csv', index_col = 0)
            except FileNotFoundError:
                print(f'Erro para a entrada wv_name: {self.wv_matrix_name}')
                print('Entradas disponíveis: olhar os arquivos .csv na pasta ~/Outputs/WV/')
                print('> Abortando a classe: machine')
                self.abort_class = True
                return        
            
            self.wv_emb_dim = len(self.wv.columns)
            
            #carregando a estatísticas dos word_vectors
            self.wv_stats = load_dic_from_json(self.diretorio + '/Outputs/WV/MATRIX_VECTORS_STATS.json')[wv_model][wv_matrix_name]
            
            #carregando os tokens do IDF
            self.IDF = pd.read_csv(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv', index_col = 0)
            self.tokens_list = self.IDF.index.values
        
        #caso seja outro ML
        else:
            #carregando a estatísticas dos topic_vectors
            self.tv_stats = load_dic_from_json(self.diretorio + '/Outputs/TFIDF/H5/DOC_TOPIC_MATRIX_STATS.json')
            
    

    def set_train_sections(self, 
                           section_name = 'methodolody',
                           sent_batch_size = 5,
                           pdf_batch_size = 5,
                           n_epochs = 5,
                           sent_stride = 1,
                           vector_type = 'wv'):
                           
        self.section_name = section_name
        self.sent_batch_size = sent_batch_size
        self.pdf_batch_size = pdf_batch_size
        self.n_epochs = n_epochs
        self.sent_stride = sent_stride
        self.vector = vector_type.lower()

        print('( Function: set_train_sections )')



    def train_on_sections(self):
        
        print('( Function: train_on_sections )')

        #checando erros de instanciação/inputs
        import random
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import error_incompatible_strings_input
        abort_class = error_incompatible_strings_input('vector', self.vector, ('wv', 'tv'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return

        import h5py
        import time
        import os
        from joblib import dump, load
        import keras.backend as backend
        from keras.optimizers import SGD, Adam, RMSprop
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from FUNCTIONS import generate_PR_results
        from FUNCTIONS import save_dic_to_json
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import save_index_to_TXT
        from FUNCTIONS import load_index_from_TXT
        from functions_TOKENS import find_min_max_token_sent_len
        from functions_VECTORS import get_wv_from_sentence
        from functions_VECTORS import get_tv_from_sent_index
        from NNs import NN_model
        import numpy as np
        import pandas as pd

        print(f'Model: sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}')

        #caso os Ngrams sejam substituidos
        if (self.replace_Ngrams is True):
            #abrindo o DF com os N2GRAMS para substituir
            n2grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')        
            n2grams_DF.set_index('index', inplace=True)
            try:
                n3grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n3grams_to_replace.csv')
                n3grams_DF.set_index('index', inplace=True)
                
            except FileNotFoundError:
                n3grams_DF = None
        
        else:
            n2grams_DF = None
            n3grams_DF = None
        
        #checando a pasta /Outputs/Sections/
        if not os.path.exists(self.diretorio + '/Outputs/Sections'):
            os.makedirs(self.diretorio + '/Outputs/Sections')
        
        #coletando os filenames para treinar as seções
        filenames = get_filenames_from_folder(self.diretorio + '/Outputs/Sections', file_type = 'csv')
        if len(filenames) == 0:    
            print('Erro! Não há arquivos para treinar na pasta ~/Outputs/Sections/')
            print('> Abortando função: train_machine.train_on_sections')
            return        
        
        #contador de arquivos (para parar e resumir o fit)
        try:
            file_name = load_index_from_TXT(f'counter_File_index_sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}', 
                                            '/Outputs/LOG', 
                                            diretorio = self.diretorio)
            print('Training_counter_File_index: ', file_name)
            file_index = filenames.index(file_name) + 1
        except FileNotFoundError:
            file_index = 0
        
        
        #caso seja NN
        nn_check = False
        if self.machine_type in ('lstm', 'conv1d', 'conv2d','conv1d_lstm'):
            nn_check = True
            #definido o maior valor permitido de sent_token_len (quantidade de tokens por sentença) o qual será o input da NN
            sent_tokens_stats = load_dic_from_json(self.diretorio + '/Outputs/LOG/stats_SENT_TOKENS.json')
            sent_max_token_len , sent_min_token_len , single_sent_min_token_len = find_min_max_token_sent_len(sent_batch_size = self.sent_batch_size, 
                                                                                                              machine_type = self.machine_type, 
                                                                                                              sent_tokens_stats = sent_tokens_stats, 
                                                                                                              filter_stopwords = self.filter_stopwords)

            #definindo os parâmetros da NN
            NN_parameters_dic = {}        
            
            #general parameters
            NN_parameters_dic['machine_type'] = self.machine_type
            NN_parameters_dic['feature'] = None
            NN_parameters_dic['section_name'] = self.section_name
            NN_parameters_dic['vector_type'] = self.vector
            NN_parameters_dic['replace_Ngrams'] = self.replace_Ngrams
            NN_parameters_dic['filter_stopwords'] = self.filter_stopwords
            NN_parameters_dic['batch_size'] = 50
            NN_parameters_dic['n_epochs'] = self.n_epochs
            NN_parameters_dic['sent_batch_size'] = self.sent_batch_size
    
            #NN inputs
            NN_parameters_dic['input_shape'] = {}
            
            #definindo o input de word_vector para as conv1d e conv2d
            if self.machine_type == 'conv1d':
                NN_parameters_dic['input_shape']['wv'] = (sent_max_token_len, self.wv_emb_dim) #word vector
                #definindo o input de topic_vector para as conv1d (não dá para usar topic_vector com conv2d)
                NN_parameters_dic['input_shape']['tv'] = (self.sent_batch_size , self.wv_emb_dim) #topic vector

            elif self.machine_type == 'conv2d':
                NN_parameters_dic['input_shape']['wv'] = (sent_max_token_len, self.wv_emb_dim, self.sent_batch_size) #word vector
                #definindo o input de topic_vector para as conv1d (não dá para usar topic_vector com conv2d)
                NN_parameters_dic['input_shape']['tv'] = (self.sent_batch_size , self.wv_emb_dim) #topic vector


        
            #NN parameters
            NN_parameters_dic['activation'] = {}
            NN_parameters_dic['activation']['conv1d'] = {}
            NN_parameters_dic['activation']['conv2d'] = {}
            NN_parameters_dic['dropout'] = {}
            NN_parameters_dic['dropout']['conv1d'] = {}
            NN_parameters_dic['dropout']['conv2d'] = {}
            NN_parameters_dic['filters'] = {}
            NN_parameters_dic['filters']['conv1d'] = {}
            NN_parameters_dic['filters']['conv2d'] = {}
            NN_parameters_dic['kernel_initializer'] = {}
            NN_parameters_dic['kernel_initializer']['conv1d'] = {}
            NN_parameters_dic['kernel_initializer']['conv2d'] = {}
            NN_parameters_dic['kernel_size'] = {}
            NN_parameters_dic['kernel_size']['conv2d'] = {}
            NN_parameters_dic['kernel_size']['conv1d'] = {}
            NN_parameters_dic['loss'] = {}
            NN_parameters_dic['loss']['conv1d'] = {}
            NN_parameters_dic['loss']['conv2d'] = {}
            NN_parameters_dic['optimizer'] = {}
            NN_parameters_dic['optimizer']['conv1d'] = {}
            NN_parameters_dic['optimizer']['conv2d'] = {}
            NN_parameters_dic['strides'] = {}
            NN_parameters_dic['strides']['conv1d'] = {}
            NN_parameters_dic['strides']['conv2d'] = {}
            NN_parameters_dic['units'] = {}
            NN_parameters_dic['units']['conv1d'] = {}
            NN_parameters_dic['units']['conv2d'] = {}
            
            
            #model conv1d
            #-----------------------        
            #l1 - conv1d
            NN_parameters_dic['activation']['conv1d']['l1'] = 'elu'
            NN_parameters_dic['filters']['conv1d']['l1'] = 100
            NN_parameters_dic['kernel_initializer']['conv1d']['l1'] = 'he_normal'
            NN_parameters_dic['kernel_size']['conv1d']['l1'] = 3
            NN_parameters_dic['strides']['conv1d']['l1'] = 2
    
            #l2 - conv1d
            NN_parameters_dic['activation']['conv1d']['l2'] = 'elu'
            NN_parameters_dic['filters']['conv1d']['l2'] = 200
            NN_parameters_dic['kernel_initializer']['conv1d']['l2'] = 'he_normal'
            NN_parameters_dic['kernel_size']['conv1d']['l2'] = 5
            NN_parameters_dic['strides']['conv1d']['l2'] = 3
            
            #l3 - dense
            NN_parameters_dic['activation']['conv1d']['l3'] = 'elu'
            NN_parameters_dic['kernel_initializer']['conv1d']['l3'] = 'he_normal'
            NN_parameters_dic['dropout']['conv1d']['l3'] = 0.2
            NN_parameters_dic['units']['conv1d']['l3'] = 500
    
            #l4 - dense
            NN_parameters_dic['activation']['conv1d']['l4'] = 'elu'
            NN_parameters_dic['kernel_initializer']['conv1d']['l4'] = 'he_normal'
            NN_parameters_dic['dropout']['conv1d']['l4'] = 0.2
            NN_parameters_dic['units']['conv1d']['l4'] = 100
    
            #l5 - dense
            NN_parameters_dic['activation']['conv1d']['l5'] = 'elu'
            NN_parameters_dic['kernel_initializer']['conv1d']['l5'] = 'he_normal'
            NN_parameters_dic['dropout']['conv1d']['l5'] = 0.2
            NN_parameters_dic['units']['conv1d']['l5'] = 10
            
            #l6 - dense
            NN_parameters_dic['activation']['conv1d']['l6'] = 'sigmoid'
            NN_parameters_dic['units']['conv1d']['l6'] = 1       
            
            #compile
            NN_parameters_dic['loss']['conv1d'] = 'binary_crossentropy'
            opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            #opt = Adam(learning_rate=1e-6)
            #opt = RMSprop(learning_rate=0.001)
            NN_parameters_dic['optimizer']['conv1d'] = opt
            
            
            #model conv2d
            #-----------------------        
            #l1 - conv2d
            NN_parameters_dic['activation']['conv2d']['l1'] = 'elu'
            NN_parameters_dic['filters']['conv2d']['l1'] = 50
            NN_parameters_dic['kernel_initializer']['conv2d']['l1'] = 'he_normal'
            NN_parameters_dic['kernel_size']['conv2d']['l1'] = 2
            NN_parameters_dic['strides']['conv2d']['l1'] = 1
    
            #l2 - conv2d
            NN_parameters_dic['activation']['conv2d']['l2'] = 'elu'
            NN_parameters_dic['filters']['conv2d']['l2'] = 30
            NN_parameters_dic['kernel_initializer']['conv2d']['l2'] = 'he_normal'
            NN_parameters_dic['kernel_size']['conv2d']['l2'] = 3
            NN_parameters_dic['strides']['conv2d']['l2'] = 2
            
            #l3 - dense
            NN_parameters_dic['activation']['conv2d']['l3'] = 'elu'
            NN_parameters_dic['kernel_initializer']['conv2d']['l2'] = 'he_normal'
            NN_parameters_dic['dropout']['conv2d']['l3'] = 0.2
            NN_parameters_dic['units']['conv2d']['l3'] = 1000
    
            #l4 - dense
            NN_parameters_dic['activation']['conv2d']['l4'] = 'elu'
            NN_parameters_dic['dropout']['conv2d']['l4'] = 0.2
            NN_parameters_dic['units']['conv2d']['l4'] = 100
    
            #l5 - dense
            NN_parameters_dic['activation']['conv2d']['l5'] = 'elu'
            NN_parameters_dic['dropout']['conv2d']['l5'] = 0.2
            NN_parameters_dic['units']['conv2d']['l5'] = 10
            
            #l6 - dense
            NN_parameters_dic['activation']['conv2d']['l6'] = 'sigmoid'
            NN_parameters_dic['units']['conv2d']['l6'] = 1       
            
            #compile
            NN_parameters_dic['loss']['conv2d'] = 'binary_crossentropy'
            opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            #opt = Adam(learning_rate=0.001)
            #opt = RMSprop(learning_rate=0.001, momentum=0.0)
            NN_parameters_dic['optimizer']['conv2d'] = opt        


        if self.vector == 'tv':
            #carregando a matriz DOC_TOPIC
            h5 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'r')
            DOC_TOPIC_matrix = h5['data']
                    
        #definindo as listas para coleta dos dados para processamento de cada batch de arquivos
        X_wv_train = []
        X_tv_train = []
        Y_train = []
        X_wv_test = []
        X_tv_test = []
        Y_test = []
                
        #treinado os modelos com batch de sentenças
        go_to_train = False
        got_pdf_counter = 0
        batch_counter= 1
        print('Preparing data to train/test...')
        for filename in filenames[ file_index : ]:

            print('Processing ', filename, '...')
            section_counter = 0
            sentDF = pd.read_csv(self.diretorio + f'/Outputs/Sections/{filename}.csv', index_col=0)

            #limites de indexes            
            index_first = sentDF.index.values[0]
            index_last = sentDF.index.values[-1]            
            
            #encontrando os index para as sentenças do documento (PDF)
            sent_indexes = range( index_first, ( index_last + 1) - self.sent_batch_size, self.sent_stride)            
            
            #varrendo as sentenças  
            for start_index in sent_indexes:                
                
                #definindo uma lista para coletar somente os targets
                Y_target_section = []
                
                #determinando os indexes para cada conjunto de sentenças (section)
                section_indexes = range( start_index , start_index + self.sent_batch_size )
                #print('Sent indexes to collect: ', section_indexes)
                #varrendo os indexes para pegar um conjunto de sentenças
                min_max_section_token_len_check = True
                min_sent_token_len_check = True
                for sent_index in section_indexes:
                    #print('Getting sentence vectors - sent index: ', sent_index)
                    #sentença
                    sent = sentDF.loc[sent_index, 'Sentence']
                    #target
                    target = sentDF.loc[sent_index, 'Target']
                    
                    #coletando os wvs da sentença
                    if self.vector == 'wv':
                        
                        #colentando os word vectors (scaled) da sentença (section)
                        sent_word_vectors = get_wv_from_sentence(sent, 
                                                                 self.wv,
                                                                 self.wv_stats,
                                                                 collection_token_list = self.tokens_list, 
                                                                 filter_stopwords = self.filter_stopwords,
                                                                 n2grams_DF = n2grams_DF,
                                                                 n3grams_DF = n3grams_DF)
                    
                        #caso o token_sent_len for menor que o determinado no SENT_TOKEN_STATS
                        if sent_word_vectors.shape[0] < single_sent_min_token_len:            
                            #print('Ignorando sentence - sent len:', sent_word_vectors.shape[0], ' (min_sent_token_len allowed: ', single_sent_min_token_len, ')')
                            min_sent_token_len_check = False
                            break
                        
                        #coletando o wv e o target
                        if self.machine_type == 'conv1d':
                            
                            try:
                                X_wv_section = np.vstack((X_wv_section, sent_word_vectors))
                            except NameError:
                                X_wv_section = sent_word_vectors
                            
                            #print('X_wv_section.shape = ', X_wv_section.shape)
    
                        elif self.machine_type == 'conv2d':
                                                                           
                            if (sent_min_token_len <= sent_word_vectors.shape[0] <= sent_max_token_len):
                                zero_m = np.zeros((sent_max_token_len - sent_word_vectors.shape[0], 300))
                                sent_word_vectors = np.vstack((sent_word_vectors, zero_m))                           
                            #caso o número de tokens na sentença seja maior que o permitido
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
                            
                            #print('X_tv_section.shape = ', X_tv_section.shape)
        
                    #coletando o tv da sentença
                    elif self.vector == 'tv':
                        
                        #coletando o vector topico da sentença
                        sent_topic_vectors = get_tv_from_sent_index(sent_index, 
                                                                    tv_stats = self.tv_stats, 
                                                                    scaling = False,
                                                                    normalize = True,
                                                                    DOC_TOPIC_matrix = DOC_TOPIC_matrix, 
                                                                    diretorio = self.diretorio)
                    
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
                    if self.machine_type == 'conv1d' and self.vector == 'wv':
                        if (sent_min_token_len <= X_wv_section.shape[0] <= sent_max_token_len):
                            zero_m = np.zeros((sent_max_token_len - X_wv_section.shape[0], 300))
                            X_wv_section = np.vstack((X_wv_section, zero_m))
                            #print('Final X_wv_section.shape = ', X_wv_section.shape)
                            #print('Final X_tv_section.shape = ', X_tv_section.shape)                                                      
                        #caso o número de tokens nas sentenças concatenadas seja maior que o permitido
                        else:
                            #print('Ignorando section - concat sent len:', X_wv_section.shape[0], ' (min_sent_token_len, ', sent_min_token_len, '; max_sent_token_len: ', sent_max_token_len, ')')
                            min_max_section_token_len_check = False
                                            
                    elif self.machine_type == 'conv2d' and self.vector == 'wv':
                        #fazendo o reshape da matrix de seção para encontrar os channels
                        #(self.sent_batch_size , max_token_len , wv_dim) -> (max_token_len , wv_dim, self.sent_batch_size)
                        X_wv_section = np.dstack(X_wv_section)
                        #print('Final X_wv_section.shape = ', X_wv_section.shape)
                        #print('Final X_tv_section.shape = ', X_tv_section.shape)                        
                                            
                    if min_max_section_token_len_check is True:
                        #separando entre train e test                
                        if self.vector == 'wv':
                            if random.random() > 0.2:
                                X_wv_train.append( X_wv_section )
                                Y_train.append( list(set(Y_target_section))[0] )
                                
                            else:
                                X_wv_test.append( X_wv_section )
                                Y_test.append( list(set(Y_target_section))[0] )
                                
                        elif self.vector == 'tv':
                            if random.random() > 0.2:
                                X_tv_train.append( X_tv_section)
                                Y_train.append( list(set(Y_target_section))[0] )
                                
                            else:
                                X_tv_test.append( X_tv_section)
                                Y_test.append( list(set(Y_target_section))[0] )
                            
                        section_counter += 1
                        #print('( X_wv_train.shape ; X_tv_train.shape ) ( ', np.array(X_wv_train).shape, ' ; ', np.array(X_tv_train).shape, ' ) Last Y_train: ', Y_train[-5 : ] )
                        #print('( X_wv_test.shape  ; X_tv_test.shape  ) ( ', np.array(X_wv_test).shape, ' ; ', np.array(X_tv_test).shape, ' ) Last Y_test: ', Y_test[-5 : ] )
                        #print('( len(Y_train)  ; len(Y_test)  ) ( ', len(Y_train), ' ; ',  len(Y_test), ' )' )
                        #print('Section counter: ', section_counter)                        
                    
                    if self.vector == 'wv':
                        del X_wv_section
                    elif self.vector == 'tv':
                        del X_tv_section
                    del Y_target_section                        

                else:
                    #deletando as arrays da seção
                    del Y_target_section
                    try:                    
                        if self.vector == 'wv':
                            del X_wv_section
                        elif self.vector == 'tv':
                            del X_tv_section
                    except UnboundLocalError:
                        pass
            
            #contador de pdfs processados    
            got_pdf_counter += 1            
            
            #caso o número de pdf processados seja igual ao batch size inserido ou caso seja o último arquivo da pasta
            if got_pdf_counter == self.pdf_batch_size or filename == filenames[-1]:
                go_to_train = True
                    
            if (go_to_train is True):
        
                #caso a máquina seja uma NN
                if nn_check is True:
                    #definindo a NN
                    NN = NN_model()
                    NN.set_parameters(NN_parameters_dic)
                    model = NN.get_model()
                
                #caso a máquina seja de logistic regression
                elif self.machine_type == 'logreg':
                    model_save_folder = self.diretorio + f'/Outputs/Models/sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.joblib'
                    if os.path.exists(model_save_folder):
                        print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
                        print('Carregando o arquivo joblib com o modelo...')
                        model = load(model_save_folder)
                    else:
                        print('Criando modelo ', self.machine_type)
                        model = LogisticRegression(warm_start=False, solver='lbfgs')
                
                #caso a máquina seja de svm
                elif self.machine_type == 'svm':
                    model_save_folder = self.diretorio + f'/Outputs/Models/sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.joblib'
                    if os.path.exists(model_save_folder):
                        print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
                        print('Carregando o arquivo joblib com o modelo...')
                        model = load(model_save_folder)
                    else:
                        print('Criando modelo ', self.machine_type)
                        model = LinearSVC()
                        
                #caso a máquina seja de random forest
                elif self.machine_type == 'randomforest':
                    model_save_folder = self.diretorio + f'/Outputs/Models/sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.joblib'
                    if os.path.exists(model_save_folder):
                        print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
                        print('Carregando o arquivo joblib com o modelo...')
                        model = load(model_save_folder)
                    else:
                        print('Criando modelo ', self.machine_type)
                        model = RandomForestClassifier()
                    
                #dicionário com os resultados do test set
                if os.path.exists(self.diretorio + f'/Outputs/Models/sections_{self.section_name}_training_results.json'): 
                    test_results = load_dic_from_json(self.diretorio + f'/Outputs/Models/sections_{self.section_name}_training_results.json')
                    try:
                        test_results[f'sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}']
                    except KeyError:
                        test_results[f'sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}'] = {}
                else:
                    test_results = {}
                    test_results[f'sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}'] = {}
                    
                print(f'Sentence batch: {batch_counter}')
                print('Shuffling...')
                if self.vector == 'wv':
                    X_wv_train_shuffled = []
                    Y_train_shuffled = []
                    index_list = list(range(len(X_wv_train)))
                    random.shuffle(index_list)
                    for index in index_list:
                        X_wv_train_shuffled.append(X_wv_train[index])
                        Y_train_shuffled.append(Y_train[index])
                    
                    print('Converting to array...')
                    #array dos conjunto de dados de word-vectors e topic-vectors das sentenças
                    X_wv_train = np.array(X_wv_train_shuffled)
                    Y_train = np.array(Y_train_shuffled)
                    X_wv_test = np.array(X_wv_test)
                    Y_test = np.array(Y_test)

                elif self.vector == 'tv':
                    X_tv_train_shuffled = []
                    Y_train_shuffled = []
                    index_list = list(range(len(X_tv_train)))
                    random.shuffle(index_list)
                    for index in index_list:                    
                        X_tv_train_shuffled.append(X_tv_train[index])
                        Y_train_shuffled.append(Y_train[index])
                    
                    print('Converting to array...')
                    #array dos conjunto de dados de word-vectors e topic-vectors das sentenças
                    X_tv_train = np.array(X_tv_train_shuffled)
                    Y_train = np.array(Y_train_shuffled)
                    X_tv_test = np.array(X_tv_test)
                    Y_test = np.array(Y_test)


                print('Training model...')                
                #training                
                if self.machine_type == 'conv1d' and self.vector == 'wv':

                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_train.shape)
                    print('Y_train shape (n_samples, ):', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = X_wv_train , y = Y_train, epochs=self.n_epochs, batch_size = NN_parameters_dic['batch_size'], verbose=1)

                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    results = model.evaluate(x = X_wv_test , y = Y_test, batch_size = NN_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {results[0]}, Acc: {results[1]}')

                elif self.machine_type == 'conv2d' and self.vector == 'wv':

                    print('Machine: ', self.machine_type)
                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim, self.sent_batch_size): ', X_wv_train.shape)
                    print('Y_train shape (n_samples, ):', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = X_wv_train , y = Y_train, epochs=self.n_epochs, batch_size = NN_parameters_dic['batch_size'], verbose=1)

                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim, self.sent_batch_size): ', X_wv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    results = model.evaluate(x = X_wv_test , y = Y_test, batch_size = NN_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {results[0]}, Acc: {results[1]}')

                elif self.machine_type in ('logreg', 'randomforest') and self.vector == 'tv':
                    
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
                    proba_threshold = 0.7
                    results = [0, 0]
                    results[0], results[1] = generate_PR_results(prediction_results, Y_test, proba_threshold = proba_threshold)
                    print(f'Evaluate results - Precision: {results[0]} ; Recall: {results[1]} ; Proba_thr: {proba_threshold}')

                elif self.machine_type == 'svm' and self.vector == 'tv':
                    
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

                
                #caso seja usado os word vectors
                else:
                    print('Erro! Combinação inadequada entre "machine_type" e "vector"')
                    print('> Abortando função: train_machine.train_on_sentences')
                    return
                
                if nn_check is True:
                    #salvando o modelo h5            
                    print('Salvando o modelo de ', self.machine_type)
                    model_save_folder = self.diretorio + f'/Outputs/Models/sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.h5'
                    model.save(model_save_folder)
                
                elif self.machine_type in ('logreg', 'randomforest', 'svm'):
                    #salvando o modelo joblib    
                    print('Salvando o modelo de ', self.machine_type)
                    model_save_folder = self.diretorio + f'/Outputs/Models/sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.joblib'
                    dump(model, model_save_folder)
                
                #salvando o número do último arquivo processado
                print('Salvando o file_index...')
                save_index_to_TXT(filename, f'counter_File_index_sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}', 
                                  '/Outputs/LOG',
                                  diretorio = self.diretorio)

                #salvando os resultados
                print('Salvando o test results...')
                if nn_check is True:
                    test_results[f'sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}']['loss'] = results[0]
                    test_results[f'sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}']['acc'] =  results[1]                
                    save_dic_to_json(self.diretorio + f'/Outputs/Models/sections_{self.section_name}_training_results.json',
                                     test_results)
                
                elif self.machine_type in ('logreg', 'randomforest', 'svm'):
                    test_results[f'sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}']['precision'] =  results[0]
                    test_results[f'sections_{self.section_name}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}']['recall'] =  results[1]
                    save_dic_to_json(self.diretorio + f'/Outputs/Models/sections_{self.section_name}_training_results.json',
                                     test_results)
                    
                #limpando as listas para os novos batchs    
                X_wv_train=[]
                X_wv_test=[]
                X_tv_train=[]
                X_tv_test=[]
                Y_train = []
                Y_test = []
                got_pdf_counter = 0
                batch_counter += 1
                go_to_train = False
                del test_results
                
                if nn_check is True:
                    #deletando o graph da última sessão
                    backend.clear_session()

                if self.vector == 'tv':
                    #fechando o arquivo h5
                    h5.close()                
                    #carregando a matriz DOC_TOPIC
                    h5 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'r')
                    DOC_TOPIC_matrix = h5['data']



class use_machine(object):
    
    def __init__(self, model_name = 'sections_methodolody_conv1d_wv_SW_True_ch_4_rNgram_False', diretorio = None):

        print('( Class: use_machine )')

        import regex as re
        
        self.diretorio = diretorio
        self.model_name = model_name
        self.sent_batch_size = int( re.search(r'ch_[0-9]', self.model_name).captures()[0][ -1 : ] )
        self.filter_stopwords = re.findall ( r'(True|False)', re.search(r'SW_(True|False)', self.model_name).captures()[0][ : ] )[0] in ['True']
        self.replace_Ngrams = re.findall ( r'(True|False)', re.search(r'rNgram_(True|False)', self.model_name).captures()[0][ : ] )[0] in ['True']
        self.machine_type = re.findall(r'(conv1d|conv2d|lstm|logreg|randomforest|svm)', self.model_name)[0]
    
        #print('Machine settings - SW: ', self.filter_stopwords, ' ; rNgram: ', self.replace_Ngrams)


    def set_machine_parameters_to_use(self, wv_model = 'w2vec', wv_matrix_name = 'w2vec_xx_kk_ll'):

        print('( Function: set_machine_parameters_to_use )')
        
        import pandas as pd
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import get_filenames_from_folder

        #caso a máquina seja uma rede neural
        if self.machine_type in ('conv1d', 'conv2d', 'lstm'):

            from keras.models import load_model
            import keras.backend as backend
            #deletando o graph da última sessão
            backend.clear_session()
            
            #definindo os settings dos tokens
            self.wv_model = wv_model
            self.wv_matrix_name = wv_matrix_name
            
            #carregando o WV
            try:
                self.wv = pd.read_csv(self.diretorio + f'/Outputs/WV/{self.wv_matrix_name}.csv', index_col = 0)
                self.wv_emb_dim = len(self.wv.columns)
            except FileNotFoundError:
                print(f'Erro para a entrada wv_name: {self.wv_matrix_name}')
                print('Entradas disponíveis na pasta ~/Outputs/Models/')
                filenames = get_filenames_from_folder(self.diretorio + '/Outputs/WV', file_type = 'csv', print_msg = False)
                for filename in [filename for filename in filenames if filename[ : len('w2vec') ] == 'w2vec' ]:
                    print(filename)
                print('> Abortando a classe: use_machine')
                self.abort_class = True
                return        
            
            #carregando as estatísticas de sentenças
            self.sent_tokens_stats = load_dic_from_json(self.diretorio + '/Outputs/LOG/stats_SENT_TOKENS.json')
            
            #carregando os tokens do IDF
            self.IDF = pd.read_csv(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv', index_col = 0)
            self.tokens_list = self.IDF.index.values

            #self.session = tf.Session()
            #self.graph = tf.get_default_graph()
            #set_session(self.session)

            #carregando a CNN 1D
            self.model = load_model(self.diretorio + f'/Outputs/Models/{self.model_name}.h5')
            print('Section filter encontrado: ', self.model_name)
        
        #caso a máquina seja um modelo do sklearn
        if self.machine_type in ('logred', 'randomforest', 'svm'):
            
            from joblib import load
            
            #carregando o modelo sklearn
            model_folder = self.diretorio + f'/Outputs/Models/{self.model_name}.joblib'
            print('Section filter encontrado: ', self.model_name)            
            self.model = load(model_folder)

    

    def check_sent_in_section_for_ML(self, sent_index, DF = 'sentDF'):

        from FUNCTIONS import load_dic_from_json
        from functions_VECTORS import get_tv_from_sent_index
        import h5py
        
        #carregando a estatísticas dos topic_vectors
        self.tv_stats = load_dic_from_json(self.diretorio + '/Outputs/TFIDF/H5/DOC_TOPIC_MATRIX_STATS.json')
        
        self.text_file = DF        
        
        #print('\nML analyzing sent...') 
        #print('sent_index: ', sent_index)
        #print('Sent: ', self.text_file.loc[ sent_index , 'Sentence'])

        #carregando a matriz DOC_TOPIC
        h5 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'r')
        DOC_TOPIC_matrix = h5['data']

        #coletando o vector topico da sentença
        topic_vector = get_tv_from_sent_index(sent_index, tv_stats = self.tv_stats, DOC_TOPIC_matrix = DOC_TOPIC_matrix, diretorio = self.diretorio)
        
        result_proba = self.model.predict_proba(topic_vector)
        
        return result_proba
        
                
        
    def check_sent_in_section_for_NN(self, sent_index, DF = 'sentDF', n2grams_DF = None, n3grams_DF = None):
                
        import numpy as np
        from functions_TEXTS import concat_DF_sent_indexes
        from functions_TOKENS import find_min_max_token_sent_len        
        from FUNCTIONS import load_dic_from_json

        #carregando a estatísticas dos word_vectors
        self.wv_stats = load_dic_from_json(self.diretorio + '/Outputs/WV/MATRIX_VECTORS_STATS.json')[self.wv_model][self.wv_matrix_name]
    
        self.text_file = DF        
        
        print('NN analyzing sent...') 
        print('sent_index: ', sent_index)
        #print('Sent: ', self.text_file.loc[ sent_index , 'Sentence'])
    
        #obtendo os indexes das sentenças concatenadas (adjacentes ao sent_index)
        concat_sent_indexes = concat_DF_sent_indexes(sent_index, self.sent_batch_size)
    
        #lista para coletar os resultados
        results = []
    
        #varrendo as sentenças concatenadas
        for concat_sent_index in concat_sent_indexes:
            try:
                #concatenando as sentenças    
                concat_word_vectors = self.get_concat_sents_wvs(concat_sent_index, n2grams_DF = n2grams_DF, n3grams_DF = n3grams_DF)
            
                #definido o maior valor permitido de sent_token_len (quantidade de tokens por sentença) o qual será o input da NN
                sent_max_token_len , sent_min_token_len , single_sent_min_token_len = find_min_max_token_sent_len(sent_batch_size = self.sent_batch_size, 
                                                                                                                  machine_type = self.machine_type, 
                                                                                                                  sent_tokens_stats = self.sent_tokens_stats, 
                                                                                                                  filter_stopwords = self.filter_stopwords)
                
                #checando o sent_token_len
                if (sent_min_token_len <= concat_word_vectors.shape[0] <= sent_max_token_len):
                    #fazendo o padding
                    if concat_word_vectors.shape[0] < sent_max_token_len:
                        zero_m = np.zeros((sent_max_token_len - concat_word_vectors.shape[0], 300))
                        concat_word_vectors = np.vstack((concat_word_vectors, zero_m))
                    else:
                        pass
                else:
                    print('concatenated sents outsized (continue)...')
                    continue
                
                #fazendo o reshape
                concat_word_vectors = concat_word_vectors.reshape(1, concat_word_vectors.shape[0], concat_word_vectors.shape[1])
                #fazendo o predict
                #print('sent_wv_shape: ', concat_word_vectors.shape)
                result = self.model.predict(concat_word_vectors)
                results.append(result[0][0])            
            
            #caso algum index esteja fora de range
            except KeyError:
                continue
        
        #print('Results: ', results)
        try:
            return max(results)
        except ValueError:
            return 0

        
    def get_concat_sents_wvs(self, concat_sent_index, n2grams_DF = None, n3grams_DF = None):

        import numpy as np
        from functions_VECTORS import get_wv_from_sentence                
                
        for i in concat_sent_index:
            #print('Getting sentence vectors - sent index: ', i)
            #sentença
            sent = self.text_file.loc[i, 'Sentence']
            
            #colentando os word vectors (scaled) da sentença (section)
            sent_word_vectors = get_wv_from_sentence(sent, 
                                                     self.wv,
                                                     self.wv_stats,
                                                     collection_token_list = self.tokens_list, 
                                                     filter_stopwords = self.filter_stopwords,
                                                     n2grams_DF = n2grams_DF,
                                                     n3grams_DF = n3grams_DF)
            
            #concatenando os word vectors
            #caso haja token na sentença
            if sent_word_vectors.shape[0] > 0:
                try:
                    concat_word_vectors = np.vstack(( concat_word_vectors , sent_word_vectors ))
                except NameError:
                    concat_word_vectors = sent_word_vectors        
                
        return concat_word_vectors



    ##########################################################################################################
    '''
    def find_in_texts(self, 
                      check_sent_regex_pattern = 'z+x?z*',
                      tokens_window_size_min = 10, 
                      tokens_window_size_max = 20,
                      tokens_window_overlap = 2,
                      get_largest_R_values = 10):
        
        print('( Function: find_in_texts )')
        
        import time
        import json    
        import os
        import numpy as np
        from keras.models import load_model
        from FUNCTIONS import filter_tokens
        from FUNCTIONS import get_tokens_from_text
        from FUNCTIONS import get_sent_to_predict
        
        file_list = os.listdir(self.diretorio + '/Outputs/Texts') #lista de arquivos
        #testar se há arquivos no diretório DB
        if len(file_list) == 0:
            print('\nERRO!')
            print('Não há arquivos no diretorio ~/DB.')
            print('Extrair os objetos de texto com o módulo TEXTOBS\n')
            return
        
        #montando a lista de pdfs da coleção
        text_documents = []
        for filename in file_list:
            if filename[ -4 :  ].lower() == 'json':
                text_documents.append(filename[ : -5 ])
        text_documents = sorted(text_documents)
    
        for filename in text_documents[ : 2 ]:
            print(f'\n---------------------------\n{filename}\n---------------------------\n')
            #abrindo os dois arquivos de text (filtrados e não filtrados)
            with open(self.diretorio + f'/Outputs/Texts/{filename}.json') as file:
                text_dic = json.load(file)
                file.close()
            
            #O texto filtrado não está sendo usado.
            with open(self.diretorio + f'/Outputs/Texts_filtered/{filename}.json') as file:
                filtered_text_dic = json.load(file)
                file.close()            

            #aplicando o spacy para o texto
            text_tokens, filtered_text_tokens = get_tokens_from_text(text_dic['Full_text'], filtered_text_dic['Full_text'],
                                                                     collection_token_list = self.tokens_list,
                                                                     filter_stopwords = self.filter_stopwords)
            
            #print(text_tokens[10:30])            
            #print(filtered_text_tokens[10:30])
            
            #tamanho do texto
            text_len = len(filtered_text_tokens)            
            
            #carregando o modelo
            model = load_model(self.diretorio + f'/Outputs/Models/{self.feature}_{self.machine_type.lower()}_{self.vector.lower()}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.h5') #model for sent
            
            #identificando o formato de entrada do modelo (input shape)
            batch_input_shape = model.get_config()['layers'][0]['config']['batch_input_shape']
            print('Model1 input shape: ', batch_input_shape)
            
            #padding model1
            sent_max_token_len = batch_input_shape[1]    

            print('Model1 Max text len (n tokens): ', sent_max_token_len)
            #print('Model2 Max text len (n tokens): ', inner_sent_max_token_len)
    
            #definindo o tamanho dos windows que varrerão o texto
            windows_size = list(range(tokens_window_size_min, tokens_window_size_max, 
                                      int( (tokens_window_size_max - tokens_window_size_min) / tokens_window_overlap )))
            
            #if windows_size[-1] != sent_max_token_len:
            #    windows_size.append(sent_max_token_len)
            print('Windows sizes: ', windows_size)
            
            #definindo os indices do texto correspondentes a cada window
            index_slices = []
            for window in windows_size:
                slices = [ [i, i + window] for i in range(0, text_len, window)]
                if slices[-1][1] > text_len - 1:
                    slices.remove(slices[-1])
                index_slices = index_slices + slices

            #print('\nWindows indexes: ', index_slices)

            #avaliando cada parte do texto com a maquina treinada com sentença
            print('Avaliando no model1...')
            m1_high_results = []
            m1_high_result_indices = []
            for indices in index_slices:
                check_get_sent = get_sent_to_predict(filtered_text_tokens[ indices[0] : indices[1] ], check_sent_regex_pattern = check_sent_regex_pattern)
                if (check_get_sent is True):
                    tokens = filtered_text_tokens[ indices[0] : indices[1] ]
                    #print('(Input) Tokens: ', tokens)
                    #time.sleep(0.1)
                    data = []
                    for token in tokens:
                        #print('Appending: ', token)
                        data.append( self.wv.loc[token].values )                    
                    
                    data=np.array(data)
                    if sent_max_token_len > data.shape[0]:
                        data = np.r_[data , np.zeros([ sent_max_token_len - data.shape[0] , self.wv_emb_dim ]) ]                            
                    #print(data.shape)
                    #avaliando o trecho com o modelo treinado
                    result = model.predict( np.array([data]) )
                    #print(result)                
                    
                    if result[0][0] > 0.60:
                        m1_high_results.append(result[0][0])
                        m1_high_result_indices.append(indices)            
                
                else:
                    continue
            
            largest_R_values_indexes1 = np.array(m1_high_results).argsort()[ :: -1][  : get_largest_R_values ]            
                
            
            #get_largest_R_values = 10
            #largest_R_values_indexes2 = np.array(m2_high_results).argsort()[ :: -1][ : get_largest_R_values ]
            counter = 0
            print('Final results...')
            for index in largest_R_values_indexes1:
                counter += 1
                print('Top m1: ', counter, f' result ({m1_high_results[index]})')
                inner_sent = text_tokens[ m1_high_result_indices[index][0] :  m1_high_result_indices[index][1] ]
                print(inner_sent)
                if counter == 3:
                    break            



      
    def set_train_sentences(self, 
                            sent_batch_size = 100, 
                            n_epochs = 5,
                            vector_type = 'wv', 
                            sentDF_filename = 'train_sents_temperature', 
                            random_sentDF_filename = 'temperature_extracted_mode_semantic_random'):
        
        print('( Function: set_train_sentences )')
        
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        
        import pandas as pd
        import os
        
        self.sent_batch_size = sent_batch_size
        self.n_epochs = n_epochs
        self.vector = vector_type.lower()
        
        cond1 = os.path.exists(self.diretorio + f'/Outputs/DataFrames/{sentDF_filename}.csv')
        cond2 = os.path.exists(self.diretorio + f'/Outputs/Extracted/{random_sentDF_filename}.csv')
        
        if False in (cond1, cond2):
            print('Erro! Sentence DF não encontrado.')
            print('> Abortando função: train_machine.set_train_sentences')
            return

        #ajustando o DF de sentenças
        sent_DF = pd.read_csv(self.diretorio + f'/Outputs/DataFrames/{sentDF_filename}.csv')
        sent_DF_columns = sent_DF.columns
        #colunas: sentenças - indexes de sentenças
        sent_DF = sent_DF[[ sent_DF_columns[3], sent_DF_columns[2], sent_DF_columns[4] ]].copy()
        sent_DF.columns = 'sent', 'sent_index', 'target'
        
        #ajustando o DF de sentenças randômicas
        sent_DF_random = pd.read_csv(self.diretorio + f'/Outputs/Extracted/{random_sentDF_filename}.csv')
        #colunas: sentenças - indexes de sentenças        
        sent_DF_random = sent_DF_random[['rand_sent', 'rand_sent_index']].copy()
        sent_DF_random['target'] = 0
        sent_DF_random.columns = 'sent', 'sent_index', 'target'
        #concatenando
        full_sent_DF = pd.concat([sent_DF, sent_DF_random], axis=0, ignore_index=True)
        #fazendo o shuffle
        self.sentDF = full_sent_DF.sample(frac=1)
        #print(self.sentDF)
        
        del sent_DF
        del sent_DF_random
        
        
        
        
    def train_on_sentences(self):
        
        print('( Function: train_on_sentences )')
        
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import error_incompatible_strings_input
        abort_class = error_incompatible_strings_input('vector', self.vector, ('wv', 'tv', 'wv_tv'), class_name = self.class_name)
        if True in (abort_class, self.abort_class):
            error_print_abort_class(self.class_name)
            return

        import time
        from keras.layers import Dense, Dropout, Flatten, LSTM
        from keras.preprocessing.sequence import pad_sequences
        from keras.optimizers import SGD
        from FUNCTIONS import save_dic_to_json
        from FUNCTIONS import load_dic_from_json
        from functions_TOKENS import get_tokens_len
        from functions_VECTORS import get_wv_from_sentence
        from functions_VECTORS import get_tv_from_sent_index
        from NNs import NN_model
        import numpy as np
        import pandas as pd
        import random

        #caso os Ngrams sejam substituidos
        if (self.replace_Ngrams is True):
            #abrindo o DF com os N2GRAMS para substituir
            n2grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')        
            n2grams_DF.set_index('index', inplace=True)
            try:
                n3grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n3grams_to_replace.csv')
                n3grams_DF.set_index('index', inplace=True)
                
            except FileNotFoundError:
                n3grams_DF = None
        
        else:
            n2grams_DF = None
            n3grams_DF = None

        #determinando o número de samples a treinar
        n_samples = len(self.sentDF.index.values)
        print('Número de instâncias a treinar: ', n_samples)
        
        #definido o maior valor permitido de sent_token_len (quantidade de tokens por sentença) o qual será o input da NN
        sent_tokens_stats = load_dic_from_json(self.diretorio + '/Outputs/LOG/stats_SENT_TOKENS.json')
        if self.machine_type == 'conv1d':
            #para conv1d, o tamanho máximo permitido de tokens por sentença será a média somada de um desvio padrão (1sigma)
            sent_max_token_len = int( (sent_tokens_stats[f'SW_{self.filter_stopwords}']['avg'] + sent_tokens_stats[f'SW_{self.filter_stopwords}']['std'] ) * self.sent_batch_size )
        elif self.machine_type == 'conv2d':
            #para conv2d, o tamanho máximo permitido de tokens por sentença será o quantile de 0.95
            sent_max_token_len = int(sent_tokens_stats[f'SW_{self.filter_stopwords}']['0.95_quantile'])
        
        #definindo os parâmetros da NN
        NN_parameters_dic = {}        
        
        #general parameters
        NN_parameters_dic['machine_type'] = self.machine_type
        NN_parameters_dic['feature'] = self.feature
        NN_parameters_dic['section_name'] = None
        NN_parameters_dic['vector_type'] = self.vector
        NN_parameters_dic['replace_Ngrams'] = self.replace_Ngrams
        NN_parameters_dic['filter_stopwords'] = self.filter_stopwords
        NN_parameters_dic['batch_size'] = 50
        NN_parameters_dic['n_epochs'] = self.n_epochs

        #NN inputs
        NN_parameters_dic['input_shape'] = {}
        NN_parameters_dic['input_shape']['wv'] = (sent_max_token_len, self.wv_emb_dim) #word vector
        NN_parameters_dic['input_shape']['tv'] = (1, self.wv_emb_dim) #topic vector
        
        #NN parameters
        NN_parameters_dic['activation'] = {}
        NN_parameters_dic['activation']['conv1d'] = {}
        NN_parameters_dic['activation']['conv1d_lstm'] = {}
        NN_parameters_dic['activation']['lstm'] = {}
        NN_parameters_dic['dropout'] = {}
        NN_parameters_dic['dropout']['conv1d'] = {}
        NN_parameters_dic['dropout']['conv1d_lstm'] = {}
        NN_parameters_dic['dropout']['lstm'] = {}
        NN_parameters_dic['filters'] = {}
        NN_parameters_dic['filters']['conv1d'] = {}
        NN_parameters_dic['filters']['conv1d_lstm'] = {}
        NN_parameters_dic['kernel_size'] = {}
        NN_parameters_dic['kernel_size']['conv1d'] = {}
        NN_parameters_dic['kernel_size']['conv1d_lstm'] = {}
        NN_parameters_dic['loss'] = {}
        NN_parameters_dic['loss']['conv1d'] = {}
        NN_parameters_dic['loss']['conv1d_lstm'] = {}
        NN_parameters_dic['loss']['lstm'] = {}
        NN_parameters_dic['optimizer'] = {}
        NN_parameters_dic['optimizer']['conv1d'] = {}
        NN_parameters_dic['optimizer']['conv1d_lstm'] = {}
        NN_parameters_dic['optimizer']['lstm'] = {}
        NN_parameters_dic['strides'] = {}
        NN_parameters_dic['strides']['conv1d'] = {}
        NN_parameters_dic['strides']['conv1d_lstm'] = {}
        NN_parameters_dic['units'] = {}
        NN_parameters_dic['units']['conv1d'] = {}
        NN_parameters_dic['units']['conv1d_lstm'] = {}
        NN_parameters_dic['units']['lstm'] = {}
        
        
        #model conv1d
        #-----------------------        
        #l1 - conv1d
        NN_parameters_dic['activation']['conv1d']['l1'] = 'elu'
        NN_parameters_dic['filters']['conv1d']['l1'] = 250
        NN_parameters_dic['kernel_size']['conv1d']['l1'] = 3
        NN_parameters_dic['strides']['conv1d']['l1'] = 1
        
        #l2 - dense
        NN_parameters_dic['activation']['conv1d']['l2'] = 'sigmoid'
        NN_parameters_dic['dropout']['conv1d']['l2'] = 0.2    
        NN_parameters_dic['units']['conv1d']['l2'] = 50
        
        #l3 - dense
        NN_parameters_dic['activation']['conv1d']['l3'] = 'sigmoid'
        NN_parameters_dic['units']['conv1d']['l3'] = 1        
        
        #compile
        NN_parameters_dic['loss']['conv1d'] = 'binary_crossentropy'
        opt = SGD(learning_rate=0.1, nesterov=True, momentum=0.9)
        NN_parameters_dic['optimizer']['conv1d'] = opt
        
        
        #model lstm
        #-----------------------        
        #l1 - lstm
        NN_parameters_dic['activation']['lstm']['l1'] = 'sigmoid'    
        NN_parameters_dic['units']['lstm']['l1'] = 50
        NN_parameters_dic['dropout']['lstm']['l1'] = 0.2
        
        #l2 - dense
        NN_parameters_dic['activation']['lstm']['l2'] = 'sigmoid'
        NN_parameters_dic['units']['lstm']['l2'] = 1
                
        #compile
        NN_parameters_dic['loss']['lstm'] = 'binary_crossentropy'
        opt = SGD(learning_rate=0.1, nesterov=True, momentum=0.9)
        NN_parameters_dic['optimizer']['lstm'] = opt


        #model #conv1d_lstm
        #-----------------------        
        #l1 - convid
        NN_parameters_dic['activation']['conv1d_lstm']['l1'] = 'relu'
        NN_parameters_dic['filters']['conv1d_lstm']['l1'] = 250
        NN_parameters_dic['kernel_size']['conv1d_lstm']['l1'] = 3
        NN_parameters_dic['strides']['conv1d_lstm']['l1'] = 1
        
        #l2 - dense
        NN_parameters_dic['activation']['conv1d_lstm']['l2'] = 'sigmoid'
        NN_parameters_dic['dropout']['conv1d_lstm']['l2'] = 0.2
        NN_parameters_dic['units']['conv1d_lstm']['l2'] = 250        
        
        #l3 - dense
        NN_parameters_dic['activation']['conv1d_lstm']['l3'] = 'sigmoid'
        NN_parameters_dic['dropout']['conv1d_lstm']['l3'] = 0.2
        NN_parameters_dic['units']['conv1d_lstm']['l3'] = 50
        
        #l4 - lstm
        NN_parameters_dic['activation']['conv1d_lstm']['l4'] = 'sigmoid'
        NN_parameters_dic['dropout']['conv1d_lstm']['l4'] = 0.2
        NN_parameters_dic['units']['conv1d_lstm']['l4'] = 250

        #l5 - dense
        NN_parameters_dic['activation']['conv1d_lstm']['l5'] = 'sigmoid'
        NN_parameters_dic['dropout']['conv1d_lstm']['l5'] = 0.2
        NN_parameters_dic['units']['conv1d_lstm']['l5'] = 50
        
        #l6 - dense (concateando os outputs de l1 + l2)
        NN_parameters_dic['activation']['conv1d_lstm']['l6'] = 'sigmoid'
        NN_parameters_dic['dropout']['conv1d_lstm']['l6'] = 0.2
        NN_parameters_dic['units']['conv1d_lstm']['l6'] = 10

        #l7- dense
        NN_parameters_dic['activation']['conv1d_lstm']['l7'] = 'sigmoid'
        NN_parameters_dic['units']['conv1d_lstm']['l7'] = 1
        
        #compile
        NN_parameters_dic['loss']['conv1d_lstm'] = 'binary_crossentropy'
        opt = SGD(learning_rate=0.1, nesterov=True, momentum=0.9)
        NN_parameters_dic['optimizer']['conv1d_lstm'] = opt
        
        
        #definindo a NN
        NN = NN_model()
        NN.set_parameters(NN_parameters_dic)
        model = NN.get_model()        
                    
        #treinado os modelos com batch de sentenças
        go_to_train = False
        total_sent_counter = 0
        got_sent_counter = 0
        batch_counter= 1
        X_wv_train=[]
        X_wv_test=[]
        X_tv_train=[]
        X_tv_test=[]
        Y_train = []
        Y_test = []
        print('Preparing data to train/test...')
        #varrendo as sentenças
        for i in self.sentDF.index:
                        
            #sentença
            sent = self.sentDF.loc[ i , 'sent']
            #index da sentença
            sent_index = self.sentDF.loc[ i , 'sent_index']
            #target
            target = self.sentDF.loc[ i , 'target']
            
            #colentando os word vectors da sentença
            word_vector = get_wv_from_sentence(sent, 
                                               self.wv, 
                                               collection_token_list = self.tokens_list, 
                                               filter_stopwords = self.filter_stopwords,
                                               n2grams_DF = n2grams_DF,
                                               n3grams_DF = n3grams_DF)
            
            #coletando o vector topico da sentença
            topic_vector = get_tv_from_sent_index(sent_index)
            
            #separando entre train e test
            
            if random.random() > 0.2:
                X_wv_train.append( word_vector )
                X_tv_train.append( topic_vector)
                Y_train.append( target )
                
            else:
                X_wv_test.append( word_vector )
                X_tv_test.append( topic_vector)
                Y_test.append( target )
            
            #print(sent)
            #time.sleep(0.5)                                    
            got_sent_counter += 1
            total_sent_counter += 1                                                            

            if total_sent_counter % 50 == 0:
                print('Processing sent ', total_sent_counter, '(Total of ', n_samples, ' )' )
            
            if total_sent_counter == n_samples or got_sent_counter == self.sent_batch_size:
                go_to_train = True
            
            if (go_to_train is True):
                    
                print(f'Sentence batch: {batch_counter}')
                #treinando o modelo 1
                #com padding
                print('Training model 1...')
                #array dos conjunto de dados de word-vectors das sentenças
                X_wv_train = pad_sequences(X_wv_train, maxlen = sent_max_token_len, padding='post')
                X_wv_test = pad_sequences(X_wv_test, maxlen = sent_max_token_len, padding='post')
                
                #array dos conjunto de dados de topic-vectors das sentenças
                X_tv_train = np.array(X_tv_train)
                X_tv_test = np.array(X_tv_test)
                #reshaping os topic arrays para ( instâncias, 1, tv_dims)
                X_tv_train = X_tv_train.reshape(X_tv_train.shape[0], 1, X_tv_train.shape[1])
                X_tv_test = X_tv_test.reshape(X_tv_test.shape[0], 1, X_tv_test.shape[1])
                
                #target array
                Y_train = np.array(Y_train)
                Y_test = np.array(Y_test)
                                
                #training
                #caso seja usado os word vectors
                if self.machine_type == 'conv1d' and self.vector == 'tv':
                    print('Erro! O vector "tv" não é compatível com a máquina "conv1d"')
                    print('> Abortando função: train_machine.train_on_sentences')
                    return
                
                elif self.machine_type != 'conv1d_lstm' and self.vector == 'wv':

                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_train.shape)
                    print('Y_train shape: (n_samples, )', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = X_wv_train , y = Y_train, epochs=self.n_epochs, batch_size = NN_parameters_dic['batch_size'], verbose=0)

                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')                    
                    loss_and_metrics = model.evaluate(x = X_wv_test , y = Y_test, batch_size = NN_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {loss_and_metrics[0]}, Acc: {loss_and_metrics[1]}')

                #caso seja usado os topic vectors
                elif self.machine_type != 'conv1d_lstm' and self.vector == 'tv':
                    
                    print('Training data shape:')
                    print('X_tv_train shape (n_samples, tv_dim): ', X_tv_train.shape)
                    print('Y_train shape: (n_samples, )', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = X_tv_train , y = Y_train, epochs=self.n_epochs, batch_size = NN_parameters_dic['batch_size'], verbose=0)

                    print('Training data shape:')
                    print('X_tv_test shape (n_samples, tv_dim): ', X_tv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')
                    loss_and_metrics = model.evaluate(x = X_tv_test, y = Y_test, batch_size = NN_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {loss_and_metrics[0]}, Acc: {loss_and_metrics[1]}')
                
                #caso seja usado ambos os word vectors e os topic vectors
                elif self.machine_type == 'conv1d_lstm' and self.vector == 'wv_tv':

                    print('Training data shape:')
                    print('X_wv_train shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_train.shape)
                    print('X_tv_train shape (n_samples, tv_dim): ', X_tv_train.shape)
                    print('Y_train shape: (n_samples, )', Y_train.shape)
                    print('Fitting...')
                    model.fit(x = [X_wv_train, X_tv_train] , y = Y_train, epochs=self.n_epochs, batch_size = NN_parameters_dic['batch_size'], verbose=0)

                    print('Training data shape:')
                    print('X_wv_test shape (n_samples, max_n_tokens_len, wv_dim): ', X_wv_test.shape)
                    print('X_tv_test shape (n_samples, tv_dim): ', X_tv_test.shape)
                    print('Y_test shape: (n_samples, )', Y_test.shape)
                    print('Testing...')
                    loss_and_metrics = model.evaluate(x = [X_wv_test, X_tv_test], y = Y_test, batch_size = NN_parameters_dic['batch_size'])
                    print(f'Evaluate results - Loss: {loss_and_metrics[0]}, Acc: {loss_and_metrics[1]}')
                                
                #salvando o modelo h5            
                model_save_folder = self.diretorio + f'/Outputs/Models/sentences_{self.feature}_{self.machine_type}_{self.vector}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.h5'
                model.save(model_save_folder)
                
                #salvando o número do último arquivo processado
                #save_index_to_TXT(sent_number, f'counter_File_index_w2vec_{train_approach}_SW{filter_stopwords}_WS_{words_window_size}', '/Outputs/WV')                    
        
                #limpando as listas para os novos batchs    
                X_wv_train=[]
                X_wv_test=[]
                X_tv_train=[]
                X_tv_test=[]
                Y_train = []
                Y_test = []
                got_sent_counter = 0
                batch_counter += 1
                go_to_train = False
                '''