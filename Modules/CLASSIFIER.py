#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import spacy
from nltk.corpus import stopwords
import numpy as np
import h5py
import os
import time

from FUNCTIONS import get_filenames_from_folder
from FUNCTIONS import save_dic_to_json
from FUNCTIONS import get_tag_name
from FUNCTIONS import get_file_batch_index_list
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import update_log
from FUNCTIONS import load_log_info

from functions_TOKENS import get_tokens_indexes
from functions_TOKENS import get_tokens_from_sent

from functions_VECTORS import load_dense_matrix, load_sparse_csr_matrix
from functions_VECTORS import save_dense_matrix, save_sparse_csr_matrix


class lda(object):

    def __init__(self, diretorio = None):

        print('\n( Class: LDA classifier )')

        self.diretorio = diretorio
        
        if not os.path.exists(self.diretorio + '/Outputs/models'):
            print('Creating folder ', self.diretorio + '/Outputs/models')
            os.makedirs(self.diretorio + '/Outputs/models')



    def set_classifier(self, doc_type: str = None, n_topics: int = 100, alpha:float = 0.0, beta:float = 0.0, file_batch_size: int = 0, use_sparse_matrix = False):


        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords_list = stopwords.words('english')
        self.n_topics = n_topics
        self.doc_type = doc_type.lower()
        self.alpha = alpha
        self.beta = beta
        self.use_sparse_matrix = use_sparse_matrix
        if self.use_sparse_matrix is True:
            self.matrix_file_format = '.npz'
            self.save_matrix = save_sparse_csr_matrix
            self.load_matrix = load_sparse_csr_matrix
        else:
            self.matrix_file_format = '.npy'
            self.save_matrix = save_dense_matrix
            self.load_matrix = load_dense_matrix

        #definições do modelo
        self.lda_definitions = f'{self.doc_type}_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}'

        #checando os caminhos das pastas das matrizes
        if not os.path.exists(self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_token_topic'):
            os.makedirs(self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_token_topic')

        #carregando o df dos IDF para pegar os tokens
        IDF = pd.read_csv(self.diretorio + f'/Outputs/tfidf/idf.csv', index_col = 0)
        self.all_tokens_array = IDF.index.values
        self.n_tokens = len(self.all_tokens_array)
        print('Tokens dictionary size: ', self.n_tokens)

        #getting the number of articles
        self.articles_filenames = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')
        self.n_articles = len(self.articles_filenames)
        print('Número de artigos: ', self.n_articles)

        #carregando o LOG file
        create_new_log_entry = False
        if os.path.exists(self.diretorio + '/Outputs/log/lda_batches_log.json'):
            self.log_batches = load_dic_from_json(self.diretorio + '/Outputs/log/lda_batches_log.json')
            try:
                self.log_batches[self.lda_definitions]['batches']
                print('O LDA_LOG para essa configuração já foi extraído.')
                print('Carregando o arquivo em ' + self.diretorio + '/Outputs/log/lda_batches_log.json')
            
            except KeyError:
                print('Adding new LOG for LDA batches...')
                create_new_log_entry = True
        else:
            print('Getting LOG for LDA batches...')
            self.log_batches = {}
            create_new_log_entry = True

        if create_new_log_entry is True:

            self.log_batches[self.lda_definitions] = {}
            self.log_batches[self.lda_definitions]['batches'] = {}
            
            #caso o valor de file_batch_size seja incorreto
            if file_batch_size > self.n_articles:
                print('ERRO!')
                print('O valor inserido para o "file_batch_size" é maior que o número total de arquivos em ~/Outputs/sents_filtered')
                return
            
            #caso o processamento não for feito em batch
            elif file_batch_size == 0:
                file_batch_size = self.n_articles
            
            #criando os batches de arquivos
            batch_indexes = get_file_batch_index_list(self.n_articles, file_batch_size)
            
            #varrendo todos os slices
            #número do batch
            batch_counter_number = 0
            for slice_begin, slice_end in batch_indexes:

                #dicionário para cada batch
                batch_counter_number += 1
                tagged_batch_counter_number = get_tag_name(batch_counter_number, prefix = '')
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number] = {}
                print('Processing batch: ', tagged_batch_counter_number, '; indexes: ', slice_begin, slice_end, '; files: ', self.articles_filenames[slice_begin], self.articles_filenames[slice_end])
                
                #contador de sentenças do batch
                count_sents_batch = 0
                for i in range(slice_begin, slice_end + 1):
                    
                    filename = self.articles_filenames[i]
                    DFsents = pd.read_csv(self.diretorio + f'/Outputs/sents_filtered/{filename}.csv', index_col = 0)
                    count_sents_batch += len(DFsents.index)
                    del DFsents
                
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['first_file_index'] = slice_begin
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['last_file_index'] = slice_end
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['n_sents'] = count_sents_batch            
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['n_articles'] = slice_end - slice_begin + 1

            self.log_batches[self.lda_definitions]['total_sents'] = load_dic_from_json(self.diretorio + '/Outputs/log/filtered_sents.json')['sentence_counter']
            self.log_batches[self.lda_definitions]['total_articles'] = self.n_articles
            
            #salvando o log de slices para calcular a matriz TFIDF
            save_dic_to_json(self.diretorio + '/Outputs/log/lda_batches_log.json', self.log_batches)



    def get_initial_doc_token_topic_m(self):


        print('> function get_initial_doc_token_topic_m')

        #varrendo os batches e atribuindo tópicos iniciais na matrix doc_token_topic
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            path_doc_token_topic = self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_token_topic/doc_{self.doc_type}_token_topic_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'

            #checando se a matriz token_topic_count já foi criada para esse batch
            if not os.path.exists(path_doc_token_topic + self.matrix_file_format):
                
                #matriz com atribuição de tópico para cada doc e cada token 
                n_matrix_lines = self.log_batches[self.lda_definitions]['batches'][batch][f'n_{self.doc_type}']
                doc_token_topic_m = np.zeros((n_matrix_lines, self.n_tokens), dtype=int)

                #varrendos os artigos
                row_n = 0
                article_token_list = []
                for filename in self.articles_filenames[ self.log_batches[self.lda_definitions]['batches'][batch]['first_file_index'] : self.log_batches[self.lda_definitions]['batches'][batch]['last_file_index'] + 1 ]:
                    
                    #abrindo o csv com as sentenças do artigo
                    sentDF = pd.read_csv(self.diretorio + '/Outputs/sents_filtered/' + f'{filename}.csv', index_col = 0)

                    #analisando cada sentença
                    for index in sentDF.index:
                        
                        sent = sentDF.loc[index, 'Sentence']
                        sent_tokens = get_tokens_from_sent(sent, tokens_list_to_filter = self.all_tokens_array, stopwords_list_to_remove = self.stopwords_list, spacy_tokenizer = self.nlp)
                        
                        if self.doc_type == 'sents':
                            
                            #coletando os indexes de todos os tokens considerando a ordem da IDF.csv
                            n_sent_tokens = len(sent_tokens)
                            #print('\nsent_tokens ', sent_tokens)
                            sent_tokens_indexes = get_tokens_indexes(sent_tokens, all_token_array = self.all_tokens_array)
                            #print('sent_tokens_indexes ', sent_tokens_indexes)
                            #atribuindo um valor de um tópico randômico para cada token de cada doc
                            row_t = np.zeros(self.n_tokens, dtype=int)
                            random_topics = get_initial_random_topics_for_tokens(n_sent_tokens, self.n_topics)
                            row_t[sent_tokens_indexes] = random_topics
                            #print('row_t ', row_t)
                            doc_token_topic_m[ row_n, : ] = row_t
                            #print('doc_token_topic_m[ row_n, : ] ', doc_token_topic_m[ row_n, : ])
                            row_n += 1
                            
                        if self.doc_type == 'articles':                    
                            
                            #coletando todos os tokens que estão no artigo
                            for token in sent_tokens:
                                if token not in article_token_list:
                                    article_token_list.append(token)
                    
                    if self.doc_type == 'articles':            
                        
                        #coletando os indexes de todos os tokens considerando a ordem da IDF.csv
                        n_articles_tokens = len(article_token_list)
                        #print('\nn_articles_tokens ', n_articles_tokens)
                        articles_tokens_indexes = get_tokens_indexes(article_token_list, all_token_array = self.all_tokens_array)                        
                        #print('articles_tokens_indexes ', articles_tokens_indexes)
                        #atribuindo um valor de um tópico randômico para cada token de cada doc
                        row_t = np.zeros(self.n_tokens, dtype=int)
                        row_t[articles_tokens_indexes] = get_initial_random_topics_for_tokens(n_articles_tokens, self.n_topics)
                        doc_token_topic_m[ row_n, : ] = row_t
                        #print('random_topics', random_topics)
                        #print('articles_tokens_indexes', articles_tokens_indexes)
                        #print('row_t', row_t)
                        #print(f'doc_token_topic_m[ {row_n}, : ]', doc_token_topic_m[ row_n, : ])
                        #time.sleep(1)
                        row_n += 1

                #salvando a matrix doc_token_topic
                self.save_matrix(path_doc_token_topic, doc_token_topic_m)
                print(f'  Criada matrix doc_{self.doc_type}_token_topic com dimensão: ', doc_token_topic_m.shape)
                        
                #apagando a matrix
                del doc_token_topic_m



    def count_doc_topics(self):
        
        import time

        print('> count_doc_topics')

        #varrendo os batches        
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            #matriz com contagem de tópico para cada doc
            n_matrix_lines = self.log_batches[self.lda_definitions]['batches'][batch][f'n_{self.doc_type}']
            doc_topic_counts_m = np.zeros((n_matrix_lines, self.n_topics), dtype=int)

            #carregando a matrix doc_token_topic
            path_doc_token_topic = self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_token_topic/doc_{self.doc_type}_token_topic_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
            doc_token_topic_m = self.load_matrix(path_doc_token_topic)

            #varrendo os docs
            for i in range(doc_token_topic_m.shape[0]):
                
                #contando o número de tópicos para cada document
                if self.use_sparse_matrix is True:
                    doc_token_topic_array = np.ravel(doc_token_topic_m[ i , : ])
                else:
                    doc_token_topic_array = doc_token_topic_m[ i , : ]

                doc_topic_array_with_topics = doc_token_topic_array[ np.nonzero( doc_token_topic_array )[0] ]
                topics_found, counts = np.unique(doc_topic_array_with_topics, return_counts=True)
                #faz-se essa subtração pois o primeiro tópico (1) tem index 0
                topics_index = topics_found.astype(int) - 1    
                doc_topic_counts_m[ ( [i] * len(topics_index) , topics_index )] = counts
                #time.sleep(1)

            #salvando as matrizes em scipy sparse
            path_doc_topic_counts = self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_topic_counts_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
            self.save_matrix(path_doc_topic_counts, doc_topic_counts_m)
            #print(f'  Criada matrix doc_{self.doc_type}_topic_counts com dimensão: ', doc_topic_counts_m.shape)
            
            del doc_topic_counts_m



    def count_token_topics(self):

        import time

        print('> count_token_topics')
    
        #matriz com contagem de tópico para cada token
        token_topic_counts_m = np.zeros((self.n_tokens, self.n_topics), dtype=int)

        #varrendo os batches        
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            #carregando a matrix doc_token_topic
            path_doc_token_topic = self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_token_topic/doc_{self.doc_type}_token_topic_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
            doc_token_topic_m = self.load_matrix(path_doc_token_topic)

            #varrendo os tokens
            for j in range(doc_token_topic_m.shape[1]):

                #contando o número de tópicos para cada token
                if self.use_sparse_matrix is True:
                    doc_token_topic_array = np.ravel(doc_token_topic_m[ : , j ])
                else:
                    doc_token_topic_array = doc_token_topic_m[ : , j ]
                
                token_topic_array_with_topics = doc_token_topic_array[ np.nonzero( doc_token_topic_array )[0] ]
                topics_found, counts = np.unique(token_topic_array_with_topics, return_counts=True)
                #faz-se essa subtração pois o primeiro tópico (1) tem index 0
                topics_index = topics_found.astype(int) - 1
                token_topic_counts_m[ ([j] * len(topics_index) , topics_index)] += counts
                #time.sleep(1)

        #salvando a matrix token_topic_counts concatenada
        path_token_topic_counts = self.diretorio + f'/Outputs/models/lda_token_topic_counts_{self.lda_definitions}'
        self.save_matrix(path_token_topic_counts, token_topic_counts_m)
        #print(f'  Criada matrix token_{self.doc_type}_topic_counts com dimensão: ', token_topic_counts_m.shape)
    
        #salvando uma matrix token_topic_counts para cada batch para ser trabalhada em paralelo
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():
            path_token_topic_counts = self.diretorio + f'/Outputs/models/lda_token_topic_counts_{self.lda_definitions}_{batch}'
            self.save_matrix(path_token_topic_counts, token_topic_counts_m)
            
        del token_topic_counts_m



    def final_matrices_processing(self):

        print(f'Concatenando a matriz {self.doc_type}_topics_full (H5 file)...')
        h5_doc_topic = h5py.File(self.diretorio + f'/Outputs/models/lda_{self.lda_definitions}_full_matrix.h5', 'w')
        h5_doc_topic.create_dataset('data', shape=(self.log_batches[self.lda_definitions][f'total_{self.doc_type}'], self.n_topics), dtype=np.float64)
        doc_topic_counts_m_full = h5_doc_topic['data']
        
        doc_counter = 0
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            #carregando as matrizes doc_topic por batch
            path_doc_topic_counts = self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_topic_counts_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
            
            if self.use_sparse_matrix is True:
                doc_topic_counts_m = load_sparse_csr_matrix(path_doc_topic_counts)
            
            else:
                doc_topic_counts_m = load_dense_matrix(path_doc_topic_counts)

            first_index = doc_counter
            last_index = doc_counter + self.log_batches[self.lda_definitions]['batches'][batch][f'n_{self.doc_type}']
            doc_topic_counts_m_full[ first_index : last_index ] = doc_topic_counts_m
            doc_counter += self.log_batches[self.lda_definitions]['batches'][batch][f'n_{self.doc_type}']
        
        h5_doc_topic.close()
        del doc_topic_counts_m_full

        
        #apagando as matrizes temporárias dos batches
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            #carregando as matrizes doc_topic por batch
            path_doc_topic_counts = self.diretorio + f'/Outputs/models/lda_doc_{self.doc_type}_topic_counts_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
            path_token_topic_counts = self.diretorio + f'/Outputs/models/lda_token_topic_counts_{self.doc_type}_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'

            if os.path.exists(path_doc_topic_counts + '.npz'):
                os.remove(path_doc_topic_counts + '.npz')

            if os.path.exists(path_doc_topic_counts + '.npy'):
                os.remove(path_doc_topic_counts + '.npy')

            if os.path.exists(path_token_topic_counts + '.npz'):
                os.remove(path_token_topic_counts + '.npz')

            if os.path.exists(path_token_topic_counts + '.npy'):
                os.remove(path_token_topic_counts + '.npy')



    def start_lda(self):

        #carregando o número da iteração
        n = load_log_info(log_name = f'iter_n_{self.lda_definitions}', logpath = self.diretorio + '/Outputs/log/lda_iter_log.json')
        if n is not None: 
            self.iter_n = int(n)
            print('\n> Iteração já realizadas: ', self.iter_n)
            self.count_doc_topics()
            self.count_token_topics()

        else:
            #gerando as matrizes iniciais
            self.get_initial_doc_token_topic_m()
            self.count_doc_topics()
            self.count_token_topics()
            self.iter_n = 0
            print('\n> Iteração já realizadas: ', self.iter_n)

    

    def run_lda(self, iterations = 10):

        setup_list = self.get_parallel_setup()
        while self.iter_n < iterations:
            start = time.time()
            [gibbs_sampling(setup) for setup in setup_list]
            self.count_token_topics()
            self.update_iter_n()
            end = time.time()
            print(' runtime: ', end-start)
        
        self.export_token_topics_to_csv()
        self.final_matrices_processing()



    def get_parallel_setup(self):

        setup_list_for_parallelization = []

        for batch in self.log_batches[self.lda_definitions]['batches'].keys():
            setup_parameters = {}
            setup_parameters['batch'] = batch
            setup_parameters['diretorio'] = self.diretorio
            setup_parameters['doc_type'] = self.doc_type
            setup_parameters['n_topics'] = self.n_topics
            setup_parameters['alpha'] = self.alpha
            setup_parameters['beta'] = self.beta
            setup_parameters['n_tokens'] = self.n_tokens
            setup_parameters['use_sparse_matrix'] = self.use_sparse_matrix
            setup_list_for_parallelization.append(setup_parameters)

        return setup_list_for_parallelization



    def update_iter_n(self):

        #salvando o número da iteração
        self.iter_n += 1
        print('\n> Iteração número: ', self.iter_n, ' realizada.')
        update_log(log_names = [f'iter_n_{self.lda_definitions}'], entries = [self.iter_n], logpath = self.diretorio + '/Outputs/log/lda_iter_log.json')



    def export_token_topics_to_csv(self, n_word_to_represent_topic = 15):


        df = pd.DataFrame(index = range(1, self.n_topics + 1), columns = ['tokens'], dtype=object)

        #carregando a matrix token_topic_counts concatenada
        path_token_topic_counts = self.diretorio + f'/Outputs/models/lda_token_topic_counts_{self.lda_definitions}'
        token_topic_counts_m = self.load_matrix(path_token_topic_counts)

        print('> Exportando a relação token_topics em csv...')
        for topic_i in range(self.n_topics):
            
            #print('\n')
            reversed_arg_sorted_counts_array = np.ravel(token_topic_counts_m[ : , topic_i]).argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_arg_sorted_counts_array[ : n_word_to_represent_topic]:
                token = self.all_tokens_array[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            df.loc[ topic_i + 1 , 'tokens'] = tokens_str
        
        tagged_niter = get_tag_name(self.iter_n, prefix='')
        df.to_csv(self.diretorio + f'/Outputs/models/lda_token_topic_{self.lda_definitions}_niter_{tagged_niter}.csv')



class lsa(object):
    
    def __init__(self, diretorio = None):

        print('\n( Class: LSA classifier )')

        self.diretorio = diretorio
        
        if not os.path.exists(self.diretorio + '/Outputs/models'):
            print('Creating folder ', self.diretorio + '/Outputs/models')
            os.makedirs(self.diretorio + '/Outputs/models')
        
        #abrindo as matrizes doc_topics
        if not os.path.exists(self.diretorio + f'/Outputs/wv/wv_sents_svd_truncated_svd.csv'):
            print('ERRO!')
            print('Matrix lsa_token_sent_topic não encontrada!')
            print('Abortando a classe...')
            return
        
        else:
            self.lsa_token_sent_topic_m = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_svd_truncated_svd.csv', index_col = 0)

        if not os.path.exists(self.diretorio + f'/Outputs/wv/wv_articles_svd_truncated_svd.csv'):
            print('ERRO!')
            print('Matrix lsa_token_article_topic não encontrada!')
            print('Abortando a classe...')
            return
        
        else:
            self.lsa_token_article_topic_m = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_articles_svd_truncated_svd.csv', index_col = 0)



    def export_token_topics_to_csv(self, n_word_to_represent_topic = 15):

        print('> Exportando a relação lsa_token_topics em csv...')

        n_topics = self.lsa_token_sent_topic_m.shape[1]

        lsa_token_sent_topic_pos = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)
        lsa_token_article_topic_pos = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)
        lsa_token_sent_topic_neg = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)
        lsa_token_article_topic_neg = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)


        #eliminando os valores negativos das matrizes (para encontrar os tokens com influcencia positivas nos tópicos)
        self.lsa_token_sent_topic_m_copy = self.lsa_token_sent_topic_m.values.copy()
        self.lsa_token_article_topic_m_copy = self.lsa_token_article_topic_m.values.copy()
        
        self.lsa_token_sent_topic_m_copy[ self.lsa_token_sent_topic_m_copy < 0 ] = 0
        self.lsa_token_article_topic_m_copy[ self.lsa_token_article_topic_m_copy < 0 ] = 0

        for topic_i in range(n_topics):
            
            #arg sorting para os maiores valores na matrix sent_topic
            reversed_arg_sorted_array_sent = self.lsa_token_sent_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_arg_sorted_array_sent[ : n_word_to_represent_topic]:
                token = self.lsa_token_sent_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_sent_topic_pos.loc[ topic_i + 1 , 'tokens'] = tokens_str


            reversed_sorted_array_article = self.lsa_token_article_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_sorted_array_article[ : n_word_to_represent_topic]:
                token = self.lsa_token_article_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_article_topic_pos.loc[ topic_i + 1 , 'tokens'] = tokens_str


        #eliminando os valores positivos das matrizes (para encontrar os tokens com influcencia negativa nos tópicos)
        self.lsa_token_sent_topic_m_copy = self.lsa_token_sent_topic_m.values.copy()
        self.lsa_token_article_topic_m_copy = self.lsa_token_article_topic_m.values.copy()

        self.lsa_token_sent_topic_m_copy[ self.lsa_token_sent_topic_m_copy > 0 ] = 0
        self.lsa_token_article_topic_m_copy[ self.lsa_token_article_topic_m_copy > 0 ] = 0
        
        #convertendo os valores negativos para positivos
        self.lsa_token_sent_topic_m_copy = self.lsa_token_sent_topic_m_copy * -1
        self.lsa_token_article_topic_m_copy = self.lsa_token_article_topic_m_copy * 1

        for topic_i in range(n_topics):
            
            #arg sorting para os maiores valores na matrix sent_topic
            reversed_arg_sorted_array_sent = self.lsa_token_sent_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_arg_sorted_array_sent[ : n_word_to_represent_topic]:
                token = self.lsa_token_sent_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_sent_topic_neg.loc[ topic_i + 1 , 'tokens'] = tokens_str


            reversed_sorted_array_article = self.lsa_token_article_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_sorted_array_article[ : n_word_to_represent_topic]:
                token = self.lsa_token_article_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_article_topic_neg.loc[ topic_i + 1 , 'tokens'] = tokens_str

        
        lsa_token_sent_topic_pos.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_sent_ntopics_{n_topics}_pos.csv')
        lsa_token_article_topic_pos.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_articles_ntopics_{n_topics}_pos.csv')
        lsa_token_sent_topic_neg.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_sent_ntopics_{n_topics}_neg.csv')
        lsa_token_article_topic_neg.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_articles_ntopics_{n_topics}_neg.csv')



def get_initial_random_topics_for_tokens(n_tokens, n_topics):

    #np.random.seed(42)
    return np.random.randint(1, n_topics + 1, size = n_tokens)



def gibbs_sampling(setup_parameters):

    batch = setup_parameters['batch']
    diretorio = setup_parameters['diretorio']
    doc_type = setup_parameters['doc_type']
    n_topics = setup_parameters['n_topics']
    alpha = setup_parameters['alpha']
    beta = setup_parameters['beta']
    n_tokens = setup_parameters['n_tokens']
    use_sparse_matrix = setup_parameters['use_sparse_matrix']

    print('  processin batch: ', batch, flush = True)
    
    #carregando as matrizes
    path_doc_token_topic = diretorio + f'/Outputs/models/lda_doc_{doc_type}_token_topic/doc_{doc_type}_token_topic_ntopics_{n_topics}_a_{alpha}_b_{beta}_{batch}'
    path_doc_topic_counts = diretorio + f'/Outputs/models/lda_doc_{doc_type}_topic_counts_ntopics_{n_topics}_a_{alpha}_b_{beta}_{batch}'
    path_token_topic_counts = diretorio + f'/Outputs/models/lda_token_topic_counts_{doc_type}_ntopics_{n_topics}_a_{alpha}_b_{beta}_{batch}'
    
    if use_sparse_matrix is True:
        doc_token_topic_m = load_sparse_csr_matrix(path_doc_token_topic)
        doc_topic_counts_m = load_sparse_csr_matrix(path_doc_topic_counts)
        token_topic_counts_m = load_sparse_csr_matrix(path_token_topic_counts)
    
    else:
        doc_token_topic_m = load_dense_matrix(path_doc_token_topic)
        doc_topic_counts_m = load_dense_matrix(path_doc_topic_counts)
        token_topic_counts_m = load_dense_matrix(path_token_topic_counts)
        

    #varrendo todos os elementos nonzero da matriz doc_token
    for i, j in zip(*np.nonzero(doc_token_topic_m)):

        #o index do tópico é o tópico - 1
        sampling_topic_index = doc_token_topic_m[ i , j ] - 1
        doc_topic_counts_m[ i , sampling_topic_index ] -= 1
        token_topic_counts_m[ j , sampling_topic_index ] -= 1

        #CDT é o vetor contendo o counts de tópico para o doc "i"
        if use_sparse_matrix is True:
            doc_topic_counts_array = np.ravel(doc_topic_counts_m[ i , : ])
        else:
            doc_topic_counts_array = doc_topic_counts_m[ i , : ]
        
        cDT = doc_topic_counts_array + alpha
        denominator_pDT = doc_topic_counts_array.sum() + (n_topics * alpha)
        #print(cDT.shape, denominator_pDT)
        pDT = cDT / denominator_pDT

        #CTT é o vetor contendo o counts de tópico para o token "j"
        if use_sparse_matrix is True:
            token_topic_counts_array = np.ravel(token_topic_counts_m[ j , : ])
            token_topic_counts_sum= np.ravel(token_topic_counts_m.sum(axis=0))
        else:
            token_topic_counts_array = token_topic_counts_m[ j , : ]
            token_topic_counts_sum = token_topic_counts_m.sum(axis=0)
        
        cTT = token_topic_counts_array + beta
        denominator_pTT = token_topic_counts_sum + (n_tokens * beta)
        #print(cTT.shape, denominator_pTT.shape)
        pTT = cTT / denominator_pTT
        
        #determinando p_t
        p_t = pDT * pTT

        #normalizando a distribuição
        p_t /= p_t.sum()
        #obtendo o novo topic index a ser atribuido
        new_t_index = np.random.multinomial(1, p_t).argmax()
        
        #soma-se 1 pois o o topic_index começa em 0 e o primeiro tópico é 1
        doc_token_topic_m[ i , j ] = new_t_index + 1

        #atualizando a contagem do novo tópico para o doc "i", token "j" 
        doc_topic_counts_m[ i , new_t_index ] += 1
        token_topic_counts_m[ j , new_t_index ] += 1
        #print(f'doc_topic_counts_m[ {i} , : ]', doc_topic_counts_m[ i , : ])
        #print(f'token_topic_counts_m[ {j} , : ]', token_topic_counts_m[ j , : ])
        #time.sleep(1)

    #salvando a matrix doc_token_topic
    if use_sparse_matrix is True:
        save_sparse_csr_matrix(path_doc_token_topic, doc_token_topic_m)
        save_sparse_csr_matrix(path_doc_topic_counts, doc_topic_counts_m)
        save_sparse_csr_matrix(path_token_topic_counts, token_topic_counts_m)
    
    else:
        save_dense_matrix(path_doc_token_topic, doc_token_topic_m)
        save_dense_matrix(path_doc_topic_counts, doc_topic_counts_m)
        save_dense_matrix(path_token_topic_counts, token_topic_counts_m)

    del doc_token_topic_m
    del doc_topic_counts_m
    del token_topic_counts_m