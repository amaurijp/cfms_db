#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
import time
import os
import h5py
import pandas as pd
import numpy as np
import regex as re
import random

from ML import stats
from ML import use_machine

from FUNCTIONS import get_filenames_from_folder
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import get_search_terms_from_input
from FUNCTIONS import get_vectors_from_input
from FUNCTIONS import save_dic_to_json
from FUNCTIONS import error_print_abort_class
from FUNCTIONS import create_article_file_list_with_filename

from functions_VECTORS import cosine_sim

from functions_TOKENS import get_tokens_from_sent



class search_engine(object):
    
    def __init__(self, diretorio = None, index_list = None):
        
        print('\n( Class: search_engine )')

        self.diretorio = diretorio

        #carregando a lista de arquivos processados
        self.filtered_articles = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')
        
        #obtendo uma lista de documentos selecionadas por index
        self.index_list_to_scan = index_list
        
        self.class_name = 'search_engine'
        self.abort_class = False

        #procurando as definições dos modelos de filtro
        if not os.path.exists(self.diretorio + '/Settings/models_setup.json'):
            print('\nERRO!')
            print('Não foi encontrado o dicionário com as definições dos modelos')
            print('Definir em um dicionário "json" quais modelos devem ser usados para a rotina de pesquisa.')
            return

        else: 
            self.models_setup = load_dic_from_json(self.diretorio + '/Settings/models_setup.json')
            print('Importando os models sets em: ', self.diretorio + '/Settings/models_setup.json')
            time.sleep(1)


    
    def set_search_conditions(self, 
                              SE_inputs = {},
                              export_random_sent = False,
                              revise_search_conditions = 'y'):
        
        print('\n( Function: set_search_conditions )')

        self.scan_sent_by_sent = SE_inputs['scan_sent_by_sent']
        self.output_DF_name = SE_inputs['filename']
        self.filter_model = SE_inputs['search_inputs']['filter_section']
        self.lower_sentence_in_semantic_search = SE_inputs['search_inputs']['lower_sent_to_semantic_search']
        self.cos_thr = SE_inputs['search_inputs']['cos_thr']
        self.export_random_sent = export_random_sent

        #separando os termos a serem buscados
        self.search_terms_dic = get_search_terms_from_input(SE_inputs['search_inputs'], diretorio = self.diretorio)
        
        #montando os vetores a serem buscados
        self.search_vectors_dic = get_vectors_from_input(SE_inputs['search_inputs'], lsa_dim = self.models_setup["classifier"]['lsa_topic_dim'], lda_dim = self.models_setup["classifier"]['lda_topic_dim'])

        print('\nSearch settings:')
        
        print('-----------------------------------------------------------------')
        print('(literal) Prim. terms: ', self.search_terms_dic['literal']['primary'])
        print()        
        print('(literal) Sec. terms: ', self.search_terms_dic['literal']['secondary'])
        print()
        print('(literal) Sec. terms operations: ', self.search_terms_dic['literal']['operations'])

        print('-----------------------------------------------------------------')
        print('(semantic) Prim. terms: ', self.search_terms_dic['semantic']['primary'])
        print()
        print('(semantic) Sec. terms: ', self.search_terms_dic['semantic']['secondary'])
        print()
        print('(semantic) Sec. terms operations: ', self.search_terms_dic['semantic']['operations'])
        print()
        print('(semantic) Sec. nGrams type: ', self.search_terms_dic['semantic']['nGrams'])

        print('-----------------------------------------------------------------')
        if self.search_vectors_dic['topic']["lda_sents"] is not None:
            print('(topic) LDA sent: topics ',  self.search_vectors_dic['topic']["lda_sents"].nonzero()[0] + 1 ,
                  '; values ' , self.search_vectors_dic['topic']["lda_sents"] [ self.search_vectors_dic['topic']["lda_sents"].nonzero()[0] ], 
                  ' ; dim: ' , self.search_vectors_dic['topic']["lda_sents"].shape)
            print()
        if self.search_vectors_dic['topic']["lda_articles"] is not None:    
            print('(topic) LDA article: topics ', self.search_vectors_dic['topic']["lda_articles"].nonzero()[0]  + 1, 
                  '; values ' , self.search_vectors_dic['topic']["lda_articles"] [ self.search_vectors_dic['topic']["lda_articles"].nonzero()[0] ], 
                  ' ; dim: ' , self.search_vectors_dic['topic']["lda_articles"].shape)
            print()
        if self.search_vectors_dic['topic']["lsa_sents"] is not None:
            print('(topic) LSA sent: topics ', self.search_vectors_dic['topic']["lsa_sents"].nonzero()[0]  + 1, '; values ' , 
                  self.search_vectors_dic['topic']["lsa_sents"] [ self.search_vectors_dic['topic']["lsa_sents"].nonzero()[0] ], 
                  ' ; dim: ' , self.search_vectors_dic['topic']["lsa_sents"].shape)
            print()
        if self.search_vectors_dic['topic']["lsa_articles"] is not None:
            print('(topic) LSA article: topics ', self.search_vectors_dic['topic']["lsa_articles"].nonzero()[0]  + 1, '; values ' , 
                  self.search_vectors_dic['topic']["lsa_articles"] [ self.search_vectors_dic['topic']["lsa_articles"].nonzero()[0] ], 
                  ' ; dim: ' , self.search_vectors_dic['topic']["lsa_articles"].shape)

        print('-----------------------------------------------------------------')
        print('(regex) Regex entry: ', self.search_terms_dic['regex']['regex_entry'])
        print()
        print('(regex) Regex patterns: ', self.search_terms_dic['regex']['regex_pattern'])

        
        self.bool_search_settings = {}
        self.bool_search_settings['literal'] = False
        self.bool_search_settings['semantic'] = False
        self.bool_search_settings['topic'] = False
        self.bool_search_settings['regex'] = False
        
        #revisando as condições de busca

        if revise_search_conditions[0] in ('y', 's'):
            while True:
                input_entry = input('\nFazer a procura com esses termos? (s/n)\n')
                
                if input_entry.lower() in ('s', 'sim', 'y', 'yes'):
                    break
                elif input_entry.lower() in ('n', 'não', 'nao', 'no'):                
                    self.abort_class = True
                    return
                else:
                    print('Erro! Digite "s" ou "n".')                


        #estabelecendo as condições dos termos de pesquisa para search modes "literal" e "semantic"
        if len(self.search_terms_dic['literal']['primary']) != 0:
            self.bool_search_settings['literal'] = True
        if len(self.search_terms_dic['semantic']['primary']) != 0:
            self.bool_search_settings['semantic'] = True
        if self.search_vectors_dic['topic']['any'] is True:
            self.bool_search_settings['topic'] = True            
        if len(self.search_terms_dic['regex']['regex_pattern']) != 0:
            self.bool_search_settings['regex'] = True
        
        print('Search boolean settings:')
        print('literal - ', self.bool_search_settings['literal'], 
              ' ; semantic - ', self.bool_search_settings['semantic'], 
              ' ; topic - ', self.bool_search_settings['topic'], 
              ' ; regex - ', self.bool_search_settings['regex'], 
              ' ; filter - ', self.filter_model
              )

        time.sleep(3)

        if self.bool_search_settings['topic'] is True:
            
            #carregando as matrizes DOC_TOPIC
            self.topic_matrices = {}

            #lsa_sents
            if self.search_vectors_dic["topic"]["lsa_sents"] is not None:
                try:
                    self.lsa_sents_topic_h5 = h5py.File(self.diretorio + f'/Outputs/models/{self.models_setup["classifier"]["lsa_sents"]}.h5', 'r')
                    self.topic_matrices['lsa_sents'] = self.lsa_sents_topic_h5['data']
                except FileNotFoundError:
                    print('ERRO!')
                    print('O modelo "lsa_sents" foi requisitado mais não foi encontrado em ~/Outputs/models.')
                    print('Checar o arquivo "models_setup.json" em ~/Settings.')

            #lsa_articles
            if self.search_vectors_dic["topic"]["lsa_articles"] is not None:
                try:
                    self.lsa_articles_topic_h5 = h5py.File(self.diretorio + f'/Outputs/models/{self.models_setup["classifier"]["lsa_articles"]}.h5', 'r')
                    self.topic_matrices['lsa_articles'] = self.lsa_articles_topic_h5['data']
        
                except FileNotFoundError:
                    print('ERRO!')
                    print('O modelo "lsa_articles" foi requisitado mais não foi encontrado em ~/Outputs/models.')
                    print('Checar o arquivo "models_setup.json" em ~/Settings.')

            #lda_sents
            if self.search_vectors_dic["topic"]["lda_sents"] is not None:
                try:
                    self.lda_sents_topic_h5 = h5py.File(self.diretorio + f'/Outputs/models/{self.models_setup["classifier"]["lda_sents"]}.h5', 'r')
                    self.topic_matrices['lda_sents'] = self.lda_sents_topic_h5['data']

                except FileNotFoundError:
                    print('ERRO!')
                    print('O modelo "lda_sents" foi requisitado mais não foi encontrado em ~/Outputs/models.')
                    print('Checar o arquivo "models_setup.json" em ~/Settings.')

            #lda_articles
            if self.search_vectors_dic["topic"]["lda_articles"] is not None and os.path.exists(self.diretorio + f'/Outputs/models/{self.models_setup["classifier"]["lda_articles"]}.h5'):
                try:
                    self.lda_articles_topic_h5 = h5py.File(self.diretorio + f'/Outputs/models/{self.models_setup["classifier"]["lda_articles"]}.h5', 'r')
                    self.topic_matrices['lda_articles'] = self.lda_articles_topic_h5['data']

                except FileNotFoundError:
                    print('ERRO!')
                    print('O modelo "lda_articles" foi requisitado mais não foi encontrado em ~/Outputs/models.')
                    print('Checar o arquivo "models_setup.json" em ~/Settings.')



        #definindo os filtros
        self.nn_filter = False
        self.ml_filter = False
        if self.filter_model in ('introduction', 'methodology', 'results'):
            
            #carregando o WV
            wv_type = self.models_setup['section_filters']['wv_model']
            wv_matrix_name = self.models_setup['section_filters']['wv_matrix_name']
            
            try:
                wv_matrix = pd.read_csv(self.diretorio + f'/Outputs/wv/{wv_matrix_name}.csv', index_col = 0)
                print('Importando a wv_matrix em: ', self.diretorio + f'/Outputs/wv/{wv_matrix_name}.csv')
                time.sleep(1)

            except FileNotFoundError:
                print(f'Erro para a entrada wv_name: {wv_matrix_name}')
                print('Entradas disponíveis na pasta ~/Outputs/models/')
                filenames = get_filenames_from_folder(self.diretorio + '/Outputs/wv', file_type = 'csv', print_msg = False)
                for filename in [filename for filename in filenames if filename[ : len('w2vec') ] == 'w2vec' ]:
                    print(filename)
                print('> Abortando a classe: use_machine')
                self.abort_class = True
                return

            #carregando os tokens do IDF
            IDF_matrix = pd.read_csv(self.diretorio + f'/Outputs/tfidf/idf.csv', index_col = 0)
            tokens_list = IDF_matrix.index.values

            #setando a máquina
            #qual seção será filtrada
            self.section_to_find = self.filter_model.lower()

            #setando os modelos de filtros para cada seção
            self.intro_filter = use_machine(model_name = self.models_setup['section_filters']['introduction'], diretorio = self.diretorio)
            if self.intro_filter.model_found is True:
                self.intro_filter.set_machine_parameters_to_use(tokens_list = tokens_list, wv_type = wv_type, wv_matrix_name = wv_matrix_name, wv_matrix = wv_matrix, doc_topic_matrix = self.lsa_sents_topic_matrix)

            self.methods_filter = use_machine(model_name = self.models_setup['section_filters']['methodology'], diretorio = self.diretorio)
            if self.methods_filter.model_found is True:
                self.methods_filter.set_machine_parameters_to_use(tokens_list = tokens_list, wv_type = wv_type, wv_matrix_name = wv_matrix_name, wv_matrix = wv_matrix, doc_topic_matrix = self.lsa_sents_topic_matrix)

            self.results_filter = use_machine(model_name = self.models_setup['section_filters']['results'], diretorio = self.diretorio)
            if self.results_filter.model_found is True:
                self.results_filter.set_machine_parameters_to_use(tokens_list = tokens_list, wv_type = wv_type, wv_matrix_name = wv_matrix_name, wv_matrix = wv_matrix, doc_topic_matrix = self.lsa_sents_topic_matrix)
                
            #caso o filtro seja None
            if True not in [ self.intro_filter.model_found, self.methods_filter.model_found, self.results_filter.model_found ]:
                self.nn_filter = False
                print('ATENÇÃO! Erro no carregamento do filtro de seção!')
                print('Checar os nomes no arquivo em: ', self.diretorio + '/Settings/models_setup.json')
                time.sleep(5)
            else:
                self.nn_filter = True

        #abrindo os DFs
        self.open_DFs()



    def search_with_combined_models(self):
        
        print('\n( Function: search_with_combined_models )')
        
        if self.abort_class is True:
            error_print_abort_class(self.class_name)
            return
        
        #checando a incompatibilidade do modo scan_sent_by_sent (os filtros de seção só funcionam com o scan_sent_by_sent == True)
        if (self.nn_filter is True or self.ml_filter is True) and self.scan_sent_by_sent is False:
            print('ERRO! Os filtros de seção só funcionam com o "scan_sent_by_sent" em "True", pois a varredura é feita sentença por sentença.')
            return
        
        #definindo a lista de documentos a ser escaneada
        if self.index_list_to_scan is not None:
            filenames_to_scan = self.index_list_to_scan
        else:
            filenames_to_scan = self.filtered_articles[ self.lastfile_index : ]
        
        #varrendo os documentos .csv com as sentenças
        for filename in filenames_to_scan:
            
            #check caso tenho sido encontrado algo no artigo
            found_in_article = False
            
            #checar se o artigo já não teve os termos extraidos
            if filename not in self.extracted_article_list:

                print('------------------------------------------------------')                
                print('Looking in ', filename, '...')
                self.filename = filename
                
                #zerando os contadores
                self.set_initial_counters()
                match_any_sent = False
                
                #abrindo o csv com as sentenças do artigo
                self.sentDF = pd.read_csv(self.diretorio + '/Outputs/sents_filtered/' + f'{self.filename}.csv', index_col = 0)

                #vai procurar no texto inteiro (sentenças concatenadas)
                if self.scan_sent_by_sent is False:
                    
                    #juntando todas as sentenças em um só texto
                    self.sentDF_copy = pd.DataFrame(columns=['first_sent_index', 'article Number', 'Sentence', 'Section'])
                    
                    first_sent_index = self.sentDF.index[0]
                    article_number = self.sentDF.loc[first_sent_index, 'article Number']
                    article_section = self.sentDF.loc[first_sent_index, 'Section']

                    concat_sents = ''
                    for sent in self.sentDF['Sentence'].values:
                        concat_sents += sent + ' '
                    
                    #tirando o último espaço
                    concat_sents = concat_sents[:-1]
                    
                    self.sentDF_copy.loc[0] = (first_sent_index, article_number, concat_sents, article_section)
                    self.sentDF_copy.set_index('first_sent_index', inplace = True)
                    self.sentDF_copy.index.name = None

                #varrer sentença por sentença
                elif self.scan_sent_by_sent is True:
                    self.sentDF_copy = self.sentDF
                
                #numero da linha da entrada (as linhas separam o número de vezes que os termos foram encontrados no artigo)
                line = 0
                #numero da coluna da entrada (as colunas separam os termos que foram procurados)
                column = 1 + len(self.ID_columns)
                
                #varrendo as sentenças
                for index in self.sentDF_copy.index:
                    
                    #carregando a sentença
                    sent = self.sentDF_copy.loc[index, 'Sentence']
                    
                    #encontrando o comprimento máxiom da sentença
                    if len(sent) > self.max_sent_len:
                        self.max_sent_len = len(sent)

                    #definindo o dic para o search check
                    check_search_bool = {}
            
                    #fazendo a procura por busca de termos literais
                    if self.bool_search_settings['literal'] is True:                        
                        check_search_bool['literal'] = self.search_with_terms(sent, terms_type = 'literal')

                    #fazendo a procura por busca de termos semânticos
                    if self.bool_search_settings['semantic'] is True:
                        #caso a sentença seja procurada em lower
                        sent_modified = sent
                        if self.lower_sentence_in_semantic_search is True:
                            sent_modified = sent.lower()
                        check_search_bool['semantic'] = self.search_with_terms(sent_modified, terms_type = 'semantic')

                    #fazendo a procura com de topic match
                    if self.bool_search_settings['topic'] is True:
                        check_search_bool['topic'] = self.search_with_topic_vectors(sent_index = index, 
                                                                                    article_index = self.filtered_articles.index(filename), 
                                                                                    vectors_dic = self.search_vectors_dic['topic'], 
                                                                                    matrices_dic = self.topic_matrices, 
                                                                                    cos_threshold = self.cos_thr)
                    
                    #fazendo a procura com o padrão regex
                    if self.bool_search_settings['regex'] is True:                    
                        check_search_bool['regex'] = self.find_regex_in_sent(sent)
                    
                    #print(check_search_bool.values())
                    #caso todos os critérios de procura usados sejam verdadeiros
                    if False not in check_search_bool.values():                        
                        
                        #checando se essa sentença faz parte de uma seção específica (com a rede neural treinada)
                        proba_result = 1
                        proba_threshold = 0.5
                        if self.nn_filter is True and self.ml_filter is False:
                            
                            section_proba_result = {}

                            #checando cada seção
                            if self.intro_filter.model_found is True:
                                section_proba_result['introduction'] = self.intro_filter.check_sent_in_section_for_NN(index, DF = self.sentDF_copy)
                            elif self.intro_filter.model_found is False:
                                section_proba_result['introduction'] = 0
                            if self.methods_filter.model_found is True:
                                section_proba_result['methodology'] = self.methods_filter.check_sent_in_section_for_NN(index, DF = self.sentDF_copy)
                            elif self.methods_filter.model_found is False:
                                section_proba_result['methodology'] = 0
                            if self.results_filter.model_found is True:
                                section_proba_result['results'] = self.results_filter.check_sent_in_section_for_NN(index, DF = self.sentDF_copy)
                            elif self.results_filter.model_found is False:
                                section_proba_result['results'] = 0

                            for section in ('introduction', 'methodology', 'results'):
                                
                                #caso seja a seção a ser encontrada e a probabilidade der positiva
                                if section_proba_result[section] >= 0.5 and section == self.section_to_find:
                                    pass
                                #caso não seja a seção a ser encontrada e a probabilidade der negativa
                                elif section_proba_result[section] < 0.5 and section != self.section_to_find:
                                    pass
                                #caso não seja a seção
                                else:
                                    proba_result = 0
                                    break

                        #o ML não foi otimizado
                        elif self.nn_filter is False and self.ml_filter is True:
                            proba_result = self.mc.check_sent_in_section_for_ML(index, DF = self.sentDF_copy)
                            #o threshold para o ML é maior devido à precisão mais baixa
                            proba_threshold = 0.7
                            
                        if proba_result >= proba_threshold:                                            
                            #print('Encontrado o termo primario: ', self.search_terms_dic['primary'][term_index])
                            #parte do texto extraida para cada match
                            self.indexed_results[self.results_counter] = ([line, column], index, sent)
                            line += 1
                            self.results_counter +=1                    
                            print('> TERMS FOUND')
                            print('> Extracted sentence: ', sent)
                            #print('Indexes collected: ', index)
                            #time.sleep(2)
                            match_any_sent = True
    
                            #contadores para exportar nas DFs de report
                            self.search_report_dic['search'][self.output_DF_name]['total_finds'] += 1
                            if found_in_article is False:
                                self.search_report_dic['search'][self.output_DF_name]['article_finds'] += 1
                                found_in_article = True

                #definindo o número de linhas e colunas que formarão o DATAFRAME
                if line > self.DF_line_number:
                    self.DF_line_number = line
                self.DF_column_number = 2 + len(self.ID_columns)
                
                if match_any_sent is True:
                    #colocando os resultados na DF
                    self.put_search_results_in_DF(self.indexed_results)
                    
                #salvando o search report
                self.search_report_dic['search'][self.output_DF_name]['last_article_processed'] = self.filename
                save_dic_to_json(self.diretorio + f'/Outputs/log/se_report.json', self.search_report_dic)

            else:        
                print(f'O documento {filename} já foi processado.')
                print('Passando para o próximo documento...')
                continue
            
        if len(filenames_to_scan) == 0 or filename == filenames_to_scan[-1]:

            #mudando o status no LOG report
            self.search_report_dic['search'][self.output_DF_name]['searching_status'] = 'finished'
            save_dic_to_json(self.diretorio + f'/Outputs/log/se_report.json', self.search_report_dic)
            
            #gerando o search report
            self.generate_search_report()

            #fechando o arquivo h5 e deletando a matriz
            try:
                self.lsa_sents_topic_h5.close()
            except AttributeError:
                pass
            try:
                self.lsa_articles_topic_h5.close()
            except AttributeError:
                pass
            try:
                self.lda_sents_topic_h5.close()
            except AttributeError:
                pass
            try:
                self.lda_articles_topic_h5.close()
            except AttributeError:
                pass

            del self.topic_matrices



    def set_initial_counters(self):

        self.indexed_results = {}
        #contador para o dicionário com as sentenças nas quais os termos foram encontrados
        self.results_counter = 1
        #contador para sentenças randômicas extraídas
        self.rand_sent_counter = 0
        #número de linhas da matrix (começa com zero)
        self.DF_line_number = 0
        #determinando o tamanho (length) da maior sentença do artigo (esse valor determinará o dtype da np.array abaixo)
        self.max_sent_len = 0



    def open_DFs(self):

        print('\n( Function: open_DFs )')
        
        #checando erros de instanciação/inputs
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return

        #checando o diretorio EXTRACTED
        if not os.path.exists(self.diretorio + '/Outputs/extracted'):
            os.makedirs(self.diretorio + '/Outputs/extracted')                             

        #checando os artigos nos quais foram extraidos as sentenças
        self.extracted_article_list = []
        #checar a existência de DF de fragmentos
        self.ID_columns = ['Filename', 'Counter']
        if os.path.exists(self.diretorio + f'/Outputs/extracted/{self.output_DF_name}.csv'):
            self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/extracted/{self.output_DF_name}.csv', index_col=[0,1])
            last_article_file_processed = self.output_DF.index.levels[0].values[-1]
            self.extracted_article_list = create_article_file_list_with_filename(last_article_file_processed)
            #print(self.extracted_article_list)
        else:
            #criando a coluna de indexação do DF de fragmentos
            self.columns = self.ID_columns + [ self.output_DF_name ] + [term + '_index' for term in [ self.output_DF_name ] ]
            self.output_DF = pd.DataFrame(columns=self.columns)
            self.output_DF.set_index(['Filename', 'Counter'], inplace=True)
                
        #checando os artigo nos quais foram extraidos as sentenças randômicos
        #lista para extrações randômicas
        self.rand_extracted_article_list = []
        if (self.export_random_sent is True):
            #checar a existência de DF de fragmentos randomicos extraídos
            if os.path.exists(self.diretorio + f'/Outputs/extracted/{self.output_DF_name}_random.csv'):
                self.rand_sent_DF = pd.read_csv(self.diretorio + f'/Outputs/extracted/{self.output_DF_name}_random.csv', index_col=[0,1])
                self.rand_extracted_article_list = self.rand_sent_DF.index.levels[0].values
            else:
                #criando a coluna de indexação do DF de fragmentos
                self.columns = self.ID_columns + ['rand_sent', 'rand_sent_index']
                self.rand_sent_DF = pd.DataFrame([], columns=self.columns)
                self.rand_sent_DF.set_index(['Filename', 'Counter'], inplace=True)
        
        #abrindo o search report (para parar e resumir o fit)
        if os.path.exists(self.diretorio + f'/Outputs/log/se_report.json'):
            #carregando o dicionário
            self.search_report_dic = load_dic_from_json(self.diretorio + f'/Outputs/log/se_report.json')
            try:
                #último filename processado
                file_name = self.search_report_dic['search'][self.output_DF_name]['last_article_processed']
                #index do ultimo filename processado
                self.lastfile_index = self.filtered_articles.index(file_name) + 1
                print('Last file searched: ', file_name)
            
            except KeyError:
                self.search_report_dic['search'][self.output_DF_name] = {}
                self.search_report_dic['search'][self.output_DF_name]['last_article_processed'] = None
                self.search_report_dic['search'][self.output_DF_name]['total_finds'] = 0
                self.search_report_dic['search'][self.output_DF_name]['article_finds'] = 0
                self.search_report_dic['search'][self.output_DF_name]['searching_status'] = 'ongoing'
                self.lastfile_index = 0

        else:
            self.search_report_dic = {}
            self.search_report_dic['search'] = {}
            self.search_report_dic['search'][self.output_DF_name] = {}
            self.search_report_dic['search'][self.output_DF_name]['last_article_processed'] = None
            self.search_report_dic['search'][self.output_DF_name]['total_finds'] = 0
            self.search_report_dic['search'][self.output_DF_name]['article_finds'] = 0
            self.search_report_dic['search'][self.output_DF_name]['searching_status'] = 'ongoing'
            self.lastfile_index = 0


    def search_with_terms(self, text, terms_type = 'literal'):
        
        #check caso os termos sejam encontrados na sentença
        found_terms = False        
            
        #procurando pelos termos primários no texto
        for i in range(len(self.search_terms_dic[terms_type]['primary'])):
            #testando a probabilidade de apariçaõ do termo
            #print('Term Primary: ', self.search_terms_dic[terms_type]['primary'][i])
            #print('Prob: ', self.term_prob_list_dic['primary'][i])
            
            term_to_find = self.search_terms_dic[terms_type]['primary'][i]
            #term_to_find = re.sub(r'\(', '\(', self.search_terms_dic[terms_type]['primary'][i])
            
            if terms_type == 'semantic':
                
                #procurando qual o tipo de nGram (0, 1, 2 ou 'x')
                if self.search_terms_dic['semantic']['nGrams'][0] in ('0', '2', 'x'):
                    #procurando o nGram na sentença
                    if re.search( term_to_find , text):
                        #print('Encontrado o termo primario: ', self.search_terms_dic[terms_type]['primary'][i])
                        found_terms = True
                        break
                
                #se o lower_sentence tiver desabilitado, pode se procurar pradrões regex associados aos nGrams
                #(isso significa que a lista .csv com nGrams foi modificada)
                elif self.lower_sentence_in_semantic_search is False:
                    #procurando o nGram na sentença
                    if re.search( term_to_find , text):
                        #print('Encontrado o termo primario: ', self.search_terms_dic[terms_type]['primary'][i])
                        found_terms = True
                        break
                
                else:
                    #procurando o termo entre os tokens
                    if term_to_find in get_tokens_from_sent(text):
                        #print('Encontrado o termo primario: ', self.search_terms_dic[terms_type]['primary'][i])
                        found_terms = True
                        break
                    
            elif terms_type == 'literal':
            
                #procurando o termo na sentença com regex
                if re.search( term_to_find , text ):
                    #print('Encontrado o termo primario: ', self.search_terms_dic[terms_type]['primary'][i])
                    found_terms = True
                    break

        #procurando os termos secundários                                     
        if (found_terms is True):
            #dicionário para coletar as operações + e - feitas com os termos secundários
            check_sec_term_operation = {}                               
            for j in range(len(self.search_terms_dic[terms_type]['secondary'])):
                #assumindo primariamente que nenhum termo foi encontrado para esse conjunto de termos secundários [j]
                check_sec_term_operation[j] = '-'
                #procurando cada termo secundário dentro do grupo de secundários
                for sec_term_index in range(len(self.search_terms_dic[terms_type]['secondary'][j])):
                    #testando a probabilidade de apariçaõ do termo
                    #print('Term Secondary: ', self.search_terms_dic[terms_type]['secondary'][j][sec_term_index])
                    #print('Prob: ', self.term_prob_list_dic['secondary'][j][sec_term_index])
                    
                    self.search_terms_dic[terms_type]['secondary'][j][sec_term_index]
                    term_to_find = self.search_terms_dic[terms_type]['secondary'][j][sec_term_index]

                    #term_to_find = re.sub(r'\(', '\(' , self.search_terms_dic[terms_type]['secondary'][j][sec_term_index])
                    
                    if terms_type == 'semantic':
                        
                        #procurando qual o tipo de nGram (0, 1, 2 ou x)
                        if self.search_terms_dic['semantic']['nGrams'][j + 1] in ('0', '2', 'x'):
                            #procurando o nGram na sentença
                            if re.search( term_to_find , text):
                                #print('Encontrado o termo secundário: ', self.search_terms_dic[terms_type]['secondary'][j][sec_term_index])
                                check_sec_term_operation[j] = '+'
                                break
                                
                        #se o lower_sentence tiver desabilitado, pode se procurar pradrões regex associados aos nGrams
                        #(isso significa que a lista .csv com nGrams foi modificada)
                        elif self.lower_sentence_in_semantic_search is False:
                            #procurando o nGram na sentença
                            if re.search( term_to_find , text):
                                #print('Encontrado o termo primario: ', self.search_terms_dic[terms_type]['primary'][i])
                                check_sec_term_operation[j] = '+'
                                break

                        else:
                            #procurando o termo entre os tokens
                            if term_to_find in get_tokens_from_sent(text):
                                #print('Encontrado o termo secundário: ', self.search_terms_dic[terms_type]['secondary'][j][sec_term_index])
                                check_sec_term_operation[j] = '+'
                                break
                        
                    elif terms_type == 'literal':
                        
                        if re.search( term_to_find , text ):
                            #print('Encontrado o termo secundário: ', self.search_terms_dic[terms_type]['secondary'][j][sec_term_index])
                            check_sec_term_operation[j] = '+'
                            break
            
            #caso as operações de termos secundários sejam as mesmas inseridas                        
            if list(check_sec_term_operation.values()) == self.search_terms_dic[terms_type]['operations']:
                pass
            else:
                found_terms = False
                                
        #caso todos os termos tenham sido encontrados
        if found_terms is True:
            #print('- match: search_with_terms')
            return True
        else:
            return False



    def search_with_topic_vectors(self, sent_index: int = None, article_index: int = None, vectors_dic: dict = None, matrices_dic: dict = None, cos_threshold: float = None):

        bool_results = {}

        if vectors_dic['lsa_sents'] is not None:
            bool_results['lsa_sents'] = True if cosine_sim(vectors_dic['lsa_sents'], matrices_dic['lsa_sents'][sent_index]) > cos_threshold else False
        
        if vectors_dic['lsa_articles'] is not None:
            bool_results['lsa_articles'] = True if cosine_sim(vectors_dic['lsa_articles'], matrices_dic['lsa_articles'][article_index]) > cos_threshold else False
        
        if vectors_dic['lda_sents'] is not None:
            bool_results['lda_sents'] = True if cosine_sim(vectors_dic['lda_sents'], matrices_dic['lda_sents'][sent_index]) > cos_threshold else False
        
        if vectors_dic['lda_articles'] is not None:
            bool_results['lda_articles'] = True if cosine_sim(vectors_dic['lda_articles'], matrices_dic['lda_articles'][article_index]) > cos_threshold else False

        #case algum vector tenha dado match no cosseno
        if True in bool_results.values():
            return True
        else:
            return False



    def find_regex_in_sent(self, text):
        
        #se foi encontrado algum regex
        if re.search( self.search_terms_dic['regex']['regex_pattern'] , text ):
            #print('- match: regex')
            return True
        else:
            return False



    def generate_search_report(self):
            
        #abrindo o SE report            
        if os.path.exists(self.diretorio + '/Settings/SE_inputs.csv'):
            search_report_DF = pd.read_csv(self.diretorio + '/Settings/SE_inputs.csv', index_col = 0)

            search_report_DF.loc[self.output_DF_name , 'total_finds' ] = self.search_report_dic['search'][self.output_DF_name]['total_finds']
            search_report_DF.loc[self.output_DF_name , 'article_finds' ] = self.search_report_dic['search'][self.output_DF_name]['article_finds']
            search_report_DF.loc[self.output_DF_name , 'search_status' ] = 'finished'

            search_report_DF.sort_index(inplace=True)
            search_report_DF.to_csv(self.diretorio + '/Settings/SE_inputs.csv')
            print('Salvando o SE report em ~/Settings/SE_inputs.csv')


    
    def put_search_results_in_DF(self, indexed_results):

        #adicionando as entradas na numpy array
        array = np.zeros([self.DF_line_number, self.DF_column_number], dtype=np.dtype(f'U{self.max_sent_len}'))
        for results_found in sorted(indexed_results.keys()):
            pos_Y = indexed_results[results_found][0][0]
            pos_X = indexed_results[results_found][0][1]
            #inserindo a sentença
            #inserindo o index da sentença
            array[pos_Y, pos_X] = indexed_results[results_found][1]
            array[pos_Y, pos_X - 1] = indexed_results[results_found][2]
            #print(indexed_results[results_found][2])
            
        #adicionando o titulo e o valor da entrada para o resultado da pesquisa
        for line in range(array.shape[0]):
            array[line, 0] = self.filename
            array[line, 1] = line

        #transformando a array em dataframe
        columns_terms = [ [term, term  + '_index'] for term in [ self.output_DF_name ] ]
        self.columns = self.ID_columns + [item for sublist in columns_terms for item in sublist] #flatten na lista com sublistas
        DF_to_concat = pd.DataFrame(array, columns=self.columns)
        DF_to_concat.set_index(['Filename', 'Counter'], inplace=True)
        #concatenando a DF
        self.output_DF = pd.concat([self.output_DF, DF_to_concat])
        #salvando a DataFram em CSV
        self.output_DF.to_csv(self.diretorio + f'/Outputs/extracted/{self.output_DF_name}.csv')
        #exportando sentenças randômicas
        if (self.export_random_sent is True):
            self.rand_sent_DF.to_csv(self.diretorio + f'/Outputs/extracted/{self.output_DF_name}_random.csv')
