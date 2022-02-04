#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
class search_engine(object):
    
    def __init__(self, diretorio = None, index_list = None):
        
        print('\n=============================\nSEARCH ENGINE...\n=============================\n')
        print('( Class: search_engine )')
        
        from FUNCTIONS import get_filenames_from_folder
        self.diretorio = diretorio

        #carregando a lista de arquivos processados
        self.extracted_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', file_type = 'csv')
        
        #obtendo uma lista de documentos selecionadas por index
        self.index_list_to_scan = index_list
        
        self.class_name = 'search_engine'


    
    def set_search_conditions(self, 
                              search_input_dic = {},
                              search_mode = 'search', 
                              output_DF_name = 'temperature',
                              export_random_sent = False,
                              filter_model = 'cnn',
                              topic_match_mode = 'largest',
                              nGrams_to_get_semantic = [0],
                              min_ngram_semantic_appearence = [5],
                              lower_sentence_in_semantic_search = False,
                              min_topic_corr_threshold = 0.15,
                              replace_Ngrams = False):
        
        print('( Function: set_search_conditions )')
                
        #testando os erros de inputs
        import time
        from ML import stats
        from ML import use_machine
        from FUNCTIONS import process_search_input_file
        from FUNCTIONS import error_incompatible_strings_input

        self.abort_class = error_incompatible_strings_input('search_mode', search_mode, ('search_with_topic_match', 'search_with_combined_models'), class_name = self.class_name)

        self.search_mode = search_mode.lower()
        self.output_DF_name = output_DF_name
        self.export_random_sent = export_random_sent
        self.nGrams_to_get_semantic = nGrams_to_get_semantic
        self.min_ngram_semantic_appearence = min_ngram_semantic_appearence
        self.lower_sentence_in_semantic_search = lower_sentence_in_semantic_search
        self.topic_match_mode = topic_match_mode
        self.replace_Ngrams = replace_Ngrams
        self.min_topic_corr_threshold = min_topic_corr_threshold

        #separando os termos a partir do arquivo TXT
        self.term_list_dic = process_search_input_file(search_input_dic, ngrams = self.nGrams_to_get_semantic, min_ngram_appearence = self.min_ngram_semantic_appearence, diretorio = self.diretorio)

        print('\nSearch settings:')
        
        print('-----------------------------------------------------------------')
        print('(literal) Prim. terms: ', self.term_list_dic['literal']['primary'])
        print()        
        print('(literal) Sec. terms: ', self.term_list_dic['literal']['secondary'])
        print()
        print('(literal) Sec. terms operations: ', self.term_list_dic['literal']['operations'])

        print('-----------------------------------------------------------------')
        print('(semantic) Prim. terms: ', self.term_list_dic['semantic']['primary'])
        print()
        print('(semantic) Sec. terms: ', self.term_list_dic['semantic']['secondary'])
        print()        
        print('(semantic) Sec. terms operations: ', self.term_list_dic['semantic']['operations'])
        
        print('-----------------------------------------------------------------')
        print('(topic) Prim. terms: ', self.term_list_dic['topic']['primary'])
        print()        
        print('(topic) Sec. terms: ', self.term_list_dic['topic']['secondary'])
        print()
        print('(topic) Sec. terms operations: ', self.term_list_dic['topic']['operations'])        
        
        print('-----------------------------------------------------------------')
        print('(regex) Regex entry: ', self.term_list_dic['regex']['regex_entry'])
        print()
        print('(regex) Regex patterns: ', self.term_list_dic['regex']['regex_pattern'])
        
        print('-----------------------------------------------------------------')
        print('(bayesian) Bayesian prob func para: ', self.term_list_dic['bayesian'])
        print()
        
        while True:
            input_entry = input('\nFazer a procura com esses termos? (s/n)\n')
            
            if input_entry.lower() in ('s', 'sim', 'y', 'yes'):
                self.bool_search_settings = {}
                self.bool_search_settings['literal'] = False
                self.bool_search_settings['semantic'] = False
                self.bool_search_settings['topic'] = False
                self.bool_search_settings['regex'] = False
                self.bool_search_settings['bayesian'] = False
                break
            elif input_entry.lower() in ('n', 'não', 'nao', 'no'):                
                self.abort_class = True
                return
            else:
                print('Erro! Digite "s" ou "n".')                


        #estabelecendo as condições dos termos de pesquisa para search modes "literal" e "semantic"
        if len(self.term_list_dic['semantic']['primary']) != 0:
            self.bool_search_settings['semantic'] = True
        if len(self.term_list_dic['literal']['primary']) != 0:
            self.bool_search_settings['literal'] = True
        
        #estabelecendo as condições dos termos de pesquisa para os search mode topic, regex e bayesian
        if len(self.term_list_dic['topic']['primary']) != 0:
            self.bool_search_settings['topic'] = True
            #fazendo as operações dos vetores tópico
            if self.search_mode == 'search_with_combined_models':
                print('ATENÇAO: o match de tópico não é usado no modo: "search_with_combined_models"')
            else:
                self.combine_topic_vectors()
        if len(self.term_list_dic['regex']['regex_pattern']) != 0:
            self.bool_search_settings['regex'] = True
        if len(self.term_list_dic['bayesian']) != 0:
            self.bool_search_settings['bayesian'] = True
            #instanciando a classe stats
            self.stats = stats(filter_stopwords = True, diretorio=self.diretorio)
            self.stats.set_bayes_classifier(prob_func_filename = self.term_list_dic['bayesian'])                
        
        print('Search boolean settings:')
        print('literal - ', self.bool_search_settings['literal'], 
              ' ; semantic - ', self.bool_search_settings['semantic'], 
              ' ; topic - ', self.bool_search_settings['topic'], 
              ' ; regex - ', self.bool_search_settings['regex'], 
              ' ; bayesian - ', self.bool_search_settings['bayesian'],
              ' ; filter - ', filter_model
              )
        time.sleep(3)


        #definindo o filtro
        try:
            if filter_model.lower() == 'cnn':
                self.nn_model = True
                self.ml_model = False
                self.wv_model = 'w2vec'
                self.wv_matrix_name = 'w2vec_cbow_SW_False_WS_3_rNgram_False'
    
                #setando a máquina
                model_name = 'sections_methodolody_conv1d_wv_SW_True_ch_4_rNgram_False'
                self.mc = use_machine(model_name = model_name, diretorio = self.diretorio)
                self.mc.set_machine_parameters_to_use(wv_model = self.wv_model, wv_matrix_name = self.wv_matrix_name)
            
            elif filter_model.lower() == 'logreg':
                self.nn_model = False
                self.ml_model = True
            
                #setando a máquina
                model_name = 'sections_methodolody_logreg_tv_SW_True_ch_1_rNgram_False'
                self.mc = use_machine(model_name = model_name, diretorio = self.diretorio)
                
        #caso o filtro seja None
        except AttributeError:
            self.nn_model = False
            self.ml_model = False
            pass
        
        time.sleep(5)
            
        #abrindo os DFs
        self.open_DFs()



    def search_with_combined_models(self):
        
        print('( Function: search_with_combined_models )')
        
        import h5py
        import regex as re
        import pandas as pd
        import time
        from FUNCTIONS import save_dic_to_json
        from FUNCTIONS import error_print_abort_class
        
        if self.abort_class is True:
            error_print_abort_class(self.class_name)
            return
        
        #definindo a lista de documentos a ser escaneada
        if self.index_list_to_scan is not None:
            filenames_to_scan = self.index_list_to_scan
        else:
            filenames_to_scan = self.extracted_DB_documents[ self.lastfile_index : ]
        
        #varrendo os documentos .csv com as sentenças
        for filename in filenames_to_scan:
            
            #check caso tenho sido encontrado algo no PDF
            found_in_pdf = False
            
            #checar se o PDF já não teve os termos extraidos
            if filename not in self.extracted_PDF_list:

                print('------------------------------------------------------')                
                print('Looking in ', filename, '...')
                self.filename = filename
                
                #zerando os contadores
                self.set_initial_counters()
                match_any_sent = False
                
                #carregando a matriz DOC_TOPIC
                h5 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'r')
                DOC_TOPIC_matrix = h5['data']
                
                #abrindo o csv com as sentenças do artigo
                self.sentDF = pd.read_csv(self.diretorio + '/Outputs/Sentences_filtered/' + f'{self.filename}.csv', index_col = 0)                    
                
                #numero da linha da entrada (as linhas separam o número de vezes que os termos foram encontrados no artigo)
                line = 0
                #numero da coluna da entrada (as colunas separam os termos que foram procurados)
                column = 1 + len(self.ID_columns)
                
                #varrendo as sentenças
                for index in self.sentDF.index:
                    
                    #print('sent index: ', index)
                    
                    #carregando a sentença
                    sent = self.sentDF.loc[index, 'Sentence']
                    
                    #encontrando o comprimento máxiom da sentença
                    if len(sent) > self.max_sent_len:
                        self.max_sent_len = len(sent)
                    
                    #pulando sentenças que possuem intervalos numéricos clusterizados (ex: 123&125&12321)
                    #esses padrões dão erro no regex
                    if re.search(r'–?[0-9]+(\.[0-9]+)?+\&–?[0-9]+(\.[0-9]+)?+\&–?[0-9]+(\.[0-9]+)?+', sent):
                        #print('Pulando a sentença: \n', sent_modified)
                        #time.sleep(1)
                        continue

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
                        check_search_bool['semantic'] = self.search_with_terms(sent_modified, terms_type = 'semantic', use_term_prob_filter = False)

                    #fazendo a procura com de topic match
                    #if self.bool_search_settings['topic'] is True:                    
                        #check_search_bool['topic'] = self.search_sent_with_topic_vectors(index, DOC_TOPIC_matrix, corr_threshold = self.min_topic_corr_threshold)
                    
                    #fazendo a procura com o classificador bayesiano
                    if self.bool_search_settings['bayesian'] is True:
                        check_search_bool['bayesian'] = self.search_sent_with_bayes_class(sent.lower()) 

                    #fazendo a procura com o padrão regex
                    if self.bool_search_settings['regex'] is True:                    
                        check_search_bool['regex'] = self.find_regex_in_sent(sent)
                    
                    #print(check_search_bool.values())
                    #caso todos os critérios de procura usados sejam verdadeiros
                    if False not in check_search_bool.values():                        
                        #checando se essa sentença faz parte de uma seção de metodologia (com a rede neural treinada)
                        proba_result = 1
                        proba_threshold = 0.5
                        if self.nn_model is True and self.ml_model is False:
                            proba_result = self.mc.check_sent_in_section_for_NN(index, DF = self.sentDF)
                        
                        elif self.nn_model is False and self.ml_model is True:
                            proba_result = self.mc.check_sent_in_section_for_ML(index, DF = self.sentDF)
                            #o threshold para o ML é maior devido à precisão mais baixa
                            proba_threshold = 0.7
                            
                        if proba_result >= proba_threshold:                                            
                            #print('Encontrado o termo primario: ', self.term_list_dic['primary'][term_index])
                            #parte do texto extraida para cada match
                            self.indexed_results[self.results_counter] = ([line, column], index, sent)
                            line += 1
                            self.results_counter +=1                    
                            print('*TERMS FOUND*')
                            print('Extracted text:')
                            print(sent)
                            #print('Indexes collected: ', index)
                            #time.sleep(2)
                            match_any_sent = True
    
                            #contadores para exportar nas DFs de report
                            self.search_report_dic['search']['total_finds'] += 1
                            if found_in_pdf is False:
                                self.search_report_dic['search']['pdf_finds'] += 1
                                found_in_pdf = True

                #definindo o número de linhas e colunas que formarão o DATAFRAME
                if line > self.DF_line_number:
                    self.DF_line_number = line
                self.DF_column_number = 2 + len(self.ID_columns)
                
                if match_any_sent is True:
                    #colocando os resultados na DF
                    self.put_search_results_in_DF(self.indexed_results)
                    
                #salvando o search report
                self.search_report_dic['search']['last_PDF_processed'] = self.filename
                save_dic_to_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.output_DF_name}_extracted_mode_{self.search_mode}.json', self.search_report_dic)
                              
                #fechando o arquivo h5
                h5.close()

            else:        
                print(f'O documento {filename} já foi processado.')
                print('Passando para o próximo documento...')
                continue
            
        if len(filenames_to_scan) == 0 or filename == filenames_to_scan[-1]:
            
            #mudando o status no LOG report
            self.search_report_dic['search']['searching_status'] = 'finished'
            save_dic_to_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.output_DF_name}_extracted_mode_{self.search_mode}.json', self.search_report_dic)
            
            #gerando o search report
            self.generate_search_report()



    def search_with_topic_match(self):

        print('( Function: search_with_topic_match )')
        
        import time
        import regex as re
        import numpy as np
        import pandas as pd
        import h5py
        import random
        from functions_STATS import calculate_crossentropy
        from functions_STATS import calculate_prob_dist_RSS
        from functions_VECTORS import normalize_1dvector
        from functions_VECTORS import get_tv_from_sent_index
        from FUNCTIONS import save_dic_to_json
            
        #checando os settings booleanos                
        if self.bool_search_settings['topic'] is True:
            pass
        else:
            print('ERRO! A função "search_with_largest_topic_val" precisa ter termos de procura por "tópico"')
            return            
                        
        #definindo a lista de documentos a ser escaneada
        if self.index_list_to_scan is not None:
            filenames_to_scan = self.index_list_to_scan
        else:
            filenames_to_scan = self.extracted_DB_documents[ self.lastfile_index : ]
        
        #varrendo os documentos .csv com as sentenças
        for filename in filenames_to_scan:
                        
            #check caso tenho sido encontrado algo no PDF
            found_in_pdf = False
            
            #checar se o PDF já não teve os termos extraidos
            if filename not in self.extracted_PDF_list:  
                
                print('------------------------------------------------------')
                print('Looking in ', filename, '...')
                self.filename = filename
                
                #zerando os contadores
                self.set_initial_counters()
                
                #abrindo o csv com as sentenças do artigo
                self.sentDF = pd.read_csv(self.diretorio + '/Outputs/Sentences_filtered/' + f'{self.filename}.csv', index_col = 0)

                #carregando a matriz DOC_TOPIC
                h5 = h5py.File(self.diretorio + f'/Outputs/TFIDF/H5/doc_topic_full_rNgram_{self.replace_Ngrams}.h5', 'r')
                DOC_TOPIC_matrix = h5['data']

                #numero da linha da entrada (as linhas separam o número de vezes que os termos foram encontrados no artigo)
                line = 0        
                #numero da coluna da entrada (considerando o número de colunas de index)
                column = 1 + len(self.ID_columns)
                
                #dicionário para coletar os valores de similaridades entre funções de probabilidade de tópicos
                sim_results_dic = {}                
                sim_results_dic['corr_coef'] = {}
                sim_results_dic['corr_coef']['overlap'] = {}
                sim_results_dic['corr_coef']['sum'] = {}
                
                sim_results_dic['cross_entropy'] = {}
                sim_results_dic['cross_entropy']['overlap'] = {}
                sim_results_dic['cross_entropy']['sum'] = {} 
                
                sim_results_dic['cdf_rss'] = {}
                sim_results_dic['cdf_rss']['overlap'] = {}
                sim_results_dic['cdf_rss']['sum'] = {}
                            
                #varrendo as sentenças
                for index in self.sentDF.index:

                    #carregando a sentença
                    sent = self.sentDF.loc[index, 'Sentence']
                    
                    #encontrando o comprimento máxiom da sentença
                    if len(sent) > self.max_sent_len:
                        self.max_sent_len = len(sent)
                
                    #pulando sentenças que possuem intervalos numéricos clusterizados (ex: 123&125&12321)
                    #esses padrões dão erro no regex
                    if re.search(r'–?[0-9]+(\.[0-9]+)?+\&–?[0-9]+(\.[0-9]+)?+\&–?[0-9]+(\.[0-9]+)?+', sent):
                        #print('Pulando a sentença: \n', sent_modified)
                        #time.sleep(1)
                        continue
                    
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
                        check_search_bool['semantic'] = self.search_with_terms(sent_modified, terms_type = 'semantic', use_term_prob_filter = False)
                    
                    #fazendo a procura com o classificador bayesiano
                    if self.bool_search_settings['bayesian'] is True:
                        check_search_bool['bayesian'] = self.search_sent_with_bayes_class(sent.lower()) 

                    #fazendo a procura com o padrão regex
                    if self.bool_search_settings['regex'] is True:                    
                        check_search_bool['regex'] = self.find_regex_in_sent(sent)
                          
                    #caso todos os matches tenham sido feitos
                    if False not in check_search_bool.values():
                        #print('sent match: ', sent)
                        #time.sleep(1)
                        #carregando o tópico vector da sentença
                        topic_vector = get_tv_from_sent_index(index, tv_stats = None, DOC_TOPIC_matrix = DOC_TOPIC_matrix, diretorio=self.diretorio)
                        #normalizando o topic vector (total p = 0)
                        topic_vector_norm = normalize_1dvector(topic_vector)        
                        if topic_vector_norm is None:
                            continue
                        #excluindo os valores negativos do topic_vector
                        topic_vector_pos = np.where( topic_vector > 0, topic_vector_norm , 0)
                        #normalizando o topic vector (total p = 0)
                        topic_vector_pos_norm = normalize_1dvector(topic_vector_pos)
                        if topic_vector_pos_norm is None:
                            continue
                        
                        #calculando a similaridade pelo coeficiente de correlação
                        #------------------------------
                        #para o vetor sum (normalizado)
                        corrcoef_sum = round(np.corrcoef(topic_vector_norm, self.topic_vector_sum_label)[0,1], 4)
                        if round(corrcoef_sum, 1) == 0.0:
                            pass
                        elif corrcoef_sum not in sim_results_dic['corr_coef']['sum'].keys():
                            sim_results_dic['corr_coef']['sum'][corrcoef_sum] = index
                        elif corrcoef_sum in sim_results_dic['corr_coef']['sum'].keys():
                            del(sim_results_dic['corr_coef']['sum'][corrcoef_sum])
        
                        #para o vetor overlap (positivo e normalizado)
                        corrcoef_overlap = round(np.corrcoef(topic_vector_pos_norm, self.topic_vector_overlap_label)[0,1], 4)
                        if round(corrcoef_overlap, 1) == 0.0:
                            pass
                        elif corrcoef_overlap not in sim_results_dic['corr_coef']['overlap'].keys():
                            sim_results_dic['corr_coef']['overlap'][corrcoef_overlap] = index
                        elif corrcoef_overlap in sim_results_dic['corr_coef']['overlap'].keys():
                            del(sim_results_dic['corr_coef']['overlap'][corrcoef_overlap])
                                    
                            
                        #calculando a similaridade pelo cross-entropy               
                        #------------------------------                    
                        #para o vetor soma (normalizado)
                        cross_entropy_sum = round(calculate_crossentropy(topic_vector_norm, self.topic_vector_sum_label), 4)
                        if round(cross_entropy_sum, 1) == 0.0:
                            pass
                        elif cross_entropy_sum not in sim_results_dic['cross_entropy']['sum'].keys():
                            sim_results_dic['cross_entropy']['sum'][cross_entropy_sum] = index
                        elif cross_entropy_sum in sim_results_dic['cross_entropy']['sum'].keys():
                            del(sim_results_dic['cross_entropy']['sum'][cross_entropy_sum])

                        #para o vetor overlap (positivo e normalizado)                        
                        cross_entropy_overlap = round(calculate_crossentropy(topic_vector_pos_norm, self.topic_vector_overlap_label), 4)
                        if round(cross_entropy_overlap, 1) == 0.0:
                            pass
                        elif cross_entropy_overlap not in sim_results_dic['cross_entropy']['overlap'].keys():
                            sim_results_dic['cross_entropy']['overlap'][cross_entropy_overlap] = index
                        elif cross_entropy_overlap in sim_results_dic['cross_entropy']['overlap'].keys():
                            del(sim_results_dic['cross_entropy']['overlap'][cross_entropy_overlap])

                        
                        #calculando a similaridade pelo RSS (residual sum of squares) das CDFs de cada probability vector
                        #------------------------------                    
                        #para o vetor soma (normalizado)
                        cdf_rss_sum = round(calculate_prob_dist_RSS(topic_vector_norm, self.topic_vector_sum_label), 4)
                        if round(cdf_rss_sum, 1) == 0.0:
                            pass
                        elif cdf_rss_sum not in sim_results_dic['cdf_rss']['sum'].keys():
                            sim_results_dic['cdf_rss']['sum'][cdf_rss_sum] = index
                        elif cdf_rss_sum in sim_results_dic['cdf_rss']['sum'].keys():
                            del(sim_results_dic['cdf_rss']['sum'][cdf_rss_sum])

                        #para o vetor overlap (positivo e normalizado)                        
                        cdf_rss_overlap = round(calculate_prob_dist_RSS(topic_vector_pos_norm, self.topic_vector_overlap_label), 4)
                        if round(cdf_rss_overlap, 1) == 0.0:
                            pass
                        elif cdf_rss_overlap not in sim_results_dic['cdf_rss']['overlap'].keys():
                            sim_results_dic['cdf_rss']['overlap'][cdf_rss_overlap] = index
                        elif cdf_rss_overlap in sim_results_dic['cdf_rss']['overlap'].keys():
                            del(sim_results_dic['cdf_rss']['overlap'][cdf_rss_overlap])


                    #Exportando sentenças randômicas
                    else:
                        if (self.export_random_sent is True):
                            if self.filename not in self.rand_extracted_PDF_list:
                                if random.random() > 0.99:
                                    self.rand_sent_DF.loc[ (self.filename , self.rand_sent_counter ) , 'rand_sent' ] = sent
                                    self.rand_sent_DF.loc[ (self.filename , self.rand_sent_counter ) , 'rand_sent_index' ] = index
                                    self.rand_sent_counter += 1
                        
                
                                
                #selecionando os resultados
                #-------------------------------                               
                #determina a função de correlação usada para avaliar
                correlation_function_used = 'corr_coef'
                collected_indexes = []
                collected_match_vals = []
                collected_match_type = []                
                
                #caso o método de match seja para encontrar somente a sentença com maior valor de correlação
                if self.topic_match_mode.lower() == 'largest':

                    #Encontrando os maiores valores de match para os tópicos
                    max_match_val = {}
                    max_match_val['corr_coef'] = {}
                    max_match_val['cross_entropy'] = {}
                    max_match_val['cdf_rss'] = {}
                    
                    best_match_index = {}
                    best_match_index['corr_coef'] = {}
                    best_match_index['cross_entropy'] = {}
                    best_match_index['cdf_rss'] = {}
                    
                    try:
                        max_match_val['corr_coef']['overlap'] = max(sim_results_dic['corr_coef']['overlap'].keys())
                        best_match_index['corr_coef']['overlap'] = sim_results_dic['corr_coef']['overlap'][ max_match_val['corr_coef']['overlap'] ]
                    #caso nenhum match tenha sido encontrado
                    except ValueError:
                        max_match_val['corr_coef']['overlap'] = 0
                        best_match_index['corr_coef']['overlap'] = None
        
                    try:                
                        max_match_val['corr_coef']['sum'] = max(sim_results_dic['corr_coef']['sum'].keys())
                        best_match_index['corr_coef']['sum'] = sim_results_dic['corr_coef']['sum'][ max_match_val['corr_coef']['sum'] ]
                    #caso nenhum match tenha sido encontrado
                    except ValueError:
                        max_match_val['corr_coef']['sum'] = 0
                        best_match_index['corr_coef']['sum'] = None
                    
                    try:
                        max_match_val['cross_entropy']['overlap'] = max(sim_results_dic['cross_entropy']['overlap'].keys())
                        best_match_index['cross_entropy']['overlap'] = sim_results_dic['cross_entropy']['overlap'][ max_match_val['cross_entropy']['overlap'] ]
                    #caso nenhum match tenha sido encontrado
                    except ValueError:
                        max_match_val['cross_entropy']['overlap'] = 0
                        best_match_index['cross_entropy']['overlap'] = None
                    
                    try:
                        max_match_val['cross_entropy']['sum'] = max(sim_results_dic['cross_entropy']['sum'].keys())
                        best_match_index['cross_entropy']['sum'] = sim_results_dic['cross_entropy']['sum'][ max_match_val['cross_entropy']['sum'] ]
                    #caso nenhum match tenha sido encontrado
                    except ValueError:
                        max_match_val['cross_entropy']['sum'] = 0
                        best_match_index['cross_entropy']['sum'] = None
                    
                    try:
                        max_match_val['cdf_rss']['overlap'] = max(sim_results_dic['cdf_rss']['overlap'].keys())
                        best_match_index['cdf_rss']['overlap'] = sim_results_dic['cdf_rss']['overlap'][ max_match_val['cdf_rss']['overlap'] ]
                    #caso nenhum match tenha sido encontrado
                    except ValueError:
                        max_match_val['cdf_rss']['overlap'] = 0
                        best_match_index['cdf_rss']['overlap'] = None
                
                    try:
                        max_match_val['cdf_rss']['sum'] = max(sim_results_dic['cdf_rss']['sum'].keys())
                        best_match_index['cdf_rss']['sum'] = sim_results_dic['cdf_rss']['sum'][ max_match_val['cdf_rss']['sum'] ]
                    #caso nenhum match tenha sido encontrado
                    except ValueError:
                        max_match_val['cdf_rss']['sum'] = 0
                        best_match_index['cdf_rss']['sum'] = None
                    
                    #encontrando o maior valor de corr_coef
                    largest_match_result = 0
                    if max_match_val[correlation_function_used]['overlap'] > 0 and max_match_val[correlation_function_used]['overlap'] > max_match_val[correlation_function_used]['sum']:
                        largest_match_result = max_match_val[correlation_function_used]['overlap']
                        if best_match_index[correlation_function_used]['overlap'] is not None:
                            collected_indexes.append(best_match_index[correlation_function_used]['overlap'])
                            collected_match_vals.append(largest_match_result)
                            collected_match_type.append('overlap')
                    elif max_match_val[correlation_function_used]['sum'] > 0 and max_match_val[correlation_function_used]['sum'] > max_match_val[correlation_function_used]['overlap']:
                        largest_match_result = max_match_val[correlation_function_used]['sum']
                        if best_match_index[correlation_function_used]['sum'] is not None:
                            collected_indexes.append(best_match_index[correlation_function_used]['sum'])
                            collected_match_vals.append(largest_match_result)
                            collected_match_type.append('sum')
                        
                    #caso algum dos coeficientes de correlação seja maior que o threshold                
                    cond1 = largest_match_result >= self.min_topic_corr_threshold
                    #checando se o index foi coletado                   
                    cond2 = len(collected_indexes) > 0

                    #imprimindo os resultados
                    #if False not in (cond1, cond2):                    
                    #    for result in ['corr_coef']: #('cross_entropy', 'cdf_rss'):
                    #        print(f'\nResults {result}:')
                    #        print('\n  Best overlap: ', max_match_val[result]['overlap'], '( sent_index: ', best_match_index[result]['overlap'], ' )')
                    #        print('  Sent: ', self.sentDF.loc[ best_match_index[result]['overlap'] , 'Sentence' ] )
                    #        print('\n  Best sum: ', max_match_val[result]['sum'], '( sent_index: ', best_match_index[result]['sum'], ' )')
                    #        print('  Sent: ', self.sentDF.loc[ best_match_index[result]['sum'] , 'Sentence' ] )
                        

                #caso o método de match seja para encontrar todas as sentenças com valor de correlação maior que o threshold
                elif self.topic_match_mode.lower() == 'threshold':
                    
                    match_vals = list(sim_results_dic[correlation_function_used]['overlap'].keys()) + list(sim_results_dic[correlation_function_used]['sum'].keys())
                    for match_val in match_vals:
                        if match_val >= self.min_topic_corr_threshold:
                            try:
                                sent_index = sim_results_dic[correlation_function_used]['overlap'][match_val]
                                if sent_index not in collected_indexes:
                                    collected_indexes.append( sent_index )
                                    collected_match_vals.append(match_val)
                                    collected_match_type.append('overlap')
                            except KeyError:
                                pass
                            try:
                                sent_index = sim_results_dic[correlation_function_used]['sum'][match_val]
                                if sent_index not in collected_indexes:
                                    collected_indexes.append(sent_index)
                                    collected_match_vals.append(match_val)
                                    collected_match_type.append('sum')
                            except KeyError:
                                pass
                    
                    #a condição 1 só é usada para o modo "largest"
                    cond1 = True                    
                    #checando se o index foi coletado
                    cond2 = len(collected_indexes) > 0
                
                #consolidando na DF
                if False not in (cond1, cond2):
                    
                    for i in range(len(collected_indexes)):
                    
                        #coletando os resultados encontrados
                        match_sent = self.sentDF.loc[ collected_indexes[i] , 'Sentence']
                        #checando se essa sentença faz parte de uma seção de metodologia (com a rede neural treinada)
                        proba_result = 1
                        proba_threshold = 0.5
                        if self.nn_model is True and self.ml_model is False:
                            proba_result = self.mc.check_sent_in_section_for_NN(collected_indexes[i], DF = self.sentDF)
                        
                        elif self.nn_model is False and self.ml_model is True:
                            proba_result = self.mc.check_sent_in_section_for_ML(collected_indexes[i], DF = self.sentDF)
                            #o threshold para o ML é maior devido à precisão mais baixa
                            proba_threshold = 0.7
                            
                        if proba_result >= proba_threshold:
                            print('\n> Extracted Sentence: \n> ', match_sent)
                            print(' > match: ', collected_match_vals[i], ' ; type: ', collected_match_type[i])
                            self.indexed_results[self.results_counter] = ([line, column], collected_indexes[i] , match_sent)
                            self.results_counter += 1
                            line +=1
                            time.sleep(1)
            
                            #definindo o número de linhas e colunas que formarão o DATAFRAME
                            if line > self.DF_line_number:
                                self.DF_line_number = line
                            self.DF_column_number = 2 + len(self.ID_columns)
                                                    
                            #contadores para exportar nas DFs de report
                            self.search_report_dic['search']['total_finds'] += 1
                            if found_in_pdf is False:
                                self.search_report_dic['search']['pdf_finds'] += 1
                                found_in_pdf = True
                    
                    #colocando os resultados na DF
                    try:
                        self.put_search_results_in_DF(self.indexed_results)
                    except AttributeError:
                        pass

                #salvando o search report
                self.search_report_dic['search']['last_PDF_processed'] = self.filename
                save_dic_to_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.output_DF_name}_extracted_mode_{self.search_mode}.json', self.search_report_dic)

                #fechando o arquivo h5
                h5.close()

            else:
                print(f'O documento {filename} já foi processado.')
                print('Passando para o próximo documento...')
                continue
            
        if len(filenames_to_scan) == 0 or filename == filenames_to_scan[-1]:
            
            #mudando o status no LOG report
            self.search_report_dic['search']['searching_status'] = 'finished'
            save_dic_to_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.output_DF_name}_extracted_mode_{self.search_mode}.json', self.search_report_dic)
            
            #gerando o search report
            self.generate_search_report()


    def set_initial_counters(self):

        self.indexed_results = {}
        #contador para o dicionário com as sentenças nas quais os termos foram encontrados
        self.results_counter = 1
        #contador para sentenças randômicas extraídas
        self.rand_sent_counter = 0
        #número de linhas da matrix (começa com zero)
        self.DF_line_number = 0
        #determinando o tamanho (length) da maior sentença do PDF (esse valor determinará o dtype da np.array abaixo)
        self.max_sent_len = 0



    def open_DFs(self):

        print('( Function: open_DFs )')
        
        #checando erros de instanciação/inputs
        #import time
        import os
        import pandas as pd
        from FUNCTIONS import error_print_abort_class
        from FUNCTIONS import create_PDFFILE_list
        from FUNCTIONS import load_dic_from_json
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return

        #checando o diretorio EXTRACTED
        if not os.path.exists(self.diretorio + '/Outputs/Extracted'):
            os.makedirs(self.diretorio + '/Outputs/Extracted')                             

        #checando os PDF nos quais foram extraidos as sentenças
        self.extracted_PDF_list = []
        #checar a existência de DF de fragmentos
        self.ID_columns = ['Filename', 'Counter']
        if os.path.exists(self.diretorio + f'/Outputs/Extracted/{self.output_DF_name}_extracted_mode_{self.search_mode}.csv'):
            self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/Extracted/{self.output_DF_name}_extracted_mode_{self.search_mode}.csv', index_col=[0,1])
            last_pdf_file_processed = self.output_DF.index.levels[0].values[-1]
            self.extracted_PDF_list = create_PDFFILE_list(last_pdf_file_processed)
            #print(self.extracted_PDF_list)
        else:
            #criando a coluna de indexação do DF de fragmentos
            self.columns = self.ID_columns + [ self.output_DF_name ] + [term + '_index' for term in [ self.output_DF_name ] ]
            self.output_DF = pd.DataFrame(columns=self.columns)
            self.output_DF.set_index(['Filename', 'Counter'], inplace=True)
                
        #checando os PDF nos quais foram extraidos as sentenças randômicos
        #lista para extrações randômicas
        self.rand_extracted_PDF_list = []
        if (self.export_random_sent is True):
            #checar a existência de DF de fragmentos randomicos extraídos
            if os.path.exists(self.diretorio + f'/Outputs/Extracted/{self.output_DF_name}_extracted_mode_{self.search_mode}_random.csv'):
                self.rand_sent_DF = pd.read_csv(self.diretorio + f'/Outputs/Extracted/{self.output_DF_name}_extracted_mode_{self.search_mode}_random.csv', index_col=[0,1])
                self.rand_extracted_PDF_list = self.rand_sent_DF.index.levels[0].values
            else:
                #criando a coluna de indexação do DF de fragmentos
                self.columns = self.ID_columns + ['rand_sent', 'rand_sent_index']
                self.rand_sent_DF = pd.DataFrame([], columns=self.columns)
                self.rand_sent_DF.set_index(['Filename', 'Counter'], inplace=True)
        
        #abrindo o search report (para parar e resumir o fit)
        if os.path.exists(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.output_DF_name}_extracted_mode_{self.search_mode}.json'):
            #carregando o dicionário
            self.search_report_dic = load_dic_from_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.output_DF_name}_extracted_mode_{self.search_mode}.json')
            #último filename processado
            file_name = self.search_report_dic['search']['last_PDF_processed']
            #index do ultimo filename processado
            self.lastfile_index = self.extracted_DB_documents.index(file_name) + 1
            print('Searching_counter_File_index_counter_File_index: ', file_name)
        else:
            self.search_report_dic = {}
            self.search_report_dic['search'] = {}
            self.search_report_dic['search']['last_PDF_processed'] = None
            self.search_report_dic['search']['total_finds'] = 0
            self.search_report_dic['search']['pdf_finds'] = 0
            self.search_report_dic['search']['searching_status'] = 'ongoing'
            self.lastfile_index = 0


    def search_with_terms(self, text, terms_type = 'literal', use_term_prob_filter = False):
            
        import random
        import regex as re
        import time
        
        #check caso os termos sejam encontrados na sentença
        found_terms = False        
            
        #procurando pelos termos primários no texto
        for i in range(len(self.term_list_dic[terms_type]['primary'])):
            #testando a probabilidade de apariçaõ do termo
            #print('Term Primary: ', self.term_list_dic[terms_type]['primary'][i])
            #print('Prob: ', self.term_prob_list_dic['primary'][i])
            if terms_type == 'semantic':
                if use_term_prob_filter is True:
                    random_prob = random.uniform(0,1)
                    #print('random_prob: ', random_prob)                                    
                    if random_prob > self.term_prob_list_dic['primary'][i]:
                        #print('Pulando o termo primaŕio: ', self.term_list_dic[terms_type]['primary'][i])
                        continue
            #se foi encontrado algum termo primário
            term_to_find = re.sub(r'\(', '\(', self.term_list_dic[terms_type]['primary'][i])
            if re.search( term_to_find , text ):
                #print('Encontrado o termo primario: ', self.term_list_dic[terms_type]['primary'][i])
                found_terms = True
                break

        #procurando os termos secundários                                     
        if (found_terms is True):
            #dicionário para coletar as operações + e - feitas com os termos secundários
            check_sec_term_operation = {}                               
            for j in range(len(self.term_list_dic[terms_type]['secondary'])):
                #assumindo primariamente que nenhum termo foi encontrado para esse conjunto de termos secundários [j]
                check_sec_term_operation[j] = '-'
                #procurando cada termo secundário dentro do grupo de secundários
                for sec_term_index in range(len(self.term_list_dic[terms_type]['secondary'][j])):
                    #testando a probabilidade de apariçaõ do termo
                    #print('Term Secondary: ', self.term_list_dic[terms_type]['secondary'][j][sec_term_index])
                    #print('Prob: ', self.term_prob_list_dic['secondary'][j][sec_term_index])
                    if terms_type == 'semantic':
                        if use_term_prob_filter is True:
                            random_prob = random.uniform(0,1)
                            #print('random_prob: ', random_prob)
                            if random_prob > self.term_prob_list_dic['secondary'][j][sec_term_index]:
                                #print('Pulando o termo secundário: ', self.term_list_dic[terms_type]['secondary'][j][sec_term_index])
                                continue
                    
                    term_to_find = re.sub(r'\(', '\(' , self.term_list_dic[terms_type]['secondary'][j][sec_term_index])
                    if re.search( term_to_find , text ):
                        #print('Encontrado o termo secundário: ', self.term_list_dic[terms_type]['secondary'][j][sec_term_index])
                        check_sec_term_operation[j] = '+'
                        break
            
            #caso as operações de termos secundários sejam as mesmas inseridas                        
            if list(check_sec_term_operation.values()) == self.term_list_dic[terms_type]['operations']:
                pass
            else:
                found_terms = False
                                
        #caso todos os termos tenham sido encontrados
        if found_terms is True:
            #print('- match: search_with_terms')
            return True
        else:
            return False


    '''
    def search_sent_with_topic_vectors(self, index, DOC_TOPIC_matrix, corr_threshold = 0.1):
        
        import numpy as np
        from functions_VECTORS import get_topic_vec_from_sent_index
        from functions_VECTORS import normalize_1dvector
        
        #carregando o tópico vector da sentença
        topic_vector = get_topic_vec_from_sent_index(index, DOC_TOPIC_matrix = DOC_TOPIC_matrix, diretorio=self.diretorio)
        #normalizando o topic vector (total p = 0)
        topic_vector_norm = normalize_1dvector(topic_vector)
        if topic_vector_norm is None:
            return False
        
        #excluindo os valores negativos do topic_vector
        topic_vector_pos = np.where( topic_vector > 0, topic_vector_norm , 0)
        #normalizando o topic vector (total p = 0)
        topic_vector_pos_norm = normalize_1dvector(topic_vector_pos)
        if topic_vector_pos_norm is None:
            return False
        
        #calculando a similaridade pelo coeficiente de correlação
        #------------------------------
        #para o vetor soma (normalizado)
        corrcoef_sum = np.corrcoef(topic_vector_norm, self.topic_vector_sum_label)[0,1]
        #para o vetor overlap (positivo e normalizado)
        corrcoef_overlap = np.corrcoef(topic_vector_pos_norm, self.topic_vector_overlap_label)[0,1]
        
        #checando se a correlação entre os tópicos são maiores que o threshold        
        if corrcoef_sum > corr_threshold:
            #print('- match: topic vector')
            return True
        elif corrcoef_overlap > corr_threshold:
            #print('- match: topic vector')
            return True
        else:
            return False'''



    def search_sent_with_bayes_class(self, text):
        
        if self.stats.use_bayes_classifier(text) > 0.99:
            #print('- match: bayesian classifier')
            return True
        else:
            return False



    def find_regex_in_sent(self, text):
            
        import regex as re
        
        #se foi encontrado algum regex
        if re.search( self.term_list_dic['regex']['regex_pattern'] , text ):
            #print('- match: regex')
            return True
        else:
            return False



    def combine_topic_vectors(self):
    
        import time
        import numpy as np
        from functions_VECTORS import normalize_1dvector
        from FUNCTIONS import load_dic_from_json

        #carrengando o dicionário com os valores de probabilidades de cada significado semântico
        topic_vectors_dic = load_dic_from_json(self.diretorio + '/Outputs/Models/sem_topic_vectors.json')
    
        #vetor de topico primário
        primary_topic = self.term_list_dic['topic']['primary'][0]
        self.topic_vector_overlap_label = np.array(topic_vectors_dic[primary_topic]['topic_overlap'])
        self.topic_vector_sum_label = np.array(topic_vectors_dic[primary_topic]['topic_sum'])
        print('Primary topic vector definido: ', primary_topic)
        
        #varrendo os termos secundários
        for secondary_N in range(len(self.term_list_dic['topic']['secondary'])):
        
            #determinando o tópico secundário
            secondary_topic = self.term_list_dic['topic']['secondary'][secondary_N][0]
            secondary_topic_vector_overlap_label = np.array(topic_vectors_dic[secondary_topic]['topic_overlap'])
            secondary_topic_vector_sum_label = np.array(topic_vectors_dic[secondary_topic]['topic_sum'])            

            #determinando a operação
            topic_operation = self.term_list_dic['topic']['operations'][secondary_N]
            
            #somando os tópicos
            if topic_operation == '+':                
                self.topic_vector_overlap_label = self.topic_vector_overlap_label + secondary_topic_vector_overlap_label
                self.topic_vector_sum_label = self.topic_vector_sum_label + secondary_topic_vector_sum_label
                print('Somando secondary topic vectors: ', secondary_topic)
                
            #subtraindo os tópicos
            elif topic_operation == '-':                    
                self.topic_vector_overlap_label = self.topic_vector_overlap_label - secondary_topic_vector_overlap_label
                self.topic_vector_sum_label = self.topic_vector_sum_label - secondary_topic_vector_sum_label
                print('Subtraindo secondary topic vectors: ', secondary_topic)
        
        #normalizando os vectores tópicos
        self.topic_vector_overlap_label = normalize_1dvector(self.topic_vector_overlap_label)
        self.topic_vector_sum_label = normalize_1dvector(self.topic_vector_sum_label)
        print('Normalizando topic vectors...')
        time.sleep(2)



    def generate_search_report(self):
        
        import os
        import pandas as pd
            
        #abrindo o SE report            
        if os.path.exists(self.diretorio + '/Settings/SE_inputs.csv'):
            search_report_DF = pd.read_csv(self.diretorio + '/Settings/SE_inputs.csv', index_col = 0)

            search_report_DF.loc[self.output_DF_name , 'total_finds' ] = self.search_report_dic['search']['total_finds']
            search_report_DF.loc[self.output_DF_name , 'pdf_finds' ] = self.search_report_dic['search']['pdf_finds']
            search_report_DF.loc[self.output_DF_name , 'search_status' ] = 'finished'

            search_report_DF.sort_index(inplace=True)
            search_report_DF.to_csv(self.diretorio + '/Settings/SE_inputs.csv')
            print('Salvando o SE report em ~/Settings/SE_inputs.csv')


    
    def put_search_results_in_DF(self, indexed_results):

        import pandas as pd
        import numpy as np

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
        self.output_DF.to_csv(self.diretorio + f'/Outputs/Extracted/{self.output_DF_name}_extracted_mode_{self.search_mode}.csv')
        #exportando sentenças randômicas
        if (self.export_random_sent is True):
            self.rand_sent_DF.to_csv(self.diretorio + f'/Outputs/Extracted/{self.output_DF_name}_extracted_mode_{self.search_mode}_random.csv')
