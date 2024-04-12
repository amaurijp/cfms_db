#!/usr/bin/env python3
# -*- coding: utf-8 -*-    

import time
import os
import pandas as pd
import h5py
import json
import numpy as np
import regex as re
from multiprocessing import Pool
from functions_VECTORS import get_vector_from_string
from functions_PARAMETERS import regex_patt_from_parameter
from functions_TOKENS import get_nGrams_list

#------------------------------
def create_article_file_list_with_filename(article_filename):
    
    if (article_filename == 'ATC00000'):
        return []
    else:
        match = re.search('[1-9]{1,5}[0-9]*', article_filename)
        file_number = int(match.group())
        return [ get_tag_name(file_N) for file_N in list(range(file_number + 1)) ]


#------------------------------
def create_h5_matrix(shape = None, filepath = None, dtype=None):

    print('Criando o arquivo h5 em: ', filepath)
    h5_file = h5py.File(filepath, 'w')
    h5_file.create_dataset('data', shape = shape, dtype=dtype)
    h5_matrix = h5_file['data']
    print('H5 matrix created - shape:', h5_matrix.shape)

    return h5_file, h5_matrix


#------------------------------
def error_incompatible_strings_input(input_name, given_input, available_inputs, class_name = ''):
    
    abort_class = False
    
    if given_input.lower() not in available_inputs:
        print(f'Erro para a entrada {input_name}: {given_input}')
        print(f'Selecionar uma entrada adequada para o {input_name} (ver abaixo).')
        print('Entradas disponíveis: ')
        for av_input in available_inputs:
            print(av_input)
        print(f'> Abortando a classe: {class_name}')
        abort_class = True
        
    return abort_class


#------------------------------
def error_print_abort_class(class_name):
        
    print('Erro na instanciação da classe.')
    print(f'> Abortando a classe: {class_name}')


#------------------------------
def extract_inputs_from_csv(csv_filename = '', diretorio = None, mode = 'search_extract'):

    inputs_DF = pd.read_csv(diretorio + f'/Settings/{csv_filename}.csv')
    
    dic = {}
    #varrendo a DF
    if mode.lower() == 'search_extract':
        for line in inputs_DF.index:
            
            dic[line] = {}
            
            #filename
            dic[line]['filename'] = inputs_DF.loc[ line , 'filename' ]

            #index_list to search
            if inputs_DF.loc[ line , 'index_list' ].lower() == 'none':
                dic[line]['index_list'] = None
            else:
                dic[line]['index_list'] = inputs_DF.loc[ line , 'index_list' ]
    
            #wholetext or sent
            if str(inputs_DF.loc[ line , 'scan_sent_by_sent']).lower() == 'false':
                dic[line]['scan_sent_by_sent'] = False
            else:
                dic[line]['scan_sent_by_sent'] = True

            #parameter to extract
            dic[line]['parameter_to_extract'] = inputs_DF.loc[ line , 'parameter_to_extract' ]
            
            #lower sentence para extrair
            if str(inputs_DF.loc[ line , 'lower_sentence_to_extract_parameters' ]).lower() == 'false':
                dic[line]['lower_sentence_to_extract_parameters'] = False
            else:
                dic[line]['lower_sentence_to_extract_parameters'] = True

            #ngrams_to_extract
            if str(inputs_DF.loc[ line , 'ngrams_to_extract' ]) in ('0', '1', '2', 'x'):
                dic[line]['ngrams_to_extract'] = inputs_DF.loc[ line , 'ngrams_to_extract' ]
            else:
                dic[line]['ngrams_to_extract'] = None
            
            #ngrams_min_app
            if int(inputs_DF.loc[ line , 'ngrams_min_app' ]) in range(100):
                dic[line]['ngrams_min_app'] = int(inputs_DF.loc[ line , 'ngrams_min_app' ])
            else:
                dic[line]['ngrams_min_app'] = None
        
            #dicionário com os inputs de search
            dic[line]['search_inputs'] = {}
            
            #literal
            if inputs_DF.loc[ line , 'literal_entry' ].lower() == 'none':
                dic[line]['search_inputs']['literal'] = '()'
            else:
                dic[line]['search_inputs']['literal'] = inputs_DF.loc[ line , 'literal_entry' ]
            
            #regex mode para literal
            if str(inputs_DF.loc[ line , 'regex_mode_for_literal' ]).lower() == 'true':
                dic[line]['search_inputs']['regex_mode_for_literal'] = True
            else:
                dic[line]['search_inputs']['regex_mode_for_literal'] = False
            
            #semantic
            if inputs_DF.loc[ line , 'semantic_entry' ].lower() == 'none':
                dic[line]['search_inputs']['semantic'] = '()'
            else:
                dic[line]['search_inputs']['semantic'] = inputs_DF.loc[ line , 'semantic_entry' ]

            #lower sentence para procurar termos com similaridade semântica
            if str(inputs_DF.loc[ line , 'lower_sent_to_semantic_search' ]).lower() == 'false':
                dic[line]['search_inputs']['lower_sent_to_semantic_search'] = False
            else:
                dic[line]['search_inputs']['lower_sent_to_semantic_search'] = True
    
            dic[line]['search_inputs']['semantic_ngrams'] = []
            ngrams_list = re.findall(r'[0-9x]+', str(inputs_DF.loc[ line , 'semantic_ngrams' ]))
            for i in ngrams_list:
                if str(i) in ('0', '1', '2', 'x'):
                    dic[line]['search_inputs']['semantic_ngrams'].append( i )
                else:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('O "semantic_ngrams" suporta três valores (0, 1, 2 e x) na forma (0, 1, 1)')
                    print('Valor de entrada: ', inputs_DF.loc[ line , 'semantic_ngrams' ])
            
            dic[line]['search_inputs']['semantic_ngrams_min_app'] = []
            min_app_list = re.findall(r'[0-9]+', str(inputs_DF.loc[ line , 'semantic_ngrams_min_app' ]))
            for number in min_app_list:
                try:
                    dic[line]['search_inputs']['semantic_ngrams_min_app'].append( int(number) )
                except:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('O "semantic_ngrams_min_app" suporta somente com valores númericos do tipo (6,50,30).')
                    print('Valor de entrada: ', inputs_DF.loc[ line , 'semantic_ngrams_min_app' ])        
            
            #topic
            if inputs_DF.loc[ line , 'lda_sents_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lda_sents_topic'] = None
            else:
                dic[line]['search_inputs']['lda_sents_topic'] = inputs_DF.loc[ line , 'lda_sents_topic' ]

            if inputs_DF.loc[ line , 'lda_articles_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lda_articles_topic'] = None
            else:
                dic[line]['search_inputs']['lda_articles_topic'] = inputs_DF.loc[ line , 'lda_articles_topic' ]

            if inputs_DF.loc[ line , 'lsa_sents_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lsa_sents_topic'] = None
            else:
                dic[line]['search_inputs']['lsa_sents_topic'] = inputs_DF.loc[ line , 'lsa_sents_topic' ]

            if inputs_DF.loc[ line , 'lsa_articles_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lsa_articles_topic'] = None
            else:
                dic[line]['search_inputs']['lsa_articles_topic'] = inputs_DF.loc[ line , 'lsa_articles_topic' ]

            if str( inputs_DF.loc[ line , 'cos_thr' ] ).lower() == 'none':
                dic[line]['search_inputs']['cos_thr'] = None
            else:
                dic[line]['search_inputs']['cos_thr'] = float( inputs_DF.loc[ line , 'cos_thr' ] )
            
            #regex
            if inputs_DF.loc[ line , 'regex_entry' ].lower() == 'none':
                dic[line]['search_inputs']['regex'] = ''
            else:
                dic[line]['search_inputs']['regex'] = inputs_DF.loc[ line , 'regex_entry' ]
    
            #filter section
            if inputs_DF.loc[ line , 'filter_section' ].lower() == 'none':
                dic[line]['search_inputs']['filter_section'] = None
            else:    
                if inputs_DF.loc[ line , 'filter_section' ].lower() in ('introduction', 'methodology', 'results'):
                    dic[line]['search_inputs']['filter_section'] = inputs_DF.loc[ line , 'filter_section' ].lower()
                else:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('A entrada de "filter_section" não é compatível.')
                    print('Entradas compatíveis: "introduction", "methodology", "results"')
                    print('Valor de entrada: ', inputs_DF.loc[ line , 'filter_section' ])
            
            #status
            for status_input in ('search_status', 'export_status'):
                try:
                    if inputs_DF.loc[ line , status_input ].lower() != 'finished':
                        dic[line][status_input] = 'ongoing'
                    else:    
                        dic[line][status_input] = inputs_DF.loc[ line , status_input ]
                    
                except (KeyError, AttributeError):
                    dic[line][status_input] = 'ongoing'
    
    elif mode.lower() == 'consolidate_df':
        for line in inputs_DF.index:
            
            dic[line] = {}
            
            #filename
            dic[line]['filename'] = inputs_DF.loc[ line , 'filename' ]

            #search mode
            dic[line]['search_mode'] = inputs_DF.loc[ line , 'search_mode' ]
    
            #parameter to extract
            dic[line]['parameter_to_extract'] = inputs_DF.loc[ line , 'parameter_to_extract' ]
            
            #mode to extract
            dic[line]['extraction_mode'] = inputs_DF.loc[ line , 'extraction_mode' ]

            #lower sentence para extrair
            if str(inputs_DF.loc[ line , 'lower_sentence_to_extract_parameters' ]).lower() == 'false':
                dic[line]['lower_sentence_to_extract_parameters'] = False
            else:
                dic[line]['lower_sentence_to_extract_parameters'] = True
                    
            #hold_filenames
            if str(inputs_DF.loc[ line , 'hold_filenames' ]).lower() == 'true':
                dic[line]['hold_filenames'] = True
            elif str(inputs_DF.loc[ line , 'hold_filenames' ]).lower() == 'false':
                dic[line]['hold_filenames'] = False
            else:
                dic[line]['hold_filenames'] = False
    
            #hold_instances_number
            if str(inputs_DF.loc[ line , 'hold_instances_number' ]).lower() == 'true':
                dic[line]['hold_instances_number'] = True
            elif str(inputs_DF.loc[ line , 'hold_instances_number' ]).lower() == 'false':
                dic[line]['hold_instances_number'] = False
            else:
                dic[line]['hold_instances_number'] = False

            #numbers_extraction_mode
            if str(inputs_DF.loc[ line , 'numbers_extraction_mode' ]).lower() in ('all', 'one'):
                dic[line]['numbers_extraction_mode'] = inputs_DF.loc[ line , 'numbers_extraction_mode' ].lower()
            else:
                dic[line]['numbers_extraction_mode'] = 'all'

            #cluster_min_max_num_vals
            if str(inputs_DF.loc[ line , 'cluster_min_max_num_vals' ]).lower() == 'true':
                dic[line]['cluster_min_max_num_vals'] = True
            elif str(inputs_DF.loc[ line , 'cluster_min_max_num_vals' ]).lower() == 'false':
                dic[line]['cluster_min_max_num_vals'] = False
            else:
                dic[line]['cluster_min_max_num_vals'] = False

            #filter_unique_results
            if str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'true':
                dic[line]['filter_unique_results'] = True
            elif str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'false':
                dic[line]['filter_unique_results'] = False
            else:
                dic[line]['filter_unique_results'] = False
                
            #ngram_for_textual_search
            if str(inputs_DF.loc[ line , 'ngram_for_textual_search' ]).lower() != 'none' and int(inputs_DF.loc[ line , 'ngram_for_textual_search' ]) in (0, 1, 2):
                dic[line]['ngram_for_textual_search'] = int(inputs_DF.loc[ line , 'ngram_for_textual_search' ])
            else:
                dic[line]['ngram_for_textual_search'] = 0

            #min_ngram_appearence
            if str(inputs_DF.loc[ line , 'min_ngram_appearence' ]).lower() != 'none' and int(inputs_DF.loc[ line , 'min_ngram_appearence' ]) in range(100):
                dic[line]['min_ngram_appearence'] = int(inputs_DF.loc[ line , 'min_ngram_appearence' ])
            else:
                dic[line]['min_ngram_appearence'] = 1

    
    #print(dic)
    return dic


#------------------------------
def filename_gen():
    import random as rdn
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    new_file_name=''
    counter = 0
    while counter < 20:
        index = rdn.randint(0, 61)
        new_file_name += letters[index]
        counter += 1
    return new_file_name


#------------------------------
def generate_PR_results(prediction_list, target_val_list, proba_threshold = 0.5):
        
    #print(prediction_list)
    #print(target_val_list)
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i in range(len(prediction_list)):
        
        result = prediction_list[i]
        target = target_val_list[i]
        
        if result[1] >= proba_threshold and int(target) == 1:
            true_positives += 1
        elif result[1] >= proba_threshold and int(target) == 0:
            false_positives += 1
        elif result[1] < proba_threshold and int(target) == 1:
            false_negatives += 1
        elif result[1] < proba_threshold and int(target) == 0:
            true_negatives += 1
                
    try:
        precision = true_positives / ( true_positives + false_positives )
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / ( true_positives + false_negatives )
    except ZeroDivisionError:
        recall = 0
    
    return precision, recall        


#------------------------------
def generate_term_list(terms_in_text_file):

    term_list = []
    for line in terms_in_text_file:
        #print(line)
        term = ''
        for char in line:
            if char == '\n':
                continue
            else:
                term += char
        term_list.append(term)
    
    return term_list


#------------------------------
def get_search_terms_from_input(search_input, diretorio=None):
        
    #definindo o dicionário final da função
    search_input_dic = {}
    
    #--------------------------- termos literais -----------------------
    #caso seja uma procura via regex
    if search_input['regex_mode_for_literal'] is True:
        
        try:
            search_input_dic['literal'] = {}
                        
            #carregando os termos do dicionário temporário
            terms = search_input['literal']
    
            prim_term_list = []
            second_term_list = []
            operation_list = []
            
            #encontrando os termos primários
            try:
                prim_term_list = re.search(r'(?<=s\()(.*?)(?=\)e)', terms).captures()
                
            #caso não haja termo primário
            except AttributeError:
                pass
            
            #encontrando os termos secundários
            sec_terms_find_list = re.findall(r'[\+\-]\s*s\((.*?)(?=\)e)', terms)
            for sec_term_str in sec_terms_find_list:
                second_term_list.append( [ sec_term_str ] )
                
            #encontrando os operadores
            ops_find_list = re.findall(r'(?<=\)e\s*)[\+\-](?=\s*s\()', terms)
            for op in ops_find_list:
                operation_list.append(op)
            
            search_input_dic['literal']['primary'] = prim_term_list
            search_input_dic['literal']['secondary'] = second_term_list
            search_input_dic['literal']['operations'] = operation_list
            search_input_dic['literal']['literal_entry'] = terms
            
        except KeyError:
            search_input_dic['literal']['primary'] = []
            search_input_dic['literal']['secondary'] = []
            search_input_dic['literal']['operations'] = []
            search_input_dic['literal']['literal_entry'] = ''
    
    else:
        try:
            search_input_dic['literal'] = {}
                        
            #carregando os termos do dicionário temporário
            terms = search_input['literal']
    
            prim_term_list = []
            second_term_list = []
            operation_list = []
            
            #encontrando os termos primários
            try:
                prim_term_list_temp = re.findall(r'[#|\w|\w\s/\s\w]+', re.search(r'(?<=[\s*\(])[^\)]+', terms).captures()[0] )
                for prim_term in prim_term_list_temp:
                    prim_term_list.append( re.sub(r'\s*$', '', re.sub(r'^\s*', '', prim_term) ) )
            #caso não haja termo primário
            except AttributeError:
                pass
            
            #encontrando os termos secundários
            sec_terms_find_list = re.findall(r'[+-]\s*\([#\w\s\,\-/]+\)', terms)
            for sec_term_str in sec_terms_find_list:
                second_term_list_inner = []
                #o primeiro char é da operação
                operation_list.append(sec_term_str[0])            
                #coletando os termos secundários
                second_term_list_temp = re.findall(r'(?<=[\s\(]*)[#\w\s]+', re.search(r'(?<=[\s*\(])[^\)]+', sec_term_str).captures()[0] )
                for sec_term in second_term_list_temp:
                    second_term_list_inner.append( re.sub(r'\s*$', '', re.sub(r'^\s*', '', sec_term) ) )
                second_term_list.append(second_term_list_inner)
            
            search_input_dic['literal']['primary'] = prim_term_list
            search_input_dic['literal']['secondary'] = second_term_list
            search_input_dic['literal']['operations'] = operation_list
            search_input_dic['literal']['literal_entry'] = terms
        
        except KeyError:
            search_input_dic['literal']['primary'] = []
            search_input_dic['literal']['secondary'] = []
            search_input_dic['literal']['operations'] = []
            search_input_dic['literal']['literal_entry'] = ''
            

    #--------------------------- termos semânticos -----------------------    
    #encontrando os termos semânticos
    try:
        search_input_dic['semantic'] = {}

        #carregando os termos do dicionário temporário
        terms = search_input['semantic']

        prim_term_list = []
        second_term_list = []
        operation_list = []
        nGrams_list = []

        #encontrando os termos primários
        try:
            prim_semantic_terms = re.findall(r'\b\w+[\w\s]+\w+\b(?=\,)+|\b\w+[\w\s]+\w+\b', re.search(r'(?<=[\s*\(])[\w\s\,]+(?=[\s*\)])', terms).group(0) )
            
            #pegando a lista a partir do termo primário
            prim_term_list = get_nGrams_list(prim_semantic_terms, ngrams = search_input['semantic_ngrams'][0], min_ngram_appearence = search_input['semantic_ngrams_min_app'][0], diretorio = diretorio)

            #colentando se é nGram 0, 1 ou 2
            nGrams_list.append( search_input['semantic_ngrams'][0] )
        
        #caso não haja termo primário
        except AttributeError:
            pass

        if len(prim_term_list) != 0:
            #encontrando os termos secundários
            sec_terms_find_list = re.findall(r'[+-]\s*\([\w\s\,\-]+\)', terms)

            for i in range(len(sec_terms_find_list)):
                #pegar o termo para buscar os semanticamente similares
                sec_semantic_terms = re.findall(r'\b\w+[\w\s]+\w+\b(?=\,)+|\b\w+[\w\s]+\w+\b', re.search(r'(?<=[\s*\(])[\w\s\,]+(?=[\s*\)])', sec_terms_find_list[i]).group(0) )
                
                #o primeiro char é da operação
                operation_list.append(sec_terms_find_list[i][0])
                
                sec_terms = get_nGrams_list(sec_semantic_terms, ngrams = search_input['semantic_ngrams'][1:][i], min_ngram_appearence = search_input['semantic_ngrams_min_app'][1:][i], diretorio = diretorio)
                second_term_list.append( sec_terms )
                nGrams_list.append( search_input['semantic_ngrams'][1:][i] )
            
        search_input_dic['semantic']['primary'] = prim_term_list
        search_input_dic['semantic']['secondary'] = second_term_list
        search_input_dic['semantic']['operations'] = operation_list            
        search_input_dic['semantic']['nGrams'] = nGrams_list
        search_input_dic['semantic']['semantic_entry'] = terms
    
    except (KeyError, IndexError):
        print('ATENÇÃO!\nErro na montagem dos termos para procura semântica.')
        print('Se houver termos secundários a serem procurados, há necessidade de introduzir o "semantic_ngrams" e o "semantic_ngrams_min_app" em uma lista.')
        print('Exemplo: (1,1) e (5,5)') 
        
        search_input_dic['semantic']['primary'] = []
        search_input_dic['semantic']['secondary'] = []
        search_input_dic['semantic']['operations'] = []
        search_input_dic['semantic']['nGrams'] = []
        search_input_dic['semantic']['semantic_entry'] = ''
            
    #--------------------------- padrão regex -----------------------
    #encontrando os padrões regex
    try:
        search_input_dic['regex'] = {}

        #carregando os termos do dicionário temporário
        terms = search_input['regex']
        
        regex_parameter = None
        regex_pattern = None
        if re.search(r'\w+', terms) is not None:
            regex_parameter = re.search(r'\w+', terms).captures()[0]
            #tentando achar o regex pattern do parâmetro introduzido
            regex_pattern = regex_patt_from_parameter(regex_parameter)
        
        if regex_pattern != None:
            regex_entry = regex_parameter
            regex_term = regex_pattern['PU_to_find_regex']
        else:
            regex_entry = ''
            regex_term = ''
            
        search_input_dic['regex']['regex_entry'] = regex_entry
        search_input_dic['regex']['regex_pattern'] = regex_term

    except KeyError:
        search_input_dic['regex']['regex_entry'] = ''
        search_input_dic['regex']['regex_pattern'] = ''
        pass
    
    #print(search_input_dic)
    #time.sleep(10)
    return search_input_dic


#------------------------------
def get_vectors_from_input(search_input_dic, lsa_dim = 10, lda_dim = 10):

    search_vectors_dic = {}
    search_vectors_dic['topic'] = {}
    search_vectors_dic['topic']['any'] = False

    #encontrando os vetores para procura de tópicos
    search_vectors_dic['topic']['lda_sents'] = get_vector_from_string( search_input_dic['lda_sents_topic'], vector_dim = lda_dim, get_versor = True)
    search_vectors_dic['topic']['lda_articles'] = get_vector_from_string(search_input_dic['lda_articles_topic'], vector_dim = lda_dim, get_versor = True)
    search_vectors_dic['topic']['lsa_sents'] = get_vector_from_string(search_input_dic['lsa_sents_topic'], vector_dim = lsa_dim, get_versor = True)
    search_vectors_dic['topic']['lsa_articles'] = get_vector_from_string(search_input_dic['lsa_articles_topic'], vector_dim = lsa_dim, get_versor = True)

    #caso haja algum vetor
    if type(np.array([])) in (type(search_vectors_dic['topic']['lda_sents']), 
                              type(search_vectors_dic['topic']['lda_articles']),
                              type(search_vectors_dic['topic']['lsa_sents']), 
                              type(search_vectors_dic['topic']['lsa_articles'])):
        
        search_vectors_dic['topic']['any'] = True

    return search_vectors_dic


#------------------------------
def get_file_batch_index_list(total_number, batch_size):
    
    #determinando os slices para os batchs
    print('Determinando os slices para os batches...')
    slice_indexes = list(range(0, total_number, batch_size))
    batch_indexes = []
    for i in range(len(slice_indexes)):
        try:
            batch_indexes.append([ slice_indexes[i] , slice_indexes[i + 1] - 1])
        except IndexError:
            pass
    batch_indexes.append([slice_indexes[-1] , total_number - 1])
    
    return batch_indexes


#------------------------------
def get_filenames_from_folder(folder, file_type = 'csv', print_msg = True):
    
    try:
        file_list = os.listdir(folder) #lista de arquivos
    except FileNotFoundError:
        print('Erro!')
        print('O diretório não existe:')
        print(folder)
        return
            
    #testar se há arquivos no diretório
    if len(file_list) == 0:
        print('Erro!')
        print('Não há arquivos no diretório:')
        print(folder)
        return
    
    if file_type == 'webscience_csv_report':
        file_type = 'txt'
        
    documents = []
    for filename in file_list:
        if filename[ -len(file_type) : ].lower() == file_type.lower():
            documents.append(filename[ : - ( len(file_type) + 1) ])

    if print_msg is True            :
        print('Procurando arquivos na pasta: ', folder)
        print('Total de arquivos encontrados: ', len(documents))
    
    return sorted(documents)


#------------------------------
def get_sent_from_index(sent_index, diretorio = None):
    
    send_indexes = pd.read_csv(diretorio + '/Outputs/log/sents_index.csv', index_col = 0)
    for article_filename in send_indexes.index:
        initial_sent = send_indexes.loc[article_filename, 'initial_sent']
        last_sent = send_indexes.loc[article_filename, 'last_sent']
        if last_sent >= sent_index >= initial_sent:            
            sent_DF = pd.read_csv(diretorio + f'/Outputs/sents_filtered/{article_filename}.csv', index_col = 0)
            sent = sent_DF.loc[sent_index].values
        else:
            continue
    
    return sent       


#------------------------------
def get_sent_to_predict(token_list, check_sent_regex_pattern = 'z+x?z*'):

    counter_one_char_tokens = 0
    counter_z_char_tokens = 0
    found_regex = False  
        
    #removendo os token listados
    get_sent = True
    for token in token_list:
        temp_token = str(token)
        #primeiro filtro
        #------------------------------
        if re.search(check_sent_regex_pattern, temp_token):
            counter_z_char_tokens += 1
        #segundo filtro
        #------------------------------        
        if len(temp_token) == 1:
            found_regex = True
            counter_one_char_tokens += 1
    
    cond1 = ( counter_z_char_tokens > 2 )
    cond2 = ( counter_one_char_tokens >= 3 )
    cond3 = ( found_regex is False)
    
    if True in (cond1, cond2, cond3):
        get_sent = False
        #print('\n(Filter) Excluindo: ', token_list)
    
    return get_sent


#------------------------------
def get_tag_name(file_N, prefix = 'ATC'):
    if file_N < 10:
        tag = prefix + '0000'
    elif 10 <= file_N < 100:
        tag = prefix + '000'
    elif 100 <= file_N < 1000:
        tag = prefix + '00'
    elif 1000 <= file_N < 10000:
        tag = prefix + '0'
    elif 10000 <= file_N < 100000:
        tag = prefix
    return tag + str(file_N)


#------------------------------
def load_h5_matrix(filepath, mode = 'r'):

    print('Carregando o arquivo h5 em: ', filepath)
    h5_file = h5py.File(filepath, mode)
    h5_matrix = h5_file['data']
    print('H5 matrix loaded - shape:', h5_matrix.shape)
    
    return h5_file, h5_matrix
    

#------------------------------
def load_log_info(log_name = None, logpath = None):

    if os.path.exists(logpath):
        try:
            dic = load_dic_from_json(logpath)
            return dic[log_name]
        except KeyError:
            None
    else:
        return None


#------------------------------
def load_dic_from_json(file_path):
    
    with open(file_path, 'r') as file:
        dic = json.load(file)
        file.close()
        
    return dic


#------------------------------
def merge_DFs(DF_filename1, DF_filename2, concatDF_filename, diretorio = None):
    
    DF1 = diretorio + f'/Outputs/{DF_filename1}.csv'
    DF2 = diretorio + f'/Outputs/{DF_filename2}.csv'
    
    DF1 = pd.read_csv(DF1, index_col=[0,1])
    DF2 = pd.read_csv(DF2, index_col=[0,1])
    
    DF = pd.merge(DF1, DF2, on=['Filename', 'Index'], how='outer')
    DF.sort_values(by=['Filename', 'Index'], inplace=True)
    DF.to_csv(diretorio + f'/Outputs/{concatDF_filename}.csv')


#------------------------------
def run_func_in_parallel(func, args, workers = 2):

    with Pool(workers) as pool:
        pool.map(func, args)
        pool.close()
        pool.join()


#------------------------------
def save_dic_to_json(path, dic):
    
    with open(path, 'w') as file:
        json.dump(dic, file, indent = 3)
        file.close()


#------------------------------
def saving_acc_to_CSV(last_article_file = 'ATC00000', settings = 'w2vec', acc = 0, folder = '/', diretorio = None):
    
    if not os.path.exists(diretorio + folder + 'wv_accuracy.csv'):
        DF = pd.DataFrame(columns=['Filename', 'Settings', 'Accuracy'])
        DF.set_index(['Filename', 'Settings'], inplace=True)
    else:
        DF = pd.read_csv(diretorio + folder + 'wv_accuracy.csv', index_col = [0,1])
    
    DF.loc[(last_article_file, settings), 'Accuracy'] = acc
    DF.sort_values(by=['Filename', 'Settings'], inplace=True)
    DF.to_csv(diretorio + folder + 'wv_accuracy.csv')


#------------------------------
def update_log(log_names = None, entries = None, logpath = None):

    #confirmando se o número de entradas está batendo
    if len(log_names) == len(entries):

        if not os.path.exists(logpath):
            log_dic = {}            
            for i in range(len(log_names)):
                log_dic[log_names[i]] = entries[i]
        
        else:
            log_dic = load_dic_from_json(logpath)
            for i in range(len(log_names)):
                log_dic[log_names[i]] = entries[i]
        
        save_dic_to_json(logpath, log_dic)
    
    else:
        print('Erro nos inputs da função "update_log"!')
        print('Inserir o mesmo número de elementos nas listas dos args: "log_names" e "entries".')



'''
*** Essas funções abaixo não estão sendo usadas *** 
#------------------------------
#combinações de termo
def find_term_combinations(term):

    terms = []    
    #caso o termo seja composto por várias palavras
    if len(term.split()) > 1:
        concat_term = ''
        #contanando os termos
        for token in term.split():
            concat_term += token
        terms.append(concat_term)            
        for char_index in range(1, len(concat_term)):
            s1_term = concat_term[ : char_index ]  + '-' + concat_term[ char_index : ]
            terms.append(s1_term)
        terms.append(concat_term + '-')

    #caso o termo seja composto só por um token
    else:
        terms.append(term)
        for char_index in range(1, len(term)):
            s1_term = term[ : char_index ]  + '-' + term[ char_index : ]
            terms.append(s1_term)
        terms.append(term + '-')
    
    #print('\n', terms,'\n')    
    return terms


#------------------------------
#combinações das listas de termos
def find_terms_combinations(term_list):
    
    terms = []
    for term_N in range(len(term_list)):
        terms.append(term_list[term_N])
        for char_index in range(1, len(term_list[term_N])):
            s1_term = term_list[term_N][ : char_index ]  + '-' + term_list[term_N][ char_index : ]    
            terms.append(s1_term)
        terms.append(term_list[term_N] + '-')    
    #print('\n',terms,'\n')
    
    return terms
'''
