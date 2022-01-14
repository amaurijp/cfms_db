#!/usr/bin/env python3
# -*- coding: utf-8 -*-    

#------------------------------
def check_memory_use(memory_used):
    import psutil
    #registrando o uso atual de memória        
    current_memory_used = {}
    current_memory_used['ram'] = psutil.virtual_memory()[3]
    current_memory_used['buffers'] = psutil.virtual_memory()[7]
    current_memory_used['cached'] = psutil.virtual_memory()[8]

    print('Memory usage - used: ', round(current_memory_used['ram']/1000000000, 2),'(GB); variation: ', round( ( current_memory_used['ram'] / memory_used['ram'] ), 2),' %')
    print('Memory usage - buffers: ', round(current_memory_used['buffers']/1000000000, 2),'(GB); variation: ', round( ( current_memory_used['buffers'] / memory_used['buffers'] ), 2), ' %')
    print('Memory usage - cached: ', round(current_memory_used['cached']/1000000000, 2),'(GB); variation: ', round( ( current_memory_used['cached'] / memory_used['cached'] ), 2), ' %')


#------------------------------
def check_regex_subs(pattern, substitute_text, text, char_range = 30):
    import regex as re
    res = re.finditer(pattern, text)
    for match in res:        
        before_text = text[ match.start() - int(char_range/2) : match.end() + char_range ]
        print('Before: ', repr(before_text))
        after_text = re.sub(pattern, substitute_text, before_text)
        print('After : ', repr(after_text))


#------------------------------
def create_DF_columns_names_with_index(features, index_columns = ['Filename', 'Counter'], index_name = '_sent_index'):
    
    l = [ [ parameter, parameter + index_name ] for parameter in features ]                            
    columns = index_columns + [item for sublist in l for item in sublist] 
    
    return columns
    

#------------------------------
def create_PDFFILE_list(pdf_filename):
    import regex as re
    if (pdf_filename == 'PDF00000'):
        return []
    else:
        match = re.search('[1-9]{1,5}[0-9]*', pdf_filename)
        file_number = int(match.captures()[0])
        return [ get_tag_name(file_N) for file_N in list(range(file_number + 1)) ]


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
    
    import numpy as np
    import pandas as pd
    import regex as re

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
    
            #search mode
            dic[line]['search_mode'] = inputs_DF.loc[ line , 'search_mode' ]
    
            #parameter to extract
            dic[line]['parameter_to_extract'] = inputs_DF.loc[ line , 'parameter_to_extract' ]
            
            #lower sentence para extrair
            if str(inputs_DF.loc[ line , 'lower_sentence_to_extract_categories' ]).lower() == 'false':
                dic[line]['lower_sentence_to_extract_categories'] = False
            else:
                dic[line]['lower_sentence_to_extract_categories'] = True

            #ngrams_to_extract
            if int(inputs_DF.loc[ line , 'ngrams_to_extract' ]) in (0, 1, 2):
                dic[line]['ngrams_to_extract'] = inputs_DF.loc[ line , 'ngrams_to_extract' ]
            else:
                dic[line]['ngrams_to_extract'] = 0
            
            #ngrams_min_app
            if int(inputs_DF.loc[ line , 'ngrams_min_app' ]) in range(100):
                dic[line]['ngrams_min_app'] = int(inputs_DF.loc[ line , 'ngrams_min_app' ])
            else:
                dic[line]['ngrams_min_app'] = 1
        
            #dicionário com os inputs de search
            dic[line]['search_inputs'] = {}
            
            #literal
            if inputs_DF.loc[ line , 'literal_entry' ].lower() == 'none':
                dic[line]['search_inputs']['literal'] = '()'
            else:
                dic[line]['search_inputs']['literal'] = inputs_DF.loc[ line , 'literal_entry' ]
            
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
            ngrams_list = re.findall(r'[0-9]+', str(inputs_DF.loc[ line , 'semantic_ngrams' ]))
            for number in ngrams_list:
                if int(number) in (0, 1, 2):
                    dic[line]['search_inputs']['semantic_ngrams'].append( int(number) )
                else:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('O "semantic_ngrams" suporta três valores (0, 1 e 2) na forma 0-1-1')
                    print('Valor de entrada: ', inputs_DF.loc[ line , 'semantic_ngrams' ])
            
            dic[line]['search_inputs']['semantic_ngrams_min_app'] = []
            min_app_list = re.findall(r'[0-9]+', str(inputs_DF.loc[ line , 'semantic_ngrams_min_app' ]))
            for number in min_app_list:
                try:
                    dic[line]['search_inputs']['semantic_ngrams_min_app'].append( int(number) )
                except:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('O "semantic_ngrams_min_app" suporta somente com valores númericos do tipo 6-50-30.')
                    print('Valor de entrada: ', inputs_DF.loc[ line , 'semantic_ngrams_min_app' ])        
            
            #topic
            if inputs_DF.loc[ line , 'topic_entry' ].lower() == 'none':
                dic[line]['search_inputs']['topic'] = '()'
            else:
                dic[line]['search_inputs']['topic'] = inputs_DF.loc[ line , 'topic_entry' ]
                
            if inputs_DF.loc[ line , 'topic_match_mode' ].lower() in ('largest', 'threshold', 'none'):
                dic[line]['search_inputs']['topic_match_mode'] = inputs_DF.loc[ line , 'topic_match_mode' ]
            else:
                print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                print('O "topic_match_mode" suporta dois valores ("largest", "threshold", "None")')
                print('Valor de entrada: ', inputs_DF.loc[ line , 'topic_match_mode' ])
            
            if inputs_DF.loc[ line , 'topic_entry' ].lower() == 'none':
                dic[line]['search_inputs']['topic_thr_corr_val'] = None
            else:
                try:
                    dic[line]['search_inputs']['topic_thr_corr_val'] = round(float(inputs_DF.loc[ line , 'topic_thr_corr_val' ]), 2)
                except:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('O valor de "topic_thr_corr_val" precisa ser menor que 1.0')
                    print('Valor de entrada: ', inputs_DF.loc[ line , 'topic_thr_corr_val' ])
            
            #regex
            if inputs_DF.loc[ line , 'regex_entry' ].lower() == 'none':
                dic[line]['search_inputs']['regex'] = ''
            else:
                dic[line]['search_inputs']['regex'] = inputs_DF.loc[ line , 'regex_entry' ]
            
            #bayesian
            if inputs_DF.loc[ line , 'bayesian_entry' ].lower() == 'none':
                dic[line]['search_inputs']['bayesian'] = ''
            else:    
                dic[line]['search_inputs']['bayesian'] = inputs_DF.loc[ line , 'bayesian_entry' ]
    
            #filter section
            if inputs_DF.loc[ line , 'filter_section' ].lower() == 'none':
                dic[line]['search_inputs']['filter_section'] = None
            else:    
                if inputs_DF.loc[ line , 'filter_section' ].lower() in ('cnn', 'logreg', 'randomforest', 'svm'):
                    dic[line]['search_inputs']['filter_section'] = inputs_DF.loc[ line , 'filter_section' ]
                else:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('A entrada de "filter_section" não é compatível.')
                    print('Entradas compatíveis: "cnn", "randomforest", "logreg", "svm" e "None"')
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
            
            #lower sentence para extrair
            if str(inputs_DF.loc[ line , 'lower_sentence_to_extract_categories' ]).lower() == 'false':
                dic[line]['lower_sentence_to_extract_categories'] = False
            else:
                dic[line]['lower_sentence_to_extract_categories'] = True
                    
            #hold_samples
            if str(inputs_DF.loc[ line , 'hold_samples' ]).lower() == 'true':
                dic[line]['hold_samples'] = True
            elif str(inputs_DF.loc[ line , 'hold_samples' ]).lower() == 'false':
                dic[line]['hold_samples'] = False
            else:
                dic[line]['hold_samples'] = False
    
            #hold_sample_number
            if str(inputs_DF.loc[ line , 'hold_sample_number' ]).lower() == 'true':
                dic[line]['hold_sample_number'] = True
            elif str(inputs_DF.loc[ line , 'hold_sample_number' ]).lower() == 'false':
                dic[line]['hold_sample_number'] = False
            else:
                dic[line]['hold_sample_number'] = False

            #numbers_extraction_mode
            if str(inputs_DF.loc[ line , 'numbers_extraction_mode' ]).lower() in ('all', 'first'):
                dic[line]['numbers_extraction_mode'] = inputs_DF.loc[ line , 'numbers_extraction_mode' ].lower()
            else:
                dic[line]['numbers_extraction_mode'] = 'all'

            #get_avg_num_results
            if str(inputs_DF.loc[ line , 'get_avg_num_results' ]).lower() == 'true':
                dic[line]['get_avg_num_results'] = True
            elif str(inputs_DF.loc[ line , 'get_avg_num_results' ]).lower() == 'false':
                dic[line]['get_avg_num_results'] = False
            else:
                dic[line]['get_avg_num_results'] = False

            #filter_unique_results
            if str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'true':
                dic[line]['filter_unique_results'] = True
            elif str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'false':
                dic[line]['filter_unique_results'] = False
            else:
                dic[line]['filter_unique_results'] = False

            #match_all_sent_findings
            if str(inputs_DF.loc[ line , 'match_all_sent_findings' ]).lower() == 'true':
                dic[line]['match_all_sent_findings'] = True
            elif str(inputs_DF.loc[ line , 'match_all_sent_findings' ]).lower() == 'false':
                dic[line]['match_all_sent_findings'] = False
            else:
                dic[line]['match_all_sent_findings'] = False
                
            #ngram_for_textual_search
            if int(inputs_DF.loc[ line , 'ngram_for_textual_search' ]) in (0, 1, 2):
                dic[line]['ngram_for_textual_search'] = int(inputs_DF.loc[ line , 'ngram_for_textual_search' ])
            else:
                dic[line]['ngram_for_textual_search'] = 0

            #min_ngram_appearence
            if int(inputs_DF.loc[ line , 'min_ngram_appearence' ]) in range(100):
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
        
    print(prediction_list)
    print(target_val_list)
    
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
            
    import os
    
    try:
        file_list = os.listdir(folder) #lista de arquivos
    except FileNotFoundError:
        print('Erro!')
        print('O diretório não existe:')
        print(folder)
        return
            
    #testar se há arquivos no diretório DB
    if len(file_list) == 0:
        print('Erro!')
        print('Não há arquivos no diretório:')
        print(folder)
        return
    
    documents = []
    for filename in file_list:
        if filename[ -len(file_type) : ].lower() == file_type:
            documents.append(filename[ : - ( len(file_type) + 1) ])

    if print_msg is True            :
        print('Procurando arquivos na pasta: ', folder)
        print('Total de arquivos encontrados: ', len(documents))
    
    return sorted(documents)


#------------------------------
def get_sent_from_index(sent_index, diretorio = None):
    
    import pandas as pd
    
    send_indexes = pd.read_csv(diretorio + '/Outputs/LOG/SENT_INDEX.csv', index_col = 0)
    for pdf_filename in send_indexes.index:
        initial_sent = send_indexes.loc[pdf_filename, 'initial_sent']
        last_sent = send_indexes.loc[pdf_filename, 'last_sent']
        if last_sent >= sent_index >= initial_sent:            
            sent_DF = pd.read_csv(diretorio + f'/Outputs/Sentences_filtered/{pdf_filename}.csv', index_col = 0)
            sent = sent_DF.loc[sent_index].values
        else:
            continue
    
    return sent       


#------------------------------
def get_sent_to_predict(token_list, check_sent_regex_pattern = 'z+x?z*'):
    
    import regex as re
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
        #print('(Filter) Excluindo: ', token_list)
    
    return get_sent


#------------------------------
def get_tag_name(file_N, prefix = 'PDF'):
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
def load_index_from_TXT(file_name, basefolder, diretorio = None):

    with open(diretorio + basefolder + '/' + file_name + '.txt', 'r') as file:
        index = file.readline()
        if index[-1] == '\n':
            index = index[ : -1]
        try:
            index = int(index)
        except ValueError:
            pass
        file.close()
    
    return index


#------------------------------
def load_dic_from_json(file_path):
    
    import json
    
    with open(file_path, 'r') as file:
        dic = json.load(file)
        file.close()
        
    return dic


#------------------------------
def merge_DFs(DF_filename1, DF_filename2, concatDF_filename, diretorio = None):
    
    import pandas as pd
    
    DF1 = diretorio + f'/Outputs/{DF_filename1}.csv'
    DF2 = diretorio + f'/Outputs/{DF_filename2}.csv'
    
    DF1 = pd.read_csv(DF1, index_col=[0,1])
    DF2 = pd.read_csv(DF2, index_col=[0,1])
    
    DF = pd.merge(DF1, DF2, on=['Filename', 'Index'], how='outer')
    DF.sort_values(by=['Filename', 'Index'], inplace=True)
    DF.to_csv(diretorio + f'/Outputs/{concatDF_filename}.csv')


#------------------------------
def process_search_input_file(search_input, ngrams = [0], min_ngram_appearence = [5], diretorio=None):
    
    #import time
    import regex as re
    from functions_PARAMETERS import regex_patt_from_parameter
    from functions_TOKENS import get_nGrams_list
    
    #caso o search_input já esteja no arquivo txt em ~/TXT_Setiings
    if type(search_input) == str:
        #abrindo o input file
        with open(diretorio + f'/Settings/{search_input}','r') as search_terms_file:
            print('Abrindo Settings: ', f'/Settings/{search_input}')
            terms_list = generate_term_list(search_terms_file)
            search_terms_file.close()
        #print(terms_list)
    
        #criando um dicionário temporário para armazenar os termos
        temp_term_list_dic = {}
        
        #separando os terms list por literal, semântico, bayesiano, etc
        entry_counter = 1
        for item in terms_list:
            try:
                item_search_mode = re.search(r'[a-z]+(?=\s*\=)', item).captures()[0]
                item_terms = re.search(r'(?<=\s*\=)[^.]+', item).captures()[0]
                #caso as entradas sejam todas diferentes (ex: literal, semantic, topic)
                if item_search_mode not in temp_term_list_dic:
                    temp_term_list_dic[ item_search_mode ] = item_terms
                    #caso seja a primeira linha de entrada
                    if entry_counter == 1:
                        first_item_search_mode = item_search_mode
                        first_item_terms = item_terms
                    
                #caso as entradas sejam todas iguais (ex: só literal)
                else:
                    #caso seja a primeira linha de entrada
                    if entry_counter == 1:
                        #adicionando a primeira entrada no dic com indicador numérico
                        temp_term_list_dic[ first_item_search_mode + str(entry_counter) ] = first_item_terms
                        entry_counter += 1
                    
                    temp_term_list_dic[ item_search_mode + str(entry_counter) ] = item_terms
                    entry_counter += 1
                    
            except AttributeError:
                continue
    
    #caso o search_input já esteja no formato de dicionário e seja introduzido diretamente
    elif type(search_input) == dict:
        temp_term_list_dic = search_input
        
    #print(temp_term_list_dic)
    #definindo o dicionário final da função
    term_list_dic = {}
    

    #--------------------------- termos literais -----------------------
    #caso haja multiplas linhas para a procura com termos no modo literal
    multi_line_entry_list = [key for key in temp_term_list_dic.keys() if (key[-1] in '0123456789') and ( (key[ : len('literal') ] == 'literal') or (key[ : len('topic') ] == 'topic') )]
    if len(multi_line_entry_list) > 0:
        literal_topic_entry_list = multi_line_entry_list
    #caso haja somente uma linha para procura com termos no modo literal
    else:
        literal_topic_entry_list = ['literal']
        
    for search_mode in literal_topic_entry_list:
        try:
            term_list_dic[search_mode] = {}
                        
            #carregando os termos do dicionário temporário
            terms = temp_term_list_dic[search_mode]

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
            
            term_list_dic[search_mode]['primary'] = prim_term_list
            term_list_dic[search_mode]['secondary'] = second_term_list
            term_list_dic[search_mode]['operations'] = operation_list
            term_list_dic[search_mode][f'{search_mode}_entry'] = terms
        
        except KeyError:
            term_list_dic[search_mode]['primary'] = []
            term_list_dic[search_mode]['secondary'] = []
            term_list_dic[search_mode]['operations'] = []
            term_list_dic[search_mode][f'{search_mode}_entry'] = ''
            break

    #--------------------------- termos semânticos -----------------------    
    #encontrando os termos semânticos
    try:
        term_list_dic['semantic'] = {}
    
        #carregando os termos do dicionário temporário
        terms = temp_term_list_dic['semantic']
    
        prim_term_list = []
        prim_term_prob_list = []
        second_term_list = []
        second_term_prob_list = []
        operation_list = []

        #encontrando os termos primários            
        try:
            prim_terms_find_list = re.findall(r'[A-za-z0-9]+', re.search(r'(?<=[\s*\(])[^\)]+', terms).captures()[0] )
            #no modo de procura semântico, só pode ter um termo por operação (dentro dos parêntesis)
            if len(prim_terms_find_list) > 1:
                print('Erro! No modo de procura semântico, só pode haver um termo primário.')
                print('Termos Primários introduzidos: ', prim_terms_find_list)
                
            else:
                prim_semantic_term = prim_terms_find_list[0]
            
                #pegando a lista a partir do termo primário
                prim_term_list, prim_term_prob_list = get_nGrams_list(prim_semantic_term, ngrams = ngrams[0], min_ngram_appearence = min_ngram_appearence[0], diretorio = diretorio)
        
        #caso não haja termo primário
        except AttributeError:
            pass

        if len(prim_term_list) != 0:
            #encontrando os termos secundários
            sec_terms_find_list = re.findall(r'[+-]\s*\([\w\s\,\-]+\)', terms)
            for i in range(len(sec_terms_find_list)):
                #pegar o termo para buscar os semanticamente similares
                sec_semantic_term = re.findall(r'\w+', sec_terms_find_list[i])
                #no modo de procura semântico, só pode ter um termo por operação (dentro dos parêntesis)
                if len(sec_semantic_term) > 1:
                    print('Erro! No modo de procura semântico, só pode haver um termo por operação.')
                    print('Termos Secundários introduzidos: ', sec_semantic_term)
                    break
                    
                else:
                    #o primeiro char é da operação
                    operation_list.append(sec_terms_find_list[i][0])
                    sec_semantic_term = sec_semantic_term[0]            
                    
                    sec_terms, sec_prob_terms = get_nGrams_list(sec_semantic_term, ngrams = ngrams[1:][i], min_ngram_appearence = min_ngram_appearence[1:][i], diretorio = diretorio)
                    second_term_list.append( sec_terms )
                    second_term_prob_list.append( sec_prob_terms )    
            
        term_list_dic['semantic']['primary'] = prim_term_list
        term_list_dic['semantic']['secondary'] = second_term_list
        term_list_dic['semantic']['operations'] = operation_list            
        term_list_dic['semantic']['primary_prob'] = prim_term_prob_list
        term_list_dic['semantic']['secundary_prob'] = second_term_prob_list
        term_list_dic['semantic']['semantic_entry'] = terms
    
    except (KeyError, IndexError):        
        term_list_dic['semantic']['primary'] = []
        term_list_dic['semantic']['secondary'] = []
        term_list_dic['semantic']['operations'] = []        
        term_list_dic['semantic']['primary_prob'] = []
        term_list_dic['semantic']['secundary_prob'] = []
        term_list_dic['semantic']['semantic_entry'] = ''
        pass
    

    #--------------------------- tópicos -----------------------
    #encontrando os termos semânticos
    try:
        term_list_dic['topic'] = {}
    
        #carregando os termos do dicionário temporário
        terms = temp_term_list_dic['topic']

        prim_term_list = []
        second_term_list = []
        operation_list = []
        
        #encontrando os termos primários entre parêntesis
        try:
            prim_term_list = re.findall(r'\w+', re.search(r'(?<=[\s*\(])[^\)]+', terms).captures()[0] )
            #no modo de procura de tópico, só pode ter um termo por operação (dentro dos parêntesis)
            if len(prim_term_list) > 1:
                print('Erro! No modo de procura por tópico, só pode haver um termo primário.')
                print('Termos Primários introduzidos: ', prim_term_list)
                prim_term_list = []
                
        #caso não haja termo primário
        except AttributeError:
            pass

        if len(prim_term_list) != 0:
            #encontrando os termos secundários entre parêntesis
            sec_terms_find_list = re.findall(r'[+-]\s*\([\w\s\,\-]+\)', terms)
            for sec_term_str in sec_terms_find_list:
                #coletando os termos secundários
                second_term_list_inner = re.findall(r'\w+', sec_term_str)            
                #no modo de procura por tópico, só pode ter um termo por operação (dentro dos parêntesis)
                if len(second_term_list_inner) > 1:
                    print('Erro! No modo de procura por tópico, só pode haver um termo por operação.')
                    print('Termos Secundários introduzidos: ', second_term_list_inner)
                    second_term_list = []
                    break
                else:
                    #o primeiro char é da operação
                    operation_list.append(sec_term_str[0])
                    second_term_list.append( second_term_list_inner )
                    
        term_list_dic['topic']['primary'] = prim_term_list
        term_list_dic['topic']['secondary'] = second_term_list
        term_list_dic['topic']['operations'] = operation_list
        term_list_dic['topic']['topic_entry'] = terms
        
    except KeyError:
        term_list_dic['topic']['primary'] = []
        term_list_dic['topic']['secondary'] = []
        term_list_dic['topic']['operations'] = []
        term_list_dic['topic']['topic_entry'] = ''
        pass
            
    #--------------------------- padrão regex -----------------------
    #encontrando os padrões regex
    try:
        term_list_dic['regex'] = {}
    
        #carregando os termos do dicionário temporário
        terms = temp_term_list_dic['regex']
        
        try:
            regex_parameter = re.search(r'\w+', terms).captures()[0]
            #tentando achar o regex pattern do parâmetro introduzido
            regex_pattern = regex_patt_from_parameter(regex_parameter)
            if regex_pattern != None:
                regex_entry = regex_parameter
                regex_term = regex_pattern['PU_to_find_regex']
            else:
                regex_entry = regex_parameter
                regex_term = regex_parameter
        
        #quando não existe nenhum padrão regex
        except AttributeError:        
            regex_entry = ''
            regex_term = ''
            
        term_list_dic['regex']['regex_entry'] = regex_entry
        term_list_dic['regex']['regex_pattern'] = regex_term

    except KeyError:
        term_list_dic['regex']['regex_entry'] = ''
        term_list_dic['regex']['regex_pattern'] = ''
        pass

    #--------------------------- padrão bayesiano -----------------------
    try:
        bayes_term = re.search( r'\w+', temp_term_list_dic['bayesian'] ).captures()[0]
        term_list_dic['bayesian'] = bayes_term
    except (AttributeError, KeyError):
        term_list_dic['bayesian'] = ''
        pass
    
    #--------------------------------------------------------------------
    
    #print(term_list_dic)
    #time.sleep(10)
    
    return term_list_dic


#------------------------------
def save_dic_to_json(file_path, dic):
    
    import json
    
    with open(file_path, 'w') as file:
        json.dump(dic, file)
        file.close()


#------------------------------
def save_index_to_TXT(index, file_name, basefolder, diretorio=None):

    with open(diretorio + basefolder + '/' + file_name + '.txt', 'w') as file:
        file.write(str(index))
        file.close()
    del file


#------------------------------
def save_sentences_to_csv(sentences_generator, filename, documents_counter, folder = 'Sentences', diretorio=None):
    
    import os
    import pandas as pd
    
    DF = pd.DataFrame([],columns=['PDF Number', 'Sentence'], dtype=object)
    counter = documents_counter
    for sent in sentences_generator:
        DF.loc[counter] = filename, sent
        counter += 1
    #caso não haja o diretorio ~/Outputs/folder
    if not os.path.exists(diretorio + f'/Outputs/{folder}'):
        os.makedirs(diretorio + f'/Outputs/{folder}')
    DF.to_csv(diretorio + f'/Outputs/{folder}/' + filename + '.csv')
    print(f'Salvando as sentenças extraidas em ~/Outputs/{folder}/' + filename + '.csv')
    
    return counter


#------------------------------
def saving_acc_to_CSV(last_PDF_file = 'PDF00000', settings = 'w2vec', acc = 0, folder = '/', diretorio = None):
    
    import os
    import pandas as pd
    
    if not os.path.exists(diretorio + folder + 'WV_Accuracy.csv'):
        DF = pd.DataFrame(columns=['Filename', 'Settings', 'Accuracy'])
        DF.set_index(['Filename', 'Settings'], inplace=True)
    else:
        DF = pd.read_csv(diretorio + folder + 'WV_Accuracy.csv', index_col = [0,1])
    
    DF.loc[(last_PDF_file, settings), 'Accuracy'] = acc
    DF.sort_values(by=['Filename', 'Settings'], inplace=True)
    DF.to_csv(diretorio + folder + 'WV_Accuracy.csv')



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
