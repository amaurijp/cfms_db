#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class text_objects(object):
    
    def __init__(self, text_break_function = 'function_text', tokenizer = 'spacy', replace_numbers = False, replace_Ngrams = False, diretorio = None):
        
        import os
        
        print('\n=============================\nTEXTOBS...\n=============================\n')
        print('( Class: text_objects )')
        
        from FUNCTIONS import get_filenames_from_folder
        self.diretorio = diretorio
                
        self.replace_Ngrams = replace_Ngrams
        self.replace_numbers = replace_numbers
        self.tokenizer = tokenizer
        self.text_break_function = text_break_function
        
        #montando a lista de pdfs da coleção
        self.DB_documents = get_filenames_from_folder(self.diretorio + '/DB', file_type = 'pdf')
        
        #definindo o tokenizer
        if self.tokenizer == 'spacy':
            import spacy
            self.tokenizer_func = spacy.load('en_core_web_sm')
        else:
            self.tokenizer_func = None

        #checando a existência das pastas
        if not os.path.exists(self.diretorio + '/Outputs/Sentences'):
            os.makedirs(self.diretorio + '/Outputs/Sentences')
        
        if not os.path.exists(self.diretorio + '/Outputs/Sentences_filtered'):
            os.makedirs(self.diretorio + '/Outputs/Sentences_filtered')
        
        if not os.path.exists(self.diretorio + '/Outputs/LOG'):
            os.makedirs(self.diretorio + '/Outputs/LOG')
    
    
    def get_text_objects(self, save_TXT = False, save_JSON = False, filter_by_TF = False, min_tokens_to_break_sent = 2, min_tokens_per_sent = 2):

        print('\n( Function: get_text_objects )')
        
        #import time
        from FUNCTIONS import save_sentences_to_csv
        from functions_TEXTS import save_text_to_TXT
        from functions_TEXTS import save_full_text_to_json
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import save_index_to_TXT
        from PDF import PDFTEXT
        import pandas as pd
        import os

        #abrindo o DF TF YEAR para filtrar os documentos que serão processados      
        if os.path.exists(self.diretorio + '/Outputs/DB_TF_YEAR.csv'):
            DF_TF_Year = pd.read_csv(self.diretorio + '/Outputs/DB_TF_YEAR.csv', index_col = 0)
            try:
                DF_TF_Year.index.values[-1]
            except IndexError:
                print('ERRO! DF TF_Year não possui entradas.')
                print('> Abortando função: text_objects.get_text_objects')
                return            
        else:
            print('ERRO! DF TF_Year não encontrado.')
            print('> Abortando função: text_objects.get_text_objects')
            return

        #abrindo o json com o DB_TF_stats para filtrar os documentos que serão processados      
        if os.path.exists(self.diretorio + '/Outputs/DB_TF_stats.json'):
            DB_TF_stats = load_dic_from_json(self.diretorio + '/Outputs/DB_TF_stats.json')
            #o threshold do TF para os arquivos está definido com a média menos um desvio padrão
            TF_threshold = DB_TF_stats['1_quartile']
        else:
            print('ERRO! Dicionário TF_stats não encontrado.')
            print('> Abortando função: text_objects.get_text_objects')
            return
        
        #resumindo o processamento
        (current_file_name, 
        current_file_index, 
        last_file_index, 
        sentence_counter, 
        sentence_counter_numbers_replaced) = self.resume_text_processing(input_file_list = self.DB_documents, 
                                                                         output_folder = 'Sentences',
                                                                         log_TXT_file_name = 'counter_File_index_get_text_objects')
                
        #caso o último documento processado seja igual ao último arquivo da pasta /Sentences
        if self.DB_documents[-1] == last_file_index:
            print(f'Todos os documentos ({len(self.DB_documents)}) já foram processados')
            print('> Abortando função: text_objects.get_text_objects')
            return
        
        #processando os documentos
        for filename in self.DB_documents[ current_file_index : ]:
            
            #só será processado o documento cujo TF estiver acima do THRESHOLD
            if filter_by_TF is True:
                if DF_TF_Year.loc[filename , 'TF'] >= TF_threshold:
                    pass
                else:
                    print('\nTF Erro para ', filename)
                    print('TF abaixo do threshold: ', DF_TF_Year.loc[filename , 'TF'], '( THR: , ', TF_threshold, ' )' )
                    print('Pulando para o próximo documento...')
                    continue
            
            try:                    
                PDF = PDFTEXT(filename, 
                              folder_name = 'DB', 
                              language = 'english',
                              text_break_function = self.text_break_function,
                              replace_numbers = self.replace_numbers,
                              diretorio = self.diretorio)
                
                PDF.process_text(apply_filters = True, min_tokens_per_sent = min_tokens_per_sent, min_tokens_to_break_sent = min_tokens_to_break_sent)
                
                #salvando os textos em formato JSON
                if (save_JSON is True):
                    #salvando o texto raw
                    #save_full_text_to_json(PDF.raw_text, f'PDF{file_N_conv}', 'Texts_raw', raw_string = False)
                    #salvando o texto filtrado
                    save_full_text_to_json(PDF.filtered_text, filename, 'Texts', raw_string = False, diretorio = self.diretorio)
                    if (self.replace_numbers is True):
                        #salvando o texto filtrado e com números substituidos
                        save_full_text_to_json(PDF.filtered_text_replaced, filename, 'Texts_filtered', raw_string = False, diretorio = self.diretorio)
                                                        
                #salvando as sentenças
                sentence_counter = save_sentences_to_csv(PDF.sentences, 
                                                         filename, 
                                                          sentence_counter, 
                                                          folder = 'Sentences', 
                                                          diretorio = self.diretorio)
                print('Document (sentence) counter = ', sentence_counter)
                if (self.replace_numbers is True):
                    sentence_counter_numbers_replaced = save_sentences_to_csv(PDF.sentences_replaced, 
                                                                              filename, 
                                                                              sentence_counter_numbers_replaced, 
                                                                              folder = 'Sentences_numbers_replaced', 
                                                                              diretorio = self.diretorio)
                    print('Document (sentence) counter (number replaced) = ', sentence_counter_numbers_replaced)
                #salvando os textos em TXT (formato legível)
                if save_TXT is True:
                    #salvando o texto extraido em TXT
                    save_text_to_TXT(repr(PDF.raw_text), filename)
                    
                #salvando o file_index
                save_index_to_TXT(filename, 'counter_File_index_get_text_objects', '/Outputs/LOG', diretorio = self.diretorio)
                    
        
            except ValueError:
                with open(self.diretorio + '/Outputs/Doc_ProcErrors.txt', 'a') as file:
                    file.write(filename + '\n')
                    file.close()
                print('Erro! Não foi possível processar o documento: ' + filename)
                print('Registrando o erro em: ~/Outputs/Doc_ProcErrors.txt')
                continue
        

                
    def filter_sentences(self, cut_quantile_low = 0.1, cut_quantile_high = 0.1, min_sent_tokens = 5, check_function = False):
        
        print('\n( Function: filter_sentences )')

        import time
        import numpy as np        
        import pandas as pd
        import regex as re
        from FUNCTIONS import save_index_to_TXT
        from FUNCTIONS import save_sentences_to_csv
        from FUNCTIONS import get_filenames_from_folder
        
        extracted_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences', file_type = 'csv')
        
        (current_file_name, 
        current_file_index, 
        last_file_index, 
        sentence_counter, 
        sentence_counter_numbers_replaced) = self.resume_text_processing(input_file_list = extracted_DB_documents, 
                                                                         output_folder = 'Sentences_filtered',
                                                                         log_TXT_file_name = 'counter_File_index_filter_sentences')
                                                                          
        #caso o último documento processado seja igual ao último arquivo da pasta /Sentences
        if extracted_DB_documents[-1] == last_file_index:
            print(f'Todos os documentos ({len(extracted_DB_documents)}) já foram filtrados')
            print('> Abortando função: text_objects.filter_sentences')
            return
        
        #contadores para calcular a eficácia da seleção de seção de referências
        total_file_counter = 0
        ref_found_counter = 0
        counter_references_marker = 0
        counter_conclusions_marker = 0
        
        for filename in extracted_DB_documents[ current_file_index : ]:
            
            print('\nProcessing ', filename, '...')            
            total_file_counter += 1
                        
            #abrinado o DF do documento
            sentDF = pd.read_csv(self.diretorio + f'/Outputs/Sentences/{filename}.csv', index_col = 0)

            #check se encontrou a seção de referências
            found_ref_section = False

            #quantidad minima de referências para determinar que se encontrou a seção            
            min_ref_to_find = 10
            
            #na primeira tentativa procura-se pelo marcador de referências
            (found_ref_section_delimiter1, 
            reference_section_index_list1,
            token_number_list1, 
            sent_n_tokens_dic1) = self.looking_for_ref_section(sentDF, 
                                                              ref_section_index_range = 50, 
                                                              min_ref_to_find_for_section = min_ref_to_find, 
                                                              section_marker = 'references', 
                                                              check_function = check_function)
            
            #caso as referências tenham sido encontradas na primeira tentativa
            if found_ref_section_delimiter1 is True and len(reference_section_index_list1) >= min_ref_to_find:
                counter_references_marker += 1
                found_ref_section = True
                reference_section_index_list = reference_section_index_list1
                token_number_list = token_number_list1
                sent_n_tokens_dic = sent_n_tokens_dic1
                
            #na segunda tentativa procura-se pelo marcador de conclusions            
            else:
                (found_ref_section_delimiter2, 
                reference_section_index_list2,
                token_number_list2, 
                sent_n_tokens_dic2) = self.looking_for_ref_section(sentDF, 
                                                                   ref_section_index_range = 50, 
                                                                   min_ref_to_find_for_section = min_ref_to_find, 
                                                                   section_marker = 'conclusions', 
                                                                   check_function = check_function)
                                                              

                #caso as referências tenham sido encontradas na primeira tentativa
                if found_ref_section_delimiter2 is True and len(reference_section_index_list2) >= min_ref_to_find:
                    counter_conclusions_marker += 1
                    found_ref_section = True
                    reference_section_index_list = reference_section_index_list2
                    token_number_list = token_number_list2
                    sent_n_tokens_dic = sent_n_tokens_dic2
                    
                    
            #documento cujas referências foram encontradas
            if found_ref_section is True:            
                
                #fazendo a estatística dos tokens            
                sent_token_len_array = np.array(token_number_list)            
                sent_token_stats = {}
                sent_token_stats['low_quantile'] = round(np.quantile(sent_token_len_array, cut_quantile_low), 10)
                sent_token_stats['high_quantile'] = round(np.quantile(sent_token_len_array, 1 - cut_quantile_high), 10)
                
                #coletando somente as sentenças filtradas
                sents_to_get = []
                for i in sent_n_tokens_dic.keys():
                    if (sent_token_stats['low_quantile'] < sent_n_tokens_dic[i] < sent_token_stats['high_quantile']) and (sent_n_tokens_dic[i] > min_sent_tokens):                    
                        first_ref_section_index = reference_section_index_list[0]
                        #considera-se como inicio das referências o primeiro index encontrado nos patterns
                        if i not in range( first_ref_section_index, sentDF.index.values[-1] + 1):
                            sents_to_get.append(sentDF.loc[i, 'Sentence'])
            
                #salvando as sents filtradas
                sentence_counter = save_sentences_to_csv(sents_to_get,
                                                         filename, 
                                                         sentence_counter, 
                                                         folder = 'Sentences_filtered', 
                                                         diretorio=self.diretorio)

                #salvando o file_index
                save_index_to_TXT(filename, 'counter_File_index_filter_sentences', '/Outputs/LOG', diretorio = self.diretorio)

                #contador de documentos com referências encontradas
                ref_found_counter += 1                
                print('Total de documentos cujas referências foram encontradas: ', round(ref_found_counter/total_file_counter, 3) * 100, ' % (Total: ', total_file_counter,' )', )
                print('Número de marcadores - "references": ', counter_references_marker, ' ; "conclusions" : ', counter_conclusions_marker)
                
                if check_function is True:
                    print(filename)
                    print('refs found: ', len(reference_section_index_list))
                    #time.sleep(3)
            
            else:
                print('Atenção! As referências não foram encontradas para o arquivo: ', filename)


    def find_text_sections(self, sections_names=['A', 'B']):

        print('\n( Function: find_text_sections )')
        
        import os
        import time
        import regex as re
        import pandas as pd
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import save_index_to_TXT
        
        if not os.path.exists(self.diretorio + '/Outputs/Sections'):
            os.makedirs(self.diretorio + '/Outputs/Sections')
        
        #dividindo os dois nomes de seção
        begin_section_pattern = sections_names[0]
        end_section_pattern = sections_names[1]
        print('Section begin pattern: ', begin_section_pattern)
        print('Section end pattern: ', end_section_pattern)
        
        extracted_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences', file_type = 'csv')
        
        (current_file_name, 
        current_file_index, 
        last_file_index, 
        sentence_counter, 
        sentence_counter_numbers_replaced) = self.resume_text_processing(input_file_list = extracted_DB_documents, 
                                                                         output_folder = 'Sentences',
                                                                         log_TXT_file_name = 'counter_File_index_find_sections')

                                                                             
        for filename in extracted_DB_documents[ current_file_index : ]:

            print('Processing ', filename, '...')
            sentDF = pd.read_csv(self.diretorio + f'/Outputs/Sentences/{filename}.csv', index_col = 0)
            
            #caso exista a sentença filtrada
            try:
                #varrendo as sentenças filtradas
                sentDF_filtered = pd.read_csv(self.diretorio + f'/Outputs/Sentences_filtered/{filename}.csv', index_col = 0)
                index_len = len(sentDF_filtered.index)
            except FileNotFoundError:
                continue

            begin_index = None
            begin_index_filtered = None
            end_index = None
            end_index_filtered = None
            begin_check = False
            end_check = False
            
            #varrendo as sentenças em formato raw
            for i in sentDF.index:
                sent = sentDF.loc[i, 'Sentence']
                if re.match(begin_section_pattern, sent) and begin_check is False:
                    begin_index = i
                    begin_check = True
                elif re.match(end_section_pattern, sent) and end_check is False:
                    end_index = i
                    end_check = True
                
            #caso os separadores de seção tenham sido encontrados no DF de sentenças
            if None not in (begin_index, end_index):
                
                counter = 0
                sent_fil_dic = {}
                for i in sentDF_filtered.index:
                    sent = sentDF_filtered.loc[i, 'Sentence']
                    sent_fil_dic[sent] = counter
                    counter += 1 
                
                #identificando as sentenças no DF de sentenças filtradas
                for i in range(begin_index, end_index + 1):
                    sent = sentDF.loc[i, 'Sentence']
                    if sent in sent_fil_dic.keys():
                        begin_index_filtered = sent_fil_dic[sent]
                        break
                                
                for i in range(end_index, sentDF.index[-1] + 1):
                    sent = sentDF.loc[i, 'Sentence']
                    if sent in sent_fil_dic.keys():
                        end_index_filtered = sent_fil_dic[sent]
                        break
                
                #caso os separadores de seção tenham sido encontrados no DF de sentenças filtradas
                if None not in (begin_index_filtered, end_index_filtered) and (end_index_filtered > begin_index_filtered): 
                    #criando os targets
                    first_part = [0] * begin_index_filtered
                    sec_part = [1] * (end_index_filtered - begin_index_filtered)
                    third_part = [0] * (index_len - end_index_filtered)
                    concat_list = first_part + sec_part + third_part
                    
                    sentDF_filtered_copy = sentDF_filtered.copy()
                    sentDF_filtered_copy['Target'] = concat_list
                    sentDF_filtered_copy.to_csv(self.diretorio + f'/Outputs/Sections/{filename}.csv')
                    print(f'Section DF exportada em ~/Outputs/Sections/{filename}.csv')

            #salvando o file_index
            save_index_to_TXT(filename, 'counter_File_index_find_sections', '/Outputs/LOG', diretorio = self.diretorio)



    def get_Ngrams_appearance(self, min_sent_tokens = 5, get_3gram_appearence = False):
        
        print('\n( Function: get_Ngrams_appereance )')
        
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import check_memory_use
        from FUNCTIONS import load_index_from_TXT
        from FUNCTIONS import save_index_to_TXT
        from functions_TOKENS import from_token_indexed_csv_to_DICT
        from functions_TOKENS import get_tokens_from_sent
        #import regex as re
        import pandas as pd
        from nltk.corpus import stopwords
        stopwords_list = stopwords.words('english') 
        import os
        import psutil
        #registrando o uso de memória        
        self.initial_memory_used = {}        
        self.initial_memory_used['ram'] = psutil.virtual_memory()[3]
        self.initial_memory_used['buffers'] = psutil.virtual_memory()[7]
        self.initial_memory_used['cached'] = psutil.virtual_memory()[8]

        #carregando a lista de arquivos processados
        extracted_filtered_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', 
                                                                         file_type = 'csv')

        #checando a existência das pastas
        if not os.path.exists(self.diretorio + '/Outputs/nGrams/Appearence'):
            os.makedirs(self.diretorio + '/Outputs/nGrams/Appearence')

        #checando se os arquivos já tiveram os tokens contados
        if os.path.exists(self.diretorio + '/Outputs/nGrams/Appearence/Tokens_appereance_counts.csv'):
            token_dic = from_token_indexed_csv_to_DICT(self.diretorio + '/Outputs/nGrams/Appearence/Tokens_appereance_counts.csv')
            #contador de arquivos                        
            file_name = load_index_from_TXT('counter_File_index_get_token_appearance', '/Outputs/LOG', diretorio = self.diretorio)
            print('Token_counter_File_name: ', file_name)
            file_index = extracted_filtered_DB_documents.index(file_name) + 1
            #print('Próximo PDF a ser processado: ', extracted_filtered_DB_documents[ file_index : file_index + 10 ])
            #contador de sentença
            sentence_counter = load_index_from_TXT('Sentence_counts', '/Outputs/LOG', diretorio = self.diretorio)
            print('Sentence_counter: ', sentence_counter)
            
        else:
            #definindo os dicionários para contar os findings de tokens
            token_dic = {}
            #contador de arquivos
            file_index = 0
            file_name = 'PDF00000'
            print('Token_counter_File_name: ', file_name)
            #contador de sentença
            sentence_counter = 0
            print('Sentence_counter: ', sentence_counter)
        print('Token_counter_File_index: ', file_index)
        
        
        #caso o último arquivo processado seja o ultimo da pasta
        if extracted_filtered_DB_documents[ -1 ] == file_name:
            print(f'Os tokens de todos os documentos ({len(self.DB_documents)}) já foram contados')
            print('> Abortando função: text_objects.get_Ngrams_appereance')
            return
        
        print('Finding 2grams CSV...')
        #checando se há o DF de 2grams
        if os.path.exists(self.diretorio + '/Outputs/nGrams/Appearence/2grams_appereance_counts.csv'):
            #carregando o DF com os 2grams
            n2grams_dic = from_token_indexed_csv_to_DICT(self.diretorio + '/Outputs/nGrams/Appearence/2grams_appereance_counts.csv')
            print('DataFrame de 2grams encontrado...')            
        else:
            #criando um dicionário de 2grams
            n2grams_dic = {}       
            print('Criando DF de 2grams...')

        print('Finding 3grams CSV...')
        #checando se há o DF de 3grams
        if os.path.exists(self.diretorio + '/Outputs/nGrams/Appearence/3grams_appereance_counts.csv'):
            #carregando o DF com os 3grams
            n3grams_dic = from_token_indexed_csv_to_DICT(self.diretorio + '/Outputs/nGrams/Appearence/3grams_appereance_counts.csv')
            print('DataFrame de 3grams encontrado...')            
        else:
            #criando um dicionário de 3grams
            n3grams_dic = {}       
            print('Criando DF de 3grams...')
            
        
        #varrendo os documentos .csv com as sentenças
        for file_tag in extracted_filtered_DB_documents[ file_index : ]:
            
            #lista para coletar os tokens presentes no arquivo PDF
            tokens_got_in_PDF = []
            n2grams_got_in_PDF = []
            n3grams_got_in_PDF = []
            
            #checando o uso de memória                    
            check_memory_use(self.initial_memory_used)
            print(f'\nProcessando CSV (~/Outputs/Sentences_filtered/{file_tag}.csv)')
            
            #abrindo o csv com as sentenças do artigo
            sentDF = pd.read_csv(self.diretorio + '/Outputs/Sentences_filtered/' + f'{file_tag}.csv', index_col = 0)
            
            #analisando cada sentença
            for index in sentDF.index:

                #lista para coletar os tokens presentes na sentença
                tokens_got_in_SENT = []
                n2grams_got_in_SENT = []
                n3grams_got_in_SENT = []
                
                sent = sentDF.loc[index, 'Sentence']
                sentence_counter += 1
                
                
                #splitando a sentença em tokens
                sent_tokens = get_tokens_from_sent(sent, tokenizer_func = self.tokenizer_func)
                
                #se a sentença tiver mais que 5 tokens
                if len(sent_tokens) > min_sent_tokens:
                    pass
                else:
                    #print('Sentença ignorada: ', sent_tokens, '')
                    continue
                    
                #varrendo os tokens
                for token in sent_tokens:

                    #caso haja números colados, não se pega o token
                    if ('&' in token):
                        continue
                    #caso haja underline, não se pega o token
                    if ('_' in token):
                        continue
                    
                    get_token = False
                    #caso seja token de letra
                    if token[0] in 'abcdefghijklmnopqrstuvxwyz%$':
                        get_token = True
                    #caso seja token de número (pega-se o número nessa etapa para que depois seja calculado o score dos Ngrams)
                    #OBS: entretanto, os números não são usados no cálculo do TF-IDF nem dos WV
                    elif token[0] in ('–0123456789'):
                        get_token = True
                    #caso não seja nem número nem letra, não se pega o token
                    else:
                        #print('Token ignorado: ', token, '')
                        continue
                            
                    if (get_token is True):
                        
                        try:
                            #print('Econtrado o token ( ', token, ' ) na sentença.')
                            token_dic[token]['TOTAL'] += 1
                            #print('Token ' + token + ' encontrado.')
    
                            if (token_dic[token]['check_SENT_pres'] is False):
                                #checando a presença do token no PDF
                                token_dic[token]['SENT'] += 1
                                token_dic[token]['check_SENT_pres'] = True
                                
                            if (token_dic[token]['check_PDF_pres'] is False):
                                #checando a presença do token no PDF
                                token_dic[token]['PDF'] += 1
                                token_dic[token]['check_PDF_pres'] = True
                        
                        except KeyError:
                            token_dic[token] = {}
                            token_dic[token]['TOTAL'] = 1
                            token_dic[token]['SENT'] = 1
                            token_dic[token]['PDF'] = 1
                            token_dic[token]['check_PDF_pres'] = True
                            token_dic[token]['check_SENT_pres'] = True
                        
                        tokens_got_in_SENT.append(token)
                        tokens_got_in_PDF.append(token)
                    

                #################################################################################        
                        
                #retirando as stopwords
                sent_tokens_filtered = [ token for token in sent_tokens if token not in stopwords_list]
                #mostrando as sentenças
                #if len(sent_tokens_raw) != len(sent_tokens):
                #    print('TOKENS SENT RAW: ', sent_tokens_raw)
                #    print('TOKENS SENT FILTERED1: ', sent_tokens)
                #    print('TOKENS SENT FILTERED2: ', sent_tokens_filtered)
                #    time.sleep(1)
                
                #varrendo os tokens para 2grams
                for token_index in range(len(sent_tokens_filtered)):
                    try:
                        token_0 = sent_tokens_filtered[token_index]
                        token_1 = sent_tokens_filtered[token_index + 1]
                            
                        bigram = token_0 + '_' + token_1
                        #print(token_0, token_1)
                        
                        #varrendo os tokens da sentença
                        get_bigram = False
                        #caso haja números colados, não se pega o token
                        if ('&' in token_0) or ('&' in token_1):
                            continue
                        #caso haja underline, não se pega o token
                        if ('_' in token_0) or ('_' in token_1):
                            continue
                        
                        #caso seja token de letra
                        if token_0[0] in 'abcdefghijklmnopqrstuvxwyz%$':
                            get_bigram = True
                        #caso seja token de número
                        elif token_0[0] in ('–0123456789'):
                            get_bigram = True
                        #caso não seja nem número nem letra, não se pega o token
                        else:
                            continue
                        
                        if (get_bigram is True):
                            try:                            
                                #print('Econtrado o bigram ( ', bigram, ' ) na sentença.')
                                n2grams_dic[bigram]['TOTAL'] += 1
                                #print('Bigram ' + bigram + ' encontrado.')
            
                                if (n2grams_dic[bigram]['check_SENT_pres'] is False):
                                    #checando a presença do bigram no PDF
                                    n2grams_dic[bigram]['SENT'] += 1
                                    n2grams_dic[bigram]['check_SENT_pres'] = True
                                    
                                if (n2grams_dic[bigram]['check_PDF_pres'] is False):
                                    #checando a presença do bigram no PDF
                                    n2grams_dic[bigram]['PDF'] += 1
                                    n2grams_dic[bigram]['check_PDF_pres'] = True
                                
                            except KeyError:
                                n2grams_dic[bigram] =  {}
                                n2grams_dic[bigram]['TOTAL'] = 1
                                n2grams_dic[bigram]['SENT'] = 1
                                n2grams_dic[bigram]['PDF'] = 1
                                n2grams_dic[bigram]['check_PDF_pres'] = True
                                n2grams_dic[bigram]['check_SENT_pres'] = True

                            n2grams_got_in_SENT.append(bigram)
                            n2grams_got_in_PDF.append(bigram)
                        
                    except IndexError:
                        continue

                #varrendo os tokens para 3grams                                        
                if get_3gram_appearence is True:
                    for token_index in range(len(sent_tokens_filtered)):
                        try:
                            token_0 = sent_tokens_filtered[token_index]
                            token_1 = sent_tokens_filtered[token_index + 1]
                            token_2 = sent_tokens_filtered[token_index + 2]
                            
                            trigram = token_0 + '_' + token_1 + '_' + token_2
                            #print(token_0, token_1, token_2)
                            
                            #varrendo os tokens da sentença
                            get_trigram = False
                            #caso haja números colados, não se pega o token
                            if ('&' in token_0) or ('&' in token_1) or ('&' in token_2):
                                continue
                            #caso haja underline, não se pega o token
                            if ('_' in token_0) or ('_' in token_1) or ('_' in token_2):
                                continue
                            
                            #caso seja token de letra
                            if token_0[0] in 'abcdefghijklmnopqrstuvxwyz%$':
                                get_trigram = True
                            #caso seja token de número
                            elif token_0[0] in ('–0123456789'):
                                get_trigram = True
                            #caso não seja nem número nem letra, não se pega o token
                            else:
                                continue
                            
                            if (get_trigram is True):
                                try:
                                    #print('Econtrado o trigram ( ', trigram, ' ) na sentença.')
                                    n3grams_dic[trigram]['TOTAL'] += 1
                                    #print('Trigram ' + trigram + ' encontrado.')
                
                                    if (n3grams_dic[trigram]['check_SENT_pres'] is False):
                                        #checando a presença do trigram no PDF
                                        n3grams_dic[trigram]['SENT'] += 1
                                        n3grams_dic[trigram]['check_SENT_pres'] = True
                                        
                                    if (n3grams_dic[trigram]['check_PDF_pres'] is False):
                                        #checando a presença do trigram no PDF
                                        n3grams_dic[trigram]['PDF'] += 1
                                        n3grams_dic[trigram]['check_PDF_pres'] = True
                                    
                                except KeyError:
                                    n3grams_dic[trigram] =  {}
                                    n3grams_dic[trigram]['TOTAL'] = 1
                                    n3grams_dic[trigram]['SENT'] = 1
                                    n3grams_dic[trigram]['PDF'] = 1
                                    n3grams_dic[trigram]['check_PDF_pres'] = True
                                    n3grams_dic[trigram]['check_SENT_pres'] = True
    
                                n3grams_got_in_SENT.append(trigram)
                                n3grams_got_in_PDF.append(trigram)
                        
                        except IndexError:
                                continue    


                #limpando os check de token e Ngrams nas sentenças
                for token in tokens_got_in_SENT:
                    token_dic[token]['check_SENT_pres'] = False
                
                for bigram in n2grams_got_in_SENT:
                    n2grams_dic[bigram]['check_SENT_pres'] = False
                
                if get_3gram_appearence is True:
                    for trigram in n3grams_got_in_SENT:
                        n3grams_dic[trigram]['check_SENT_pres'] = False
                
                del sent
                del sent_tokens
                del tokens_got_in_SENT
                del n2grams_got_in_SENT
                if get_3gram_appearence is True:
                    del n3grams_got_in_SENT

            #limpando os check de token e Ngrams nos PDFs
            for token in tokens_got_in_PDF:
                token_dic[token]['check_PDF_pres'] = False
            for bigram in n2grams_got_in_PDF:
                n2grams_dic[bigram]['check_PDF_pres'] = False
            if get_3gram_appearence is True:
                for trigram in n3grams_got_in_PDF:
                    n3grams_dic[trigram]['check_PDF_pres'] = False
                
            print('Sentence counter: ', sentence_counter)
            print('Tokens counter: ', len(token_dic.keys()))
            print('2gram counter: ', len(n2grams_dic.keys()))
            if get_3gram_appearence is True:
                print('3gram counter: ', len(n3grams_dic.keys()))
            
            #salvando a cada 30 PDFs ou quando for o último
            if file_index % 200 == 0 or extracted_filtered_DB_documents[file_index] == extracted_filtered_DB_documents[-1]:            
                print('#########################')
                print('Salvando os arquivos...')
                print('#########################')
                #salvando a contagem de tokens
                token_DF = pd.DataFrame.from_dict(token_dic, orient='index')
                token_DF.index.name = 'index'
                token_DF.sort_values(by=['TOTAL'], ascending=False, inplace=True)
                token_DF.to_csv(self.diretorio + '/Outputs/nGrams/Appearence/Tokens_appereance_counts.csv')
                
                n2gram_DF = pd.DataFrame.from_dict(n2grams_dic, orient='index')
                n2gram_DF.index.name = 'index'
                n2gram_DF.sort_values(by=['TOTAL'], ascending=False, inplace=True)
                n2gram_DF.to_csv(self.diretorio + '/Outputs/nGrams/Appearence/2grams_appereance_counts.csv')
                #print(n2gram_DF)
                
                if get_3gram_appearence is True:
                    n3gram_DF = pd.DataFrame.from_dict(n3grams_dic, orient='index')
                    n3gram_DF.index.name = 'index'
                    n3gram_DF.sort_values(by=['TOTAL'], ascending=False, inplace=True)
                    n3gram_DF.to_csv(self.diretorio + '/Outputs/nGrams/Appearence/3grams_appereance_counts.csv')
                    #print(n3gram_DF)
                
                #salvando o número do último arquivo processado
                save_index_to_TXT(file_tag, 'counter_File_index_get_token_appearance', '/Outputs/LOG', diretorio=self.diretorio)
                #salvando o número total de sentença (documents)
                save_index_to_TXT(sentence_counter, 'Sentence_counts', '/Outputs/LOG', diretorio=self.diretorio)

            file_index += 1
            del tokens_got_in_PDF
            del n2grams_got_in_PDF
            del n3grams_got_in_PDF
            del sentDF
            
            
        del token_dic
        del n2grams_dic
        del n3grams_dic
        del token_DF
        del n2gram_DF
        if get_3gram_appearence is True:
            del n3gram_DF
            


    def filter_Ngrams(self, N = 2,  threshold_2gram = 0.1, threshold_3gram = 0.1, n2gram_mim_score_allowed = 1e-7, n3gram_mim_score_allowed = 1e-16):
        
        print('\n( Function: filter_Ngrams )')

        #import time
        import regex as re
        import pandas as pd
        import os

        print('Abrindo a DF com a contagem de tokens...')
        #abrindo a contagem de tokens e a quantidade total de documentos
        token_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/Appearence/Tokens_appereance_counts.csv')
        token_DF.dropna(inplace=True)
        token_DF.set_index('index', inplace=True)

        #checando a existência das pastas
        if not os.path.exists(self.diretorio + '/Outputs/nGrams/Filtered_Scores'):
            os.makedirs(self.diretorio + '/Outputs/nGrams/Filtered_Scores')
        if not os.path.exists(self.diretorio + '/Outputs/nGrams/To_replace'):
            os.makedirs(self.diretorio + '/Outputs/nGrams/To_replace')
        
        if N == 2:
            try:
                print('Abrindo a DF com a contagem de bigrams...')
                n2grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/Appearence/2grams_appereance_counts.csv')
                n2grams_DF.dropna(inplace=True)
                n2grams_DF.set_index('index', inplace=True)
                n2gram_scores_dic = {}
                print('Existem ', n2grams_DF.shape[0], ' bigrams para processar.')
            except FileNotFoundError:
                print('Erro! Nenhum arquivo CSV com 2grams encontrado.')
            
            if not os.path.exists(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n2grams_scores.csv'):
                print('Calculando os scores dos bigrams...')
                counter = 0
                for index in n2grams_DF.index:
                    try:
                        bigram = index
                        if bigram[0] in 'abcdefghijklmnopqrstuvxwyz':
                            #encontrando os tokens a partir dos bigrams
                            match_indeces = re.search('.+_', bigram).span()
                            token_0 = bigram[ : match_indeces[1] - 1 ]
                            token_1 = bigram[ match_indeces[1] :  ]
                            n2gram_scores_dic[bigram] = {}
                            n2gram_scores_dic[bigram]['Score'] = ( n2grams_DF.loc[bigram , 'TOTAL'] - threshold_2gram ) / ( token_DF.loc[token_0 , 'TOTAL'] * token_DF.loc[token_1 , 'TOTAL'] )
                            
                            token_tuple = (token_0, token_1)
                            deltas = []
                            for i in range(len(token_tuple)):
                                delta = ( ( token_DF.loc[token_tuple[i] , 'TOTAL'] - n2grams_DF.loc[bigram , 'TOTAL'] ) / token_DF.loc[token_tuple[i] , 'TOTAL'] ) * 100
                                n2gram_scores_dic[bigram]['delta_token_' + str(i)] = delta
                                deltas.append(delta)
                            n2gram_scores_dic[bigram]['min_delta'] = min(deltas)
                            n2gram_scores_dic[bigram]['token_0'] = token_0
                            n2gram_scores_dic[bigram]['token_1'] = token_1
                            
                            #print('token_0: ', token_0, ' ; token_1: ', token_1)
                            #print(n2gram_scores_dic[bigram])
                            #time.sleep(0.1)
                        
                        if counter % 100000 == 0:
                            print('n2gram processed ', counter)
                        counter += 1
                    
                    except KeyError:
                        continue    
                
                n2gram_scores_DF = pd.DataFrame.from_dict(n2gram_scores_dic, orient='index')
                #n2gram_scores_series.sort_values(ascending=False, inplace=True)                
                n2gram_scores_DF.to_csv(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n2grams_scores.csv')
        
            else:
                n2gram_scores_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n2grams_scores.csv', index_col = 0)
                

            print('Filtering n2gram DF...')
            n2gram_scores_DF = n2gram_scores_DF[ n2gram_scores_DF['Score'] > n2gram_mim_score_allowed ]
            n2gram_scores_DF.sort_values(by=['min_delta'], ascending=True, inplace=True)            
            n2grams_concat = pd.concat([n2grams_DF, n2gram_scores_DF], axis = 1)
            n2grams_concat.dropna(inplace = True)
            n2grams_concat.index.name = 'index'
            n2grams_concat.to_csv(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n2grams_filtered.csv')
            print('Saving to ~/Outputs/nGrams/Filtered_Scores/n2grams_filtered.csv')

            if os.path.exists(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv'):
                n2grams_to_replace = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')
                n2grams_to_replace.dropna(inplace = True)
                n2grams_to_replace.set_index('index', inplace=True)
                n2grams_to_replace.to_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')
                print('DropNA to ~/Outputs/nGrams/To_replace/n2grams_to_replace.csv')
                del n2grams_to_replace             
                
            del token_DF
            del n2grams_DF
            del n2gram_scores_dic
            del n2gram_scores_DF
            del n2grams_concat            


        if N == 3:
            try:
                print('Abrindo a DF com a contagem de trigrams...')
                n3grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/Appearence/3grams_appereance_counts.csv')
                n3grams_DF.dropna(inplace=True)
                n3grams_DF.set_index('index', inplace=True)
                n3gram_scores_dic = {}
                print('Existem ', n3grams_DF.shape[0], ' trigrams para processar.')
            except FileNotFoundError:
                print('Erro! Nenhum arquivo CSV com 3grams encontrado.')
            
            if not os.path.exists(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n3grams_scores.csv'):                
                print('Calculando os scores dos trigrams...')
                counter = 0
                for index in n3grams_DF.index:
                    try:    
                        trigram = index
                        if trigram[0] in 'abcdefghijklmnopqrstuvxwyz':
                            #encontrando os tokens a partir dos bigrams
                            trigram = index
                            match_indeces = re.search('_.+_', trigram).span()
                            token_0 = trigram[ : match_indeces[0] ]
                            token_1 = trigram[ match_indeces[0] + 1 : match_indeces[1] - 1 ]
                            token_2 = trigram[ match_indeces[1] : ]
                            n3gram_scores_dic[trigram] = {}
                            n3gram_scores_dic[trigram]['Score'] = ( n3grams_DF.loc[trigram , 'TOTAL'] - threshold_3gram ) / ( token_DF.loc[token_0 , 'TOTAL'] * token_DF.loc[token_1 , 'TOTAL'] * token_DF.loc[token_2 , 'TOTAL'] )
                            
                            token_tuple = (token_0, token_1, token_2)
                            deltas = []
                            for i in range(len(token_tuple)):
                                delta = ( ( token_DF.loc[token_tuple[i] , 'TOTAL'] - n3grams_DF.loc[trigram , 'TOTAL'] ) / token_DF.loc[token_tuple[i] , 'TOTAL'] ) * 100
                                n3gram_scores_dic[trigram]['delta_token_' + str(i)] = delta
                                deltas.append(delta)
                            n3gram_scores_dic[trigram]['min_delta'] = min(deltas)
                        
                            #print('token_0: ', token_0, ' ; token_1: ', token_1, ' ; token_2: ', token_2)
                            #print(n3gram_scores_dic[trigram])
                            #time.sleep(0.1)
            
                        if counter % 100000 == 0:
                            print('n3gram processed ', counter)
                        counter += 1        
                    
                    except KeyError:
                        continue
    
                n3gram_scores_DF = pd.DataFrame.from_dict(n3gram_scores_dic, orient='index')
                #n3gram_scores_series.sort_values(ascending=False, inplace=True)                
                n3gram_scores_DF.to_csv(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n3grams_scores.csv')
        
            else:
                n3gram_scores_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n3grams_scores.csv', index_col = 0)
                
            print('Filtering n3gram DF...')
            n3gram_scores_DF = n3gram_scores_DF[ n3gram_scores_DF['Score'] > n3gram_mim_score_allowed ]
            n3gram_scores_DF.sort_values(by=['min_delta'], ascending=True, inplace=True)                        
            n3grams_concat = pd.concat([n3grams_DF, n3gram_scores_DF], axis = 1)
            n3grams_concat.dropna(inplace = True)
            n3grams_concat.index.name = 'index'
            n3grams_concat.to_csv(self.diretorio + '/Outputs/nGrams/Filtered_Scores/n3grams_filtered.csv')
            print('Saving to ~/Outputs/nGrams/Filtered_Scores/n3grams_filtered.csv')
            
            if os.path.exists(self.diretorio + '/Outputs/nGrams/To_replace/n3grams_to_replace.csv'):
                n3grams_to_replace = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n3grams_to_replace.csv')
                n3grams_to_replace.dropna(inplace = True)
                n3grams_to_replace.set_index('index', inplace=True)
                n3grams_to_replace.to_csv(self.diretorio + '/Outputs/nGrams/To_replace/n3grams_to_replace.csv')
                print('DropNA to ~/Outputs/nGrams/To_replace/n3grams_to_replace.csv')
                del n3grams_to_replace
            
            del token_DF
            del n3grams_DF
            del n3gram_scores_dic
            del n3gram_scores_DF
            del n3grams_concat
            
                
        
    def find_IDF(self, min_token_appereance = 1):
        
        print('\n( Function: find_IDF )')
        
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import load_index_from_TXT
        import pandas as pd
        import math
        import os
        
        if os.path.exists(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv'):
            print('O IDF já foi calculado.')
            print('> Abortando função: text_objects.find_IDF')
            return
            
        #carregando a lista de arquivos
        sent_file_list = sorted(os.listdir(self.diretorio + '/Outputs/Sentences_filtered')) #lista de arquivos
        print('Total de PDFs encontrados: ', len(sent_file_list))
        #testar se há arquivos no diretório ~/Outputs/Sentences_filtered
        if len(sent_file_list) == 0:
            print('ERRO!')
            print('Não há arquivos no self.diretorio ~/Outputs/Sentences_filtered.')
            print('Usar a função "text_objects.get_text_objects()"')
            return

        #montando a lista de arquivos de sentenças extraídas
        n_PDFs = len( get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', file_type = 'csv') )
                
        #abrindo a contagem de ngrams e a quantidade total de documentos
        n1gram_count_appereance = pd.read_csv(self.diretorio + '/Outputs/nGrams/Appearence/Tokens_appereance_counts.csv')
        n1gram_count_appereance.dropna(inplace=True)
        n1gram_count_appereance.set_index('index', inplace=True)
        print('Shape n1gram DF: ', n1gram_count_appereance.shape)
        
        if (self.replace_Ngrams is True):
            n2gram_count_appereance = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')
            n2gram_count_appereance.dropna(inplace=True)
            n2gram_count_appereance.set_index('index', inplace=True)
            n2gram_count_appereance = n2gram_count_appereance[['TOTAL', 'PDF', 'SENT', 'check_PDF_pres', 'check_SENT_pres']].copy()
            print('Shape n2gram DF: ', n2gram_count_appereance.shape)
            
            '''
            n3gram_count_appereance = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n3grams_to_replace.csv')
            n3gram_count_appereance.dropna(inplace=True)
            n3gram_count_appereance.set_index('index', inplace=True)
            n3gram_count_appereance = n3gram_count_appereance[['TOTAL', 'PDF', 'SENT', 'check_PDF_pres', 'check_SENT_pres']].copy()
            print(n3gram_count_appereance.shape)
            '''
            
            #concatenando todos os Ngrams DFs
            ngram_count_appereance = pd.concat([n1gram_count_appereance, n2gram_count_appereance], axis = 0)
            print('Shape CONCAT ngrams DF: ', ngram_count_appereance.shape)
        else:
            ngram_count_appereance = n1gram_count_appereance
            print('Shape CONCAT ngrams DF: ', ngram_count_appereance.shape)
        
            
        n_sentences = load_index_from_TXT('Sentence_counts', '/Outputs/LOG', diretorio = self.diretorio)
        print('Calculando o IDF para: ', n_sentences, ' documentos (sentenças).')
                
        #calculando o IDF
        IDF = {}
        for token in ngram_count_appereance.index:
            #só serão tomados tokens que aparecem mais de n vez
            cond1 = ngram_count_appereance.loc[token, 'PDF'] >= min_token_appereance
            cond2 = token[0] in 'abcdefghijklmnopqrstuvxwyz%$'
            if False not in (cond1,cond2):                
                #cálculo do IDF para sentenças e PDFs
                SENT_IDF = round ( math.log( n_sentences / ngram_count_appereance.loc[token, 'SENT'] ) , 8 ) 
                PDF_IDF = round ( math.log( n_PDFs / ngram_count_appereance.loc[token, 'PDF'] ) , 8 )
                
                IDF[token] = {}
                IDF[token]['IDF_SENT'] = SENT_IDF
                IDF[token]['IDF_PDF'] = PDF_IDF

        #cálculo do TF normalizado pelo número total de tokens da coleção e número de PDFs da coleção
        #OBS1: esse TF calculado abaixo será usado no SUBSAMPLING durante a determinação dos word vectors
        #OBS2: o TF para determinação das matrizes TF-IDF da coleção é normaliado pelo número de tokens de cada sentença
        for token in IDF.keys():
            TOTAL_TOKEN_NORM_TF = ngram_count_appereance.loc[token, 'TOTAL'] / len(IDF.keys())
            IDF[token]['TF_TOKEN_NORM'] = TOTAL_TOKEN_NORM_TF

        #salvando o IDF
        IDF_ = pd.DataFrame.from_dict(IDF, orient='index')
        IDF_.to_csv(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv')
        print('Total de tokens no IDF DF: ', len(IDF_.index), '')
        
        del ngram_count_appereance
        del n1gram_count_appereance
        del IDF
        del IDF_
        if (self.replace_Ngrams is True):
            del n2gram_count_appereance
        

    
    def set_TFIDF_log_and_sent_stats(self, file_batch_size = 1000):

        print('\n( Function: set_TFIDF_log_and_sent_stats )')        

        import os
        import pandas as pd
        from FUNCTIONS import save_dic_to_json
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import get_file_batch_index_list
        from FUNCTIONS import get_tag_name

        #checando o LOG file
        if os.path.exists(self.diretorio + f'/Outputs/LOG/TFIDF_batches_log_rNgram_{self.replace_Ngrams}.json'):
            print('O TFIDF_LOG e SENT_INDEX já foram extraídos.')
            print('> Abortando função: text_objects.set_TFIDF_log_and_sent_stats')
            return

        SENT_INDEX_DIC = {}
        LOG_batches = {}    
        
        #carregando a lista de arquivos processados
        print('Carregando os nomes dos arquivos .csv filtrados...')
        extracted_filtered_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', file_type = 'csv')
        
        #número de documentos
        n_documents = len(extracted_filtered_DB_documents)
        
        #caso o valor de file_batch_size seja incorreto
        if file_batch_size > n_documents:
            print('ERRO!')
            print('O valor inserido para o "file_batch_size" é maior que o número total de arquivos em ~/Outputs/Sentences_filtered')
            return
        
        #criando os batches de arquivos
        batch_indexes = get_file_batch_index_list(n_documents, file_batch_size)
        
        print('Getting LOG and stats (TFIDF_log.json; SENT_INDEX.csv)...')
        #varrendo todos os slices
        #contador de sentença no total
        count_sents_total = 0
        #número do batch
        batch_counter_number = 0
        for sl in batch_indexes:

            #dicionário para cada batch
            batch_counter_number += 1
            c_batch_counter_number = get_tag_name(batch_counter_number, prefix = '')
            LOG_batches[c_batch_counter_number] = {}
            print('Processing batch: ', c_batch_counter_number, '; indexes: ', sl)
            
            #contador de sentenças do batch
            count_sents_batch = 0
            for i in range(sl[0], sl[1]+1):
                
                filename = extracted_filtered_DB_documents[i]
                #print('Processing ', filename, '...')
                DFsents = pd.read_csv(self.diretorio + f'/Outputs/Sentences_filtered/{filename}.csv', index_col = 0)

                count_sents_batch += len(DFsents.index)
                count_sents_total += len(DFsents.index)
                #print('Total sents: ', count_sents_total)
                
                #obtendo os indexes dos documentos para cada file
                SENT_INDEX_DIC[filename] = {}
                SENT_INDEX_DIC[filename]['initial_sent'] = DFsents.index.values[0]
                SENT_INDEX_DIC[filename]['last_sent'] = DFsents.index.values[-1]
                
                del DFsents
            
            LOG_batches[c_batch_counter_number]['first_file_index'] = sl[0]
            LOG_batches[c_batch_counter_number]['last_file_index'] = sl[1]
            LOG_batches[c_batch_counter_number]['n_sents'] = count_sents_batch
                                                    
        #salvando o log de slices para calcular a matriz TFIDF
        save_dic_to_json(self.diretorio + f'/Outputs/LOG/TFIDF_batches_log_rNgram_{self.replace_Ngrams}.json', LOG_batches)
        #salvando o index de sentença
        pd.DataFrame.from_dict(SENT_INDEX_DIC, orient='index').to_csv(self.diretorio + '/Outputs/LOG/SENT_INDEX.csv')
        print('Total sentence number processed: ', count_sents_total)
        
        del LOG_batches
        del SENT_INDEX_DIC



    def find_TFIDF(self):

        print('\n( Function: find_TFIDF )')

        #import time
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import check_memory_use
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import get_tag_name
        from functions_TOKENS import get_tokens_from_sent
        from scipy import sparse
        import regex as re
        import pandas as pd
        import os
        import psutil
        import h5py
        import numpy as np
        #registrando o uso de memória        
        self.initial_memory_used = {}        
        self.initial_memory_used['ram'] = psutil.virtual_memory()[3]
        self.initial_memory_used['buffers'] = psutil.virtual_memory()[7]
        self.initial_memory_used['cached'] = psutil.virtual_memory()[8]

        #checar o arquivo log
        if not os.path.exists(self.diretorio + f'/Outputs/LOG/TFIDF_batches_log_rNgram_{self.replace_Ngrams}.json'):
            print('ERRO!')
            print(f'O arquivo LOG (TFIDF_batches_log_rNgram_{self.replace_Ngrams}.json) não foi encontrado.')
            print('Executar a função text_objects.set_TFIDF_log_and_sent_stats.')
            return
        else:
            #abrindo o TFIDF log file com os slices        
            TFIDF_batches_log = load_dic_from_json(self.diretorio + f'/Outputs/LOG/TFIDF_batches_log_rNgram_{self.replace_Ngrams}.json')
            last_batch = int(sorted(list(TFIDF_batches_log.keys()))[-1])

        #checando se há o diretorio npz
        if not os.path.exists(self.diretorio + '/Outputs/TFIDF/npz'):
            os.makedirs(self.diretorio + '/Outputs/TFIDF/npz')
        
        #checando se há arquivos no ~/Outputs/TFIDF/npz
        npz_files_saved = get_filenames_from_folder(self.diretorio + '/Outputs/TFIDF/npz', file_type = 'npz')
        if npz_files_saved is not None:
            last_npz_file = sorted(npz_files_saved)[-1]
            try:
                last_batch_saved = int(re.findall(r'[0-9]+', last_npz_file)[0]) + 1
                print('\nÚltimo batch encontrado na pasta ~/Outputs/TFIDF/npz: ', last_batch_saved - 1, '\n')
                #caso haja erro na identificação dos arquivos npz
            except IndexError:
                last_batch_saved = 1
        #caso não tenha nenhum arquivo npz salvo
        else:
            last_batch_saved = 1        
            
        #caso todas as sentenças já tenham sido processadas
        if last_batch == last_batch_saved:
            print('O TFIDF de todos os documentos já foi calculado')
            print('> Abortando função: text_objects.find_TFIDF')
            return

        #checando se os IDFs foram calculados
        if not os.path.exists(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv'):
            print('Não há IDFs calculados em ~/Outputs/TFIDF')
            print('Usar a função "text_objects.find_IDF()"')
            print('> Abortando função: text_objects.find_TFIDF')
            return
        else:
            #carregando os IDFs
            IDF = pd.read_csv(self.diretorio + f'/Outputs/TFIDF/IDF_rNgram_{self.replace_Ngrams}.csv', index_col = 0)
            n_tokens = len(IDF.index)
        
            #caso não tenha nenhum token no IDF
            if n_tokens == 0:
                print('Erro! Não há tokens no IDF DF.')
                print('Diminuir o valor de min_token_appereance.')
                print('> Abortando função: text_objects.find_TFIDF')
                return            
        
        #caso os bigrams sejam substituidos
        n2grams_DF = None
        if (self.replace_Ngrams is True):
            #abrindo o DF com os N2GRAMS para substituir
            n2grams_DF = pd.read_csv(self.diretorio + '/Outputs/nGrams/To_replace/n2grams_to_replace.csv')
            n2grams_DF.dropna(inplace=True)
            n2grams_DF.set_index('index', inplace=True)

        #carregando a lista de arquivos processados
        print('Carregando os nomes dos arquivos .csv filtrados...')
        extracted_filtered_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', file_type = 'csv')
    
        #varrendo os batches estabelecidos no LOG
        for batch in range(last_batch_saved, last_batch + 1):

            #gerando o nome do arquivo para o batch
            c_batch_number = get_tag_name(batch, prefix = '')
            
            print('Criando o arquivo h5 para o batch: ', )
            h5 = h5py.File(self.diretorio + '/Outputs/TFIDF/temp_h5_matrix.h5', 'w')
            h5.create_dataset('data', shape = (TFIDF_batches_log[str(c_batch_number)]['n_sents'], n_tokens), dtype=np.float64)
            TFIDF_matrix = h5['data']
            print('TFIDF_temp_m_shape: ', TFIDF_matrix.shape)
                
            first_file_index_in_batch = TFIDF_batches_log[str(c_batch_number)]['first_file_index']
            last_file_index_in_batch = TFIDF_batches_log[str(c_batch_number)]['last_file_index']
            print('Arquivos do batch - ', extracted_filtered_DB_documents[first_file_index_in_batch], ' a ', extracted_filtered_DB_documents[last_file_index_in_batch])
            print('Indexes do batch - ', first_file_index_in_batch, ' a ', last_file_index_in_batch)
            
            row_N = 0
            #varrendo os files do batch
            for file_index  in range(first_file_index_in_batch, last_file_index_in_batch + 1):            

                #abrindo o csv com as sentenças do artigo
                filename = extracted_filtered_DB_documents[file_index]
                sentDF = pd.read_csv(self.diretorio + '/Outputs/Sentences_filtered/' + f'{filename}.csv', index_col = 0)
                print(f'\nProcessando CSV (~/Outputs/Sentences_filtered/{filename}.csv)')
                print('Primeira sentença do PDF (row number): ', row_N)

                #checando o uso de memória                    
                check_memory_use(self.initial_memory_used)
            
                #analisando cada sentença
                for index in sentDF.index:
                    sent = sentDF.loc[index, 'Sentence']
    
                    #splitando a sentença em tokens
                    sent_tokens = get_tokens_from_sent(sent, 
                                                       tokenizer_func = self.tokenizer_func,
                                                       n2grams_DF = n2grams_DF, 
                                                       replace_Ngrams = self.replace_Ngrams)
                    
                    #lista para coletar os tokens com TFIDF já calculados
                    collected_sent_tokens = []
    
                    #número de tokens por sentença
                    n_tokens_sent = len(sent_tokens)
                    
                    #checando a presença do token na sentença
                    for token in sent_tokens:
                        
                        #caso o TFIDF para o token já tenha sido calculado
                        if token not in collected_sent_tokens:
                            pass
                        else:
                            continue
                        
                        try:
                            #calculando o TF normalizando pelo número de tokens da sentença
                            TF_ = sent_tokens.count(token) / n_tokens_sent
                            IDF_ = IDF.loc[token, 'IDF_SENT']
        
                            #o valor da coluna (POS X) é a posição do token no IDF DF (index da IDF DF)
                            pos_token_in_IDF = np.argwhere(IDF.index.values == token)
                            col_N = pos_token_in_IDF[0,0]
                            
                            #calculando o TFIDF
                            TFIDF_ = TF_ * IDF_
                            TFIDF_matrix[row_N, col_N] = TFIDF_
                            
                            collected_sent_tokens.append(token)
                        
                        #caso o token não esteja no IDF
                        except KeyError:
                            TF_ = None
                            IDF_ = None
                            pos_token_in_IDF = None
                            col_N = None
                            TFIDF_ = None
                            continue
                        
                        del TF_
                        del IDF_
                        del pos_token_in_IDF
                        del col_N
                        del TFIDF_                
                    
                    row_N += 1
                    del sent
                    del sent_tokens
                    del collected_sent_tokens
                    del n_tokens_sent

            #salvando a sparse matrix do batch
            sm = sparse.csr_matrix(TFIDF_matrix, dtype = np.float64)
            sparse.save_npz(self.diretorio + f'/Outputs/TFIDF/npz/sparse_csr_rNgram_{self.replace_Ngrams}_{c_batch_number}.npz', sm, compressed=True)            
                        
            h5.close()        

            print(f'\nArquivo npz salvo (sparse_csr_rNgram_{self.replace_Ngrams}_{batch}.npz).')
            print('Arquivos salvos no npz - ', extracted_filtered_DB_documents[first_file_index_in_batch], ' a ', extracted_filtered_DB_documents[last_file_index_in_batch])
            
            del h5
            del TFIDF_matrix
            del sentDF

        if (self.replace_Ngrams is True):
            del n2grams_DF



    def resume_text_processing(self, input_file_list = None, output_folder = None, log_TXT_file_name = None):            
        
        import pandas as pd
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import load_index_from_TXT
        
        
        #montando a lista de documentos já processados
        proc_documents = get_filenames_from_folder(self.diretorio + f'/Outputs/{output_folder}', file_type = 'csv')
        
        #caso a substituição dos números esteja sendo feita
        sentence_counter_numbers_replaced = None
        
        #encontrando o index da última sentença
        if proc_documents != None and len(proc_documents) > 0:
            last_pdf_processed = proc_documents[-1]
            sentDF = pd.read_csv(self.diretorio + f'/Outputs/{output_folder}/{last_pdf_processed}.csv', index_col=0)
            sentence_counter = sentDF.index.values[-1] + 1
            print('Current Document (sentence) number = ', sentence_counter)
            if (self.replace_numbers is True):                
                sentDF_replaced = pd.read_csv(self.diretorio + f'/Outputs/{output_folder}_numbers_replaced/{last_pdf_processed}.csv', index_col=0)
                sentence_counter_numbers_replaced = sentDF_replaced.index.values[-1] + 1
                print('Current Document (sentence filtered) number ', sentence_counter_numbers_replaced)
        else:
            sentence_counter = 0
            print('Current Document (sentence) number = ', sentence_counter)
            if (self.replace_numbers is True):
                sentence_counter_numbers_replaced = 0
                print('Current Document (sentence filtered) number ', sentence_counter_numbers_replaced)

        #contador de arquivos processados
        try:
            current_file_name = load_index_from_TXT(log_TXT_file_name, '/Outputs/LOG', diretorio = self.diretorio)
            current_file_index = input_file_list.index(current_file_name) + 1
            last_file_index = input_file_list.index(current_file_name)
        except FileNotFoundError:
            current_file_name = 'PDF00000'
            current_file_index = 0
            last_file_index = 0
        
        return current_file_name, current_file_index, last_file_index, sentence_counter, sentence_counter_numbers_replaced            



    def find_sent_stats(self, mode = 'raw_sentences'):

        print('\n( Function: find_sent_stats )')
        print('mode: ', mode)
        
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import save_dic_to_json
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12,18), dpi=300)
        from nltk.corpus import stopwords
        stopwords_list = stopwords.words('english')
        
        if mode.lower() == 'raw_sentences':
            term_name = '_not_filtered'
            file_list = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences', 
                                                  file_type = 'csv')
            folder_name = 'Sentences'
        elif mode.lower() == 'filtered_sentences':
            term_name = ''
            file_list = get_filenames_from_folder(self.diretorio + '/Outputs/Sentences_filtered', 
                                                  file_type = 'csv')
            folder_name = 'Sentences_filtered'
        
        #checando os LOGs
        cond1 = os.path.exists(self.diretorio + f'/Outputs/LOG/stats_SENT_TOKENS{term_name}.json')
        cond2 = os.path.exists(self.diretorio + f'/Outputs/LOG/stats_PDF_SENT{term_name}.json')
        if False not in (cond1, cond2):
            print(f'O stats_SENT_TOKENS{term_name} e o stats_PDF_SENT{term_name} já foram extraídos.')
            print('> Abortando função: text_objects.find_sent_stats')
            return

        sent_token_len = {}
        sent_token_len['SW_True'] = []
        sent_token_len['SW_False'] = []        
        sent_token_stats = {}
        sent_token_stats['SW_True'] = {}
        sent_token_stats['SW_False'] = {}
        pdf_sent_len = []
        pdf_sent_stats = {}
        
        print(f'Varrendos os documentos em ~/Outputs/{folder_name}')
        for filename in file_list:
            
            print('Processing ', filename, '...')            
            DFsents = pd.read_csv(self.diretorio + f'/Outputs/{folder_name}/{filename}.csv', index_col = 0)

            #guardando o número de sentenças por documento para fazer a estatística
            pdf_sent_len.append(len(DFsents.index))
            
            #varrendo cada sentença de cada arquivo para determinação da SENT_TOKEN_STATS (estatística de token por sentença)
            for j in DFsents.index:
                sent = DFsents.loc[ j, 'Sentence']
                len_SW_False = len(sent.split())
                len_SW_True = len([token for token in sent.split() if (token.lower() not in stopwords_list and token[ : -1].lower() not in stopwords_list)])
                
                sent_token_len['SW_False'].append( len_SW_False )
                sent_token_len['SW_True'].append( len_SW_True )
                
        #calculando os parâmetros estatísticos de token por sentenças
        for i in range(len(['False', 'True'])):
            
            boolean = ('False', 'True')[i]
            
            sent_token_len_array = np.array(sent_token_len[f'SW_{boolean}'])
            
            sent_token_stats[f'SW_{boolean}']['min'] = int(np.min(sent_token_len_array))
            sent_token_stats[f'SW_{boolean}']['max'] = int(np.max(sent_token_len_array))
            sent_token_stats[f'SW_{boolean}']['avg'] = round(np.mean(sent_token_len_array), 10)
            sent_token_stats[f'SW_{boolean}']['std'] = round(np.std(sent_token_len_array), 10)
            sent_token_stats[f'SW_{boolean}']['median'] = round(np.median(sent_token_len_array), 10)
            sent_token_stats[f'SW_{boolean}']['1_quartile'] = round(np.quantile(sent_token_len_array, 0.25), 10)
            sent_token_stats[f'SW_{boolean}']['3_quartile'] = round(np.quantile(sent_token_len_array, 0.75), 10)
            sent_token_stats[f'SW_{boolean}']['0.05_quantile'] = round(np.quantile(sent_token_len_array, 0.05), 10)
            sent_token_stats[f'SW_{boolean}']['0.95_quantile'] = round(np.quantile(sent_token_len_array, 0.95), 10)
            
            axes[i].set_title(f'sent_token_len SW_{boolean}')
            axes[i].hist(sent_token_len_array, bins = 50, range=(0,150), color='gray', alpha=0.5)
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['avg'], color='green', label='mean', alpha=0.5)
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['avg'] + sent_token_stats[f'SW_{boolean}']['std'], color='red', alpha=0.5, label='+std')
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['avg'] - sent_token_stats[f'SW_{boolean}']['std'], color='red', alpha=0.5, label='-std')
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['median'], color='blue', alpha=0.5, label='median')
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['1_quartile'], color='orange', alpha=0.5, label='1_quartile')
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['3_quartile'], color='orange', alpha=0.5, label='3_quartile')
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['0.05_quantile'], color='black', alpha=0.5, label='0.05_quantile')
            axes[i].axvline(sent_token_stats[f'SW_{boolean}']['0.95_quantile'], color='black', alpha=0.5, label='0.95_quantile')
            axes[i].legend()

        #calculando os parâmetros estatísticos de sentença por PDF
        pdf_sent_len_array = np.array(pdf_sent_len)
        
        pdf_sent_stats['min'] = int(np.min(pdf_sent_len))
        pdf_sent_stats['max'] = int(np.max(pdf_sent_len))
        pdf_sent_stats['avg'] = round(np.mean(pdf_sent_len), 10)
        pdf_sent_stats['std'] = round(np.std(pdf_sent_len), 10)
        pdf_sent_stats['median'] = round(np.median(pdf_sent_len), 10)
        pdf_sent_stats['1_quartile'] = round(np.quantile(pdf_sent_len, 0.25), 10)
        pdf_sent_stats['3_quartile'] = round(np.quantile(pdf_sent_len, 0.75), 10)
        pdf_sent_stats['0.05_quantile'] = round(np.quantile(pdf_sent_len, 0.05), 10)
        pdf_sent_stats['0.95_quantile'] = round(np.quantile(pdf_sent_len, 0.95), 10)
        
        axes[2].set_title('pdf_sent_len')
        axes[2].hist(pdf_sent_len_array, bins = 50, color='gray', alpha=0.5)
        axes[2].axvline(pdf_sent_stats['avg'], color='green', label='mean', alpha=0.5)
        axes[2].axvline(pdf_sent_stats['avg'] + pdf_sent_stats['std'], color='red', alpha=0.5, label='+std')
        axes[2].axvline(pdf_sent_stats['avg'] - pdf_sent_stats['std'], color='red', alpha=0.5, label='-std')
        axes[2].axvline(pdf_sent_stats['median'], color='blue', alpha=0.5, label='median')
        axes[2].axvline(pdf_sent_stats['1_quartile'], color='orange', alpha=0.5, label='1_quartile')
        axes[2].axvline(pdf_sent_stats['3_quartile'], color='orange', alpha=0.5, label='3_quartile')
        axes[2].axvline(pdf_sent_stats['0.05_quantile'], color='black', alpha=0.5, label='0.05_quantile')
        axes[2].axvline(pdf_sent_stats['0.95_quantile'], color='black', alpha=0.5, label='0.95_quantile')
        axes[2].legend()       
                
        #salvando o sent_stats (estatística de token por sentença)
        save_dic_to_json(self.diretorio + f'/Outputs/LOG/stats_SENT_TOKENS{term_name}.json', sent_token_stats)
        #salvando o sent_stats (estatística de token por sentença)
        save_dic_to_json(self.diretorio + f'/Outputs/LOG/stats_PDF_SENT{term_name}.json', pdf_sent_stats)
        #salvando a figura com os pdf_sent_token stats
        fig.savefig(self.diretorio + f'/Outputs/LOG/pdf_sent_token_hist{term_name}.png')
        
        del sent_token_len
        del sent_token_stats
        del pdf_sent_len
        del pdf_sent_stats
        
        

    def looking_for_ref_section(self, sentDF, ref_section_index_range = 50, min_ref_to_find_for_section = 10, section_marker = 'references', check_function = False):

        import regex as re        

        #identificador de final de texto        
        if section_marker.lower() == 'references':
            pattern1 = r'(' +\
                        r'Literature\s[Cc]ited\s[A-Z0-9]|References?\s[A-Z0-9]|REFERENCES?\s[A-Z0-9]|Bibliography\s[A-Z0-9]|BIBLIOGRAPHY\s[A-Z0-9]' +\
                        r'|Acknowledge?ments?\s[A-Z0-9]|ACKNOWLEDGE?MENTS?\s[A-Z0-9]' +\
                        r'|Supplementary\s([Ii]nformation|Info|[Dd]ata|[Mm]aterials?)\s[A-Z0-9]' +\
                        r'|SUPPLEMENTARY\s(INFORMATION|INFO|DATA|MATERIALS?)\s[A-Z0-9]' +\
                        r'|CONFLICT\sOF\sINTEREST\s[A-Z0-9]|Conflict\sof\s[Ii]nterest\s[A-Z0-9]' +\
                        r')'
        elif section_marker.lower() == 'conclusions':
            pattern1 = r'(' +\
                        r'CONCLUSIONS?\s[A-Z0-9]|Conclusions?\s[A-Z0-9]' +\
                        r'|Concluding\s[Rr]emarks?\s[A-Z0-9]' +\
                        r')'            
            
        #identificador citações de referências (bibliografia) padrão: Zhang X-Y.,
        pattern2 = r'[A-Z][a-z\-\‐\–\−\—\―\─\‑\‒\^~]+(\s|\,|and)+[A-Z\-\‐\–\−\—\―\─\‑\‒\s\.\,]+[\,\.]'
        #identificador citações de referências (bibliografia) padrão: X-Y. Zhang,
        pattern3 = r'[A-Z\-\‐\–\−\—\―\─\‑\‒\s\.\,]+[A-Z][a-z\-\‐\–\−\—\―\─\‑\‒\^~]+[\s\,]'    
        
        #análise do número de tokens por sentença
        token_number_list = []
        sent_n_tokens_dic = {}

        #------------------------------
        #encontrando os indexes das referências
        reference_section_index_list = []
        found_ref_section_delimiter = False
        found_ref_section_counter = 0
    
        #varrendo as sentenças
        i_counter = 0            
        for i in sentDF.index:           
            
            #abrindo a sentença
            sent = sentDF.loc[i, 'Sentence']
            i_counter += 1
            
            #coletando o número de tokens                
            n_tokens = len(sent.split())
            token_number_list.append(n_tokens)
            sent_n_tokens_dic[i] = n_tokens

            #será iniciada a procura por referência após 1/5 de sentenças varridas
            mark_index = ( sentDF.index.values[-1] - sentDF.index.values[0] ) * (1/5)

            if i_counter > mark_index:                    
                
                #print('looking for references: ', i,' ; ', sent)
                #time.sleep(1)
                                
                #caso o delimitador de seção já tenha sido encontrado
                if found_ref_section_delimiter is True:
                    if found_ref_section_counter <= ref_section_index_range:
                        
                        #encontrar a citação dos nomes do tipo Zhang X-Y.,
                        if re.match(pattern2, sent):
                            found_ref_section_counter = 0
                            if i not in reference_section_index_list:
                                reference_section_index_list.append(i)
                            if check_function is True:
                                #print('Ref found: ', sent)
                                pass
        
                        #encontrar a citação dos nomes do tipo X-Y. Zhang,
                        elif re.match(pattern3, sent):
                            found_ref_section_counter = 0
                            if i not in reference_section_index_list:
                                reference_section_index_list.append(i)
                            if check_function is True:
                                #print('Ref found: ', sent)
                                pass
                        
                        #encontrar várias citações do tipo Zhang X-Y. na sentença
                        elif len( re.findall(pattern2, sent) ) >= 3:
                            found_ref_section_counter = 0
                            if i not in reference_section_index_list:
                                reference_section_index_list.append(i)
                            if check_function is True:
                                #print('Ref found: ', sent)
                                pass
                            
                        #se não encontrou nenhum padrão, soma-se no contador
                        else:
                            found_ref_section_counter += 1
                    
                    #caso não tenha sido encontrado nenhum patterns de referencias entre 20 sentenças
                    else:
                        #caso o valor minimo de patterns de referências ainda não tenha sido identificadas anteriormente, 
                        #zera-se os indicador de seção e a lista
                        if len(reference_section_index_list) < min_ref_to_find_for_section:
                            found_ref_section_delimiter = False
                            reference_section_index_list = []
                            found_ref_section_counter = 0                            
                        #caso já tenha sido encontrado o valor mínimo de patterns faz-se o break
                        else:
                            break
                
                #primeira etapa: encontrar o termo References ou Bibliography                    
                else:
                    if re.match(pattern1, sent):
                        #print(re.match(pattern1, sent))
                        found_ref_section_delimiter = True
                        reference_section_index_list.append(i)
                        if check_function is True:
                            print('> Ref section begin: ', i, ' ; ', sent)
                            
        return found_ref_section_delimiter, reference_section_index_list, token_number_list, sent_n_tokens_dic
