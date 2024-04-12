#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import time
import json
import regex as re
import numpy as np
from scipy import sparse
import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from PDF import PDFTEXT
from XML import XMLTEXT
from TXT import TXTTEXT

from FUNCTIONS import get_filenames_from_folder
from FUNCTIONS import save_dic_to_json
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import update_log
from FUNCTIONS import load_log_info
from FUNCTIONS import create_h5_matrix
from FUNCTIONS import load_h5_matrix
from FUNCTIONS import get_tag_name
from FUNCTIONS import get_file_batch_index_list

from functions_TEXTS import save_sentence_list_to_csv
from functions_TEXTS import save_text_to_TXT
from functions_TEXTS import save_full_text_to_json
from functions_TEXTS import save_sentence_dic_to_csv
from functions_TOKENS import get_tokens_from_sent
from functions_TOKENS import from_token_indexed_csv_to_dict



class text_objects(object):
    
    def __init__(self, diretorio = None, article_file_type = ''):
        
        print('\n( Class: text_objects )')
        
        self.diretorio = diretorio
        self.article_file_type = article_file_type.lower()
        self.stopwords_list = stopwords.words('english')

        #checando a existência das pastas
        if not os.path.exists(self.diretorio + '/Outputs/sents_raw'):
            os.makedirs(self.diretorio + '/Outputs/sents_raw')
        
        if not os.path.exists(self.diretorio + '/Outputs/sents_filtered'):
            os.makedirs(self.diretorio + '/Outputs/sents_filtered')
        
        if not os.path.exists(self.diretorio + '/Outputs/log'):
            os.makedirs(self.diretorio + '/Outputs/log')

        #caso os arquivos sejam pdf
        if self.article_file_type == 'pdf':
            from tika import tika
            tika.TikaClientOnly = True
            tika.TikaStartupMaxRetry = 0
            tika.startServer(diretorio + '/Modules/tika-server-1.24.1.jar')
    
    
    def get_text_objects(self, save_TXT = False, save_JSON = False):

        print('\n( Function: get_text_objects )')

        #Abrindo o DB_ID dataframe
        DB_ID_df = pd.read_csv(self.diretorio + '/Outputs/DB_ID.csv', index_col = 0)

        #montando a lista de artigos da coleção
        DB_documents = get_filenames_from_folder(self.diretorio + '/DB', file_type = self.article_file_type)
        
        #resumindo o processamento
        last_file_processed = load_log_info(log_name = 'last_processed_file', logpath = self.diretorio + '/Outputs/log/raw_sents.json')
        last_file_index = DB_documents.index(last_file_processed) + 1 if last_file_processed is not None else 0
        print('Last file processed: ', last_file_processed)
        
        sentence_counter = load_log_info(log_name = 'sentence_counter', logpath = self.diretorio + '/Outputs/log/raw_sents.json')
        sentence_counter = sentence_counter if sentence_counter is not None else 0

        #caso o último documento processado seja igual ao último arquivo da pasta /sents_raw
        if DB_documents.index( DB_documents[-1] ) == last_file_index:
            print(f'Todos os documentos ({len(DB_documents)}) já foram processados')
            print('> Abortando função: text_objects.get_text_objects')
            return
        
        #processando os documentos
        for filename in DB_documents[ last_file_index : ]:
            
            if self.article_file_type == 'pdf':
            
                PDF = PDFTEXT(filename, 
                              folder_name = 'DB', 
                              language = 'english',
                              diretorio = self.diretorio)
                
                PDF.process_document(apply_filters = True)
            
                #salvando os textos em formato JSON
                if (save_JSON is True):
                    #salvando o texto raw
                    #save_full_text_to_json(PDF.raw_text, f'ATC{file_N_conv}', 'Texts_raw', raw_string = False)
                    #salvando o texto filtrado
                    save_full_text_to_json(PDF.filtered_text, filename, folder = 'JSON_filtered_texts', raw_string = False, diretorio = self.diretorio)
                                                        
                #salvando as sentenças
                sentence_counter = save_sentence_list_to_csv(PDF.sentences, 
                                                             filename, 
                                                             sentence_counter, 
                                                             folder = 'sents_raw', 
                                                             diretorio = self.diretorio)
                print('Document (sentence) counter = ', sentence_counter)
                
                #salvando os textos em TXT (formato legível)
                if save_TXT is True:
                    #salvando o texto extraido em TXT
                    save_text_to_TXT(repr(PDF.raw_text), filename, folder = 'TXT_rawtexts', diretorio = self.diretorio)
                    
                #salvando o file_index
                update_log(log_names = ['last_processed_file', 'sentence_counter'], entries = [filename, sentence_counter], logpath = self.diretorio + '/Outputs/log/raw_sents.json')

            
            elif self.article_file_type == 'xml':
                
                XML = XMLTEXT(filename, 
                              folder_name = 'DB', 
                              language = 'english',
                              diretorio = self.diretorio)
                
                XML.process_document(apply_filters = True)
            
                #salvando os textos em formato JSON
                if (save_JSON is True):
                    #salvando o texto raw
                    #save_full_text_to_json(PDF.raw_text, f'ATC{file_N_conv}', 'Texts_raw', raw_string = False)
                    #salvando o texto filtrado
                    save_full_text_to_json(XML.filtered_text, filename, folder = 'JSON_filtered_texts', raw_string = False, diretorio = self.diretorio)
                                                        
                #salvando as sentenças
                sentence_counter = save_sentence_list_to_csv(XML.sentences, 
                                                             filename, 
                                                             sentence_counter, 
                                                             folder = 'sents_raw', 
                                                             diretorio = self.diretorio)
                print('Document (sentence) counter = ', sentence_counter)
                
                #salvando os textos em TXT (formato legível)
                if save_TXT is True:
                    #salvando o texto extraido em TXT
                    save_text_to_TXT(repr(XML.raw_text), filename, folder = 'TXT_rawtexts', diretorio = self.diretorio)
                    
                #salvando o file_index
                update_log(log_names = ['last_processed_file', 'sentence_counter'], entries = [filename, sentence_counter], logpath = self.diretorio + '/Outputs/log/raw_sents.json')


            elif self.article_file_type == 'webscience_csv_report':

                TXT = TXTTEXT(filename,
                                folder_name = 'DB', 
                                language = 'english',
                                diretorio = self.diretorio)
                
                TXT.process_document(apply_filters = True, DB_ID_df = DB_ID_df)
            
                #salvando os textos em formato JSON
                if (save_JSON is True):
                    #salvando o texto raw
                    #save_full_text_to_json(PDF.raw_text, f'ATC{file_N_conv}', 'Texts_raw', raw_string = False)
                    #salvando o texto filtrado
                    save_full_text_to_json(TXT.filtered_text, filename, folder = 'JSON_filtered_texts', raw_string = False, diretorio = self.diretorio)
                                                        
                #salvando as sentenças
                sentence_counter = save_sentence_list_to_csv(TXT.sentences, 
                                                                filename, 
                                                                sentence_counter, 
                                                                folder = 'sents_raw', 
                                                                diretorio = self.diretorio)
                print('Document (sentence) counter = ', sentence_counter)
                
                #salvando os textos em TXT (formato legível)
                if save_TXT is True:
                    #salvando o texto extraido em TXT
                    save_text_to_TXT(repr(TXT.raw_text), filename, folder = 'TXT_rawtexts', diretorio = self.diretorio)
                    
                #salvando o file_index
                update_log(log_names = ['last_processed_file', 'sentence_counter'], entries = [filename, sentence_counter], logpath = self.diretorio + '/Outputs/log/raw_sents.json')

        

    def convert_ner_dic_to_nGrams_df(self):
        
        print('\n( Function: convert_ner_dic_to_nGrams_df )')

        #criando a pasta /Outputs/ngrams/semantic/
        if not os.path.exists(self.diretorio + f'/Outputs/ngrams/semantic/'):
            os.makedirs(self.diretorio + f'/Outputs/ngrams/semantic/')

        #abrindo o dicionário com os ner rules
        with open(self.diretorio + '/Inputs/ner_rules.json', 'r') as ner_json_file:
            ner_dic = json.load(ner_json_file)
            ner_json_file.close()
        
        #varrendo as entidades ner
        for ner_entity in ner_dic.keys():

            #os termos associados a essa entidade serão o index da df
            ner_terms = ner_dic[ner_entity]['terms']
            nxgram_ner_terms_DF = pd.DataFrame(columns=['Sem_App_Counter'], index=ner_terms)
            nxgram_ner_terms_DF.index.name = 'index'

            #setando as condições inicias da DF
            for term in nxgram_ner_terms_DF.index:
                nxgram_ner_terms_DF.loc[term] = 0
            nxgram_ner_terms_DF.to_csv(self.diretorio + f'/Outputs/ngrams/semantic/nxgram_{ner_entity}.csv')
            print(f'Exportando a DF ~/Outputs/ngrams/semantic/nxgram_{ner_entity}.csv')
           

    def filter_sentences(self, cut_quantile_low: int = 0, cut_quantile_high: int = 1, min_sent_tokens = 5, check_function = False):
        
        print('\n( Function: filter_sentences )')
        
        #carregando o dicionário com as estatísticas das sentenças
        if not os.path.exists(self.diretorio + '/Outputs/log/stats_sents_tokens_len_not_filtered.json'):
            print('ERRO! Arquivo "~/Outputs/log/stats_sents_tokens_len_not_filtered.json" não encontrado. Rodar a função "find_sent_stats".')
            print('> Abortando função: text_objects.filter_sentences')
            return
        else:
            sent_token_stats = load_dic_from_json(self.diretorio + '/Outputs/log/stats_sents_tokens_len_not_filtered.json')
            
            #checando se os valores de entrada do cut_quantile_low e cut_quantile_high são compátiveis
            cond1 = str(cut_quantile_low) in ('0', '0.05', '0.1', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.75', '0.8', '0.9')
            cond2 = str(cut_quantile_high) in ('0.1', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.75', '0.8', '0.9', '0.95', '1')
            if False in (cond1, cond2):
                print('ERRO! Introduzir valores de "cut_quantile_low" e "cut_quantile_high" compatíveis com o que foi encontrado com a função "find_sent_stats".')
                print('Valores compatíveis (inserir com string):')
                for val in ('0', '0.05', '0.1', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.75', '0.8', '0.9', '1'):
                    print(val)
                return

        extracted_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/sents_raw', file_type = 'csv')

        #resumindo o processamento
        last_file_processed = load_log_info(log_name = 'last_processed_file', logpath = self.diretorio + '/Outputs/log/filtered_sents.json')
        last_file_index = extracted_DB_documents.index(last_file_processed) + 1 if last_file_processed is not None else 0
        print('Last file processed: ', last_file_processed)
        
        sentence_counter = load_log_info(log_name = 'sentence_counter', logpath = self.diretorio + '/Outputs/log/filtered_sents.json')
        sentence_counter = sentence_counter if sentence_counter is not None else 0

        #caso o último documento processado seja igual ao último arquivo da pasta /sents_filtered
        if last_file_processed ==  extracted_DB_documents[-1]:
            print(f'Todos os documentos ({len(extracted_DB_documents)}) já foram filtrados')
            print('> Abortando função: text_objects.filter_sentences')
            return
        

        #caso o arquivo seja PDF
        if self.article_file_type == 'pdf':
            
            #contadores para calcular a eficácia da seleção de seção de referências
            total_file_counter = 0
            counter_references_marker = 0
            counter_conclusions_marker = 0
            
            ref_found_counter = 0
            for filename in extracted_DB_documents[ last_file_index : ]:
                
                print('\nProcessing ', filename, '...')
                total_file_counter += 1
                            
                #abrinado o DF do documento
                sentDF = pd.read_csv(self.diretorio + f'/Outputs/sents_raw/{filename}.csv', index_col = 0)

                #check se encontrou a seção de referências
                found_ref_delimiter = False
                
                #na primeira tentativa procura-se pelo marcador de referências
                found_ref_delimiter, refs_section_index = self.looking_for_ref_section(sentDF, section_marker = 'references', check_function = check_function)
                
                #caso as referências tenham sido encontradas na primeira tentativa
                if found_ref_delimiter is True:
                    counter_references_marker += 1
                    
                #na segunda tentativa procura-se pelo marcador de conclusions            
                else:
                    found_ref_delimiter, refs_section_index  = self.looking_for_ref_section(sentDF, section_marker = 'conclusions', check_function = check_function)
                                                                
                    #caso as referências tenham sido encontradas na primeira tentativa
                    if found_ref_delimiter is True:
                        counter_conclusions_marker += 1
                        
                #documento cujas referências foram encontradas
                if found_ref_delimiter is True:
                    
                    #coletando somente as sentenças filtradas
                    sents_to_get = []
                    for i in sentDF.index:

                        sent =  sentDF.loc[ i, 'Sentence' ]
                        
                        #condições
                        n_sent_tokens = len(get_tokens_from_sent(sent))
                        cond1 = sent_token_stats["with_SW_True"][f'{cut_quantile_low}_quantile'] <= n_sent_tokens <= sent_token_stats["with_SW_True"][f'{cut_quantile_high}_quantile']
                        cond2 = n_sent_tokens >= min_sent_tokens
                        cond3 = i < refs_section_index
                        
                        if False not in (cond1, cond2, cond3):
                            sents_to_get.append(sentDF.loc[i, 'Sentence'])
                
                    #salvando as sents filtradas
                    sentence_counter = save_sentence_list_to_csv(sents_to_get,
                                                                 filename, 
                                                                 sentence_counter, 
                                                                 folder = 'sents_filtered', 
                                                                 diretorio=self.diretorio)

                    #salvando o file_index
                    update_log(log_names = ['last_processed_file', 'sentence_counter'], entry = [filename, sentence_counter], logpath = self.diretorio + '/Outputs/log/filtered_sents.json')

                    #contador de documentos com referências encontradas
                    ref_found_counter += 1
                    print('Total de documentos cujas referências foram encontradas: ', round(ref_found_counter/total_file_counter, 3) * 100, ' % (Total: ', total_file_counter,' )', )
                    print('Número de marcadores - "references": ', counter_references_marker, ' ; "conclusions" : ', counter_conclusions_marker)
                
                else:
                    print('Atenção! As referências não foram encontradas para o arquivo: ', filename)


        #caso o arquivo seja XML
        elif self.article_file_type == 'xml':
            
            #contadores para calcular a eficácia da seleção de seção de referências            
            for filename in extracted_DB_documents[ last_file_index : ]:
                
                print('\nProcessing ', filename, '...')
                            
                #abrinado o DF do documento
                sentDF = pd.read_csv(self.diretorio + f'/Outputs/sents_raw/{filename}.csv', index_col = 0)
                
                #coletando somente as sentenças filtradas
                sents_to_get = {}
                section_name = 'None'
                l_sent_counter = 0
                
                for i in sentDF.index:

                    sent =  sentDF.loc[ i, 'Sentence' ]

                    #procurando o início da seção
                    match_begin = re.search( r'(?<=The BEGIN of the section is there \(separated from the XML file\) ).+(?=\.)', sent )
                    
                    #procurando o fim da seção
                    match_end = re.search( r'The END of the section is there \(separated from the XML file\)\.', sent )
                    
                    if match_begin is not None:
                        section_name = re.match(r'(introduction|methodology|results-discussion|results|discussion|conclusion)', match_begin.group()).group()
                    
                    #condições
                    n_sent_tokens = len(get_tokens_from_sent(sent))
                    cond1 = match_begin is None
                    cond2 = match_end is None
                    cond3 = sent_token_stats["with_SW_True"][f'{cut_quantile_low}_quantile'] <= n_sent_tokens <= sent_token_stats["with_SW_True"][f'{cut_quantile_high}_quantile']
                    cond4 = n_sent_tokens >= min_sent_tokens
                    
                    #coletando a sentença
                    if False not in (cond1, cond2, cond3, cond4):

                        #colocando um index para a sentença
                        sents_to_get[ sentence_counter + l_sent_counter ] = {}
                        sents_to_get[ sentence_counter + l_sent_counter ][ 'sent' ] = sentDF.loc[i, 'Sentence']
                        sents_to_get[ sentence_counter + l_sent_counter ][ 'section' ] = section_name
                        l_sent_counter += 1

                    else:
                        print('> sent_ecluida', sent)
                        #time.sleep(0.1)
                        pass
                
                #salvando as sents filtradas
                sentence_counter = save_sentence_dic_to_csv(sents_to_get,
                                                            filename,
                                                            folder = 'sents_filtered',
                                                            diretorio=self.diretorio)

                #salvando o file_index
                update_log(log_names = ['last_processed_file', 'sentence_counter'], entry = [filename, sentence_counter], logpath = self.diretorio + '/Outputs/log/filtered_sents.json')


        #caso o arquivo seja extraido do webscience_csv_report
        elif self.article_file_type == 'webscience_csv_report':
            
            #contadores para calcular a eficácia da seleção de seção de referências            
            for filename in extracted_DB_documents[ last_file_index : ]:
                
                print('\nProcessing ', filename, '...')

                #abrinado o DF do documento
                sentDF = pd.read_csv(self.diretorio + f'/Outputs/sents_raw/{filename}.csv', index_col = 0)

                #caso haja sentenças no dataframe
                if len(sentDF.index) > 0:

                    #varrendo as sentenças     
                    sents_to_get = {}
                    sent_counter = 0
                    for i in sentDF.index:
                    #colocando um index para a sentença
                        sents_to_get[ sentence_counter + sent_counter ] = {}
                        sents_to_get[ sentence_counter + sent_counter ][ 'sent' ] = sentDF.loc[i, 'Sentence']
                        sents_to_get[ sentence_counter + sent_counter ][ 'section' ] = 'abstract'
                        sent_counter += 1

                    #salvando as sents filtradas
                    sentence_counter = save_sentence_dic_to_csv(sents_to_get,
                                                                filename,
                                                                folder = 'sents_filtered',
                                                                diretorio=self.diretorio)

                    #salvando o file_index
                    update_log(log_names = ['last_processed_file', 'sentence_counter'], entries = [filename, sentence_counter], logpath = self.diretorio + '/Outputs/log/filtered_sents.json')
                
                else:
                    print('Erro ao filtrar o arquivo: ', filename, '. Não há sentenças no arquivo csv.')
                    time.sleep(5)



    def find_text_sections(self):

        print('\n( Function: find_text_sections )')
        
        if not os.path.exists(self.diretorio + '/Outputs/sections'):
            os.makedirs(self.diretorio + '/Outputs/sections')

        extracted_DB_documents = get_filenames_from_folder(self.diretorio + '/Outputs/sents_raw', file_type = 'csv')

        #resumindo o processamento
        last_file_processed = load_log_info(log_name = 'last_processed_file', logpath = self.diretorio + '/Outputs/log/section_filter.json')
        last_file_index = extracted_DB_documents.index(last_file_processed) + 1 if last_file_processed is not None else 0


        #caso o arquivo seja PDF
        if self.article_file_type == 'pdf':
            
            #inserir os separadores de seção
            sections_names=[r'([0-9\s\.]*(Materials|MATERIALS)\s(and|And|AND)\s([Mm]ethods|METHODS)\s*([A-Z]|[0-9]\.)|(Experimental|EXPERIMENTAL)\s*[A-Z0-9][\.\sa-z])', 
                            r'([0-9\s\.]*(Results?|RESULTS?)\s(and|And|AND)\s([Dd]iscussion|DISCUSSION)\s*([A-Z]|[0-9]\.)|(Results?|RESULTS?)\s*[A-Z0-9][\.\sa-z])']

            #dividindo os dois nomes de seção
            begin_section_pattern = sections_names[0]
            end_section_pattern = sections_names[1]
            print('Section begin pattern: ', begin_section_pattern)
            print('Section end pattern: ', end_section_pattern)
                                                                                
            for filename in extracted_DB_documents[ last_file_index : ]:

                print('Processing ', filename, '...')
                sentDF = pd.read_csv(self.diretorio + f'/Outputs/sents_raw/{filename}.csv', index_col = 0)
                
                #caso exista a sentença filtrada
                try:
                    #varrendo as sentenças filtradas
                    sentDF_filtered = pd.read_csv(self.diretorio + f'/Outputs/sents_filtered/{filename}.csv', index_col = 0)
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
                    
                    #associando cada sentença a um número contador
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
                        
                        sentDF_filtered_copy = sentDF_filtered.copy()
                        
                        #criando os targets para a seção de introdução
                        first_part = [1] * begin_index_filtered
                        sec_part = [0] * (end_index_filtered - begin_index_filtered)
                        third_part = [0] * (index_len - end_index_filtered)
                        concat_list = first_part + sec_part + third_part
                        sentDF_filtered_copy['introduction'] = concat_list

                        #criando os targets para a seção de metodologia
                        first_part = [0] * begin_index_filtered
                        sec_part = [1] * (end_index_filtered - begin_index_filtered)
                        third_part = [0] * (index_len - end_index_filtered)
                        concat_list = first_part + sec_part + third_part
                        sentDF_filtered_copy['methodology'] = concat_list

                        #criando os targets para a seção de resultados/discussão
                        first_part = [0] * begin_index_filtered
                        sec_part = [0] * (end_index_filtered - begin_index_filtered)
                        third_part = [1] * (index_len - end_index_filtered)
                        concat_list = first_part + sec_part + third_part
                        sentDF_filtered_copy['results'] = concat_list

                        sentDF_filtered_copy.to_csv(self.diretorio + f'/Outputs/sections/{filename}.csv')
                        print(f'Section DF exportada em ~/Outputs/sections/{filename}.csv')

                #salvando o file_index
                update_log(log_names = 'last_processed_file', entries = [filename], logpath = self.diretorio + '/Outputs/log/section_filter.json')


        #caso o arquivo seja XML
        elif self.article_file_type == 'xml':
                                                                                
            for filename in extracted_DB_documents[ last_file_index : ]:
                
                print('Processing ', filename, '...')
                sentDF_filtered = pd.read_csv(self.diretorio + f'/Outputs/sents_filtered/{filename}.csv', index_col = 0)
                sentDF_filtered_copy = sentDF_filtered[['article Number', 'Sentence']].copy()
                sent_len = len(sentDF_filtered_copy.index)
                get_document = True
                
                for section_name in np.unique(sentDF_filtered['Section'].values):
                    #print('varrendo secao: ', section_name)
                    #definindo a função para substituição
                    def sub(input, cat=section_name):
                        if input == cat:
                            return 1
                        else:
                            return 0

                    sentDF_filtered_copy[section_name] = sentDF_filtered['Section'].apply(sub).values

                    #contanto o número de sentença para cada seção
                    section_sent_len = sentDF_filtered_copy[section_name].values.sum()

                    if section_sent_len > int(0.5 * sent_len) and section_name != 'results-discussion':
                        print(f'ATENTION! Section {section_name} has {section_sent_len} of {sent_len}.')
                        print('Not salving this article.')
                        get_document = False
                        #time.sleep(1)
                
                if get_document is True:
                    sentDF_filtered_copy.to_csv(self.diretorio + f'/Outputs/sections/{filename}.csv')
                    print(f'Section DF exportada em ~/Outputs/sections/{filename}.csv')

                #salvando o file_index
                update_log(log_names = 'last_processed_file', entries = [filename], logpath = self.diretorio + '/Outputs/log/section_filter.json')



    def get_ngrams_appearance(self):
        
        print('\n( Function: get_ngrams_appereance )')

        #checando a existência das pastas
        if not os.path.exists(self.diretorio + '/Outputs/ngrams/appearence'):
            os.makedirs(self.diretorio + '/Outputs/ngrams/appearence')

        #carregando a lista de arquivos processados
        filtered_sents_filenames = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')

        #resumindo o processamento
        last_file_processed = load_log_info(log_name = 'last_processed_file', logpath = self.diretorio + '/Outputs/log/ngrams_appearence.json')
        last_file_index = filtered_sents_filenames.index(last_file_processed) + 1 if last_file_processed is not None else 0
        print('Last file processed: ', last_file_processed)

        #checando se os arquivos já tiveram os tokens contados
        if os.path.exists(self.diretorio + '/Outputs/ngrams/appearence/tokens_appereance_counts.csv'):
            token_dic = from_token_indexed_csv_to_dict(self.diretorio + '/Outputs/ngrams/appearence/tokens_appereance_counts.csv')
        else:
            #definindo os dicionários para contar os findings de tokens
            token_dic = {}
        
        #caso o último arquivo processado seja o ultimo da pasta
        if filtered_sents_filenames[ -1 ] == last_file_processed:
            print(f'Os tokens de todos os documentos ({len(filtered_sents_filenames)}) já foram contados')
            print('> Abortando função: text_objects.get_Ngrams_appereance')
            return
        
        print('Finding 2grams CSV...')
        #checando se há o DF de 2grams
        if os.path.exists(self.diretorio + '/Outputs/ngrams/appearence/2grams_appereance_counts.csv'):
            #carregando o DF com os 2grams
            n2grams_dic = from_token_indexed_csv_to_dict(self.diretorio + '/Outputs/ngrams/appearence/2grams_appereance_counts.csv')
            print('DataFrame de 2grams encontrado...')            
        else:
            #criando um dicionário de 2grams
            n2grams_dic = {}       
            print('Criando DF de 2grams...')
        
        #varrendo os documentos .csv com as sentenças
        counter_to_save = 0
        for filename in filtered_sents_filenames[ last_file_index : ]:
            
            #lista para coletar os tokens presentes no arquivo do artigo
            tokens_got_in_article = []
            n2grams_got_in_article = []

            print(f'\nProcessando CSV (~/Outputs/sents_filtered/{filename}.csv)')
            
            #abrindo o csv com as sentenças do artigo
            sentDF = pd.read_csv(self.diretorio + '/Outputs/sents_filtered/' + f'{filename}.csv', index_col = 0)
            
            #analisando cada sentença
            for index in sentDF.index:

                #lista para coletar os tokens presentes na sentença
                tokens_got_in_sent = []
                n2grams_got_in_sent = []
                
                sent = sentDF.loc[index, 'Sentence']
                
                #pegando os tokens da sentença sem stopwords
                sent_tokens = get_tokens_from_sent(sent, stopwords_list_to_remove = self.stopwords_list, spacy_tokenizer = nlp)
                    
                #varrendo os tokens
                for token in sent_tokens:
                        
                    try:
                        #print('Econtrado o token ( ', token, ' ) na sentença.')
                        token_dic[token]['total'] += 1
                        #print('Token ' + token + ' encontrado.')

                        #checando a presença do token na sentença
                        if (token_dic[token]['check_sent_pres'] is False):
                            token_dic[token]['sent'] += 1
                            token_dic[token]['check_sent_pres'] = True
                            tokens_got_in_sent.append(token)
                        
                        #checando a presença do token no artigo
                        if (token_dic[token]['check_article_pres'] is False):
                            token_dic[token]['article'] += 1
                            token_dic[token]['check_article_pres'] = True
                            tokens_got_in_article.append(token)
                    
                    except KeyError:
                        token_dic[token] = {}
                        token_dic[token]['total'] = 1
                        token_dic[token]['sent'] = 1
                        token_dic[token]['article'] = 1
                        token_dic[token]['check_article_pres'] = True
                        token_dic[token]['check_sent_pres'] = True
                        tokens_got_in_sent.append(token)
                        tokens_got_in_article.append(token)

                #varrendo os tokens para 2grams
                for token_index in range(len(sent_tokens)):
                    try:
                        token_0 = sent_tokens[token_index]
                        token_1 = sent_tokens[token_index + 1]
                            
                        bigram = token_0 + '_' + token_1
                        #print(token_0, token_1)
                        
                        #varrendo os tokens da sentença
                        try:                            
                            #print('Econtrado o bigram ( ', bigram, ' ) na sentença.')
                            n2grams_dic[bigram]['total'] += 1
                            #print('Bigram ' + bigram + ' encontrado.')
        
                            if (n2grams_dic[bigram]['check_sent_pres'] is False):
                                #checando a presença do bigram no artigo
                                n2grams_dic[bigram]['sent'] += 1
                                n2grams_dic[bigram]['check_sent_pres'] = True
                                n2grams_got_in_sent.append(bigram)
                                
                            if (n2grams_dic[bigram]['check_article_pres'] is False):
                                #checando a presença do bigram no artigo
                                n2grams_dic[bigram]['article'] += 1
                                n2grams_dic[bigram]['check_article_pres'] = True
                                n2grams_got_in_article.append(bigram)
                            
                        except KeyError:
                            n2grams_dic[bigram] =  {}
                            n2grams_dic[bigram]['total'] = 1
                            n2grams_dic[bigram]['sent'] = 1
                            n2grams_dic[bigram]['article'] = 1
                            n2grams_dic[bigram]['check_article_pres'] = True
                            n2grams_dic[bigram]['check_sent_pres'] = True
                            n2grams_got_in_sent.append(bigram)
                            n2grams_got_in_article.append(bigram)
                        
                    except IndexError:
                        continue   

                #limpando os check de token e Ngrams nas sentenças
                for token in tokens_got_in_sent:
                    token_dic[token]['check_sent_pres'] = False
                
                for bigram in n2grams_got_in_sent:
                    n2grams_dic[bigram]['check_sent_pres'] = False

                del sent
                del sent_tokens
                del tokens_got_in_sent
                del n2grams_got_in_sent

            #limpando os check de token e Ngrams nos artigos
            for token in tokens_got_in_article:
                token_dic[token]['check_article_pres'] = False
            for bigram in n2grams_got_in_article:
                n2grams_dic[bigram]['check_article_pres'] = False
                
            print('Tokens counter: ', len(token_dic.keys()))
            print('2gram counter: ', len(n2grams_dic.keys()))
            counter_to_save += 1
            
            #salvando a cada 30 artigos ou quando for o último
            if counter_to_save % 10 == 0 or filename == filtered_sents_filenames[-1]:

                print('\nSalvando os arquivos...\n')
                
                #salvando a contagem de tokens
                token_DF = pd.DataFrame.from_dict(token_dic, orient='index')
                token_DF.index.name = 'index'
                token_DF.sort_values(by=['total'], ascending=False, inplace=True)
                token_DF.to_csv(self.diretorio + '/Outputs/ngrams/appearence/tokens_appereance_counts.csv')
                
                n2gram_DF = pd.DataFrame.from_dict(n2grams_dic, orient='index')
                n2gram_DF.index.name = 'index'
                n2gram_DF.sort_values(by=['total'], ascending=False, inplace=True)
                n2gram_DF.to_csv(self.diretorio + '/Outputs/ngrams/appearence/2grams_appereance_counts.csv')
                #print(n2gram_DF)
                
                #salvando o número do último arquivo processado
                update_log(log_names = ['last_processed_file'], entries = [filename], logpath = self.diretorio + '/Outputs/log/ngrams_appearence.json')

            del tokens_got_in_article
            del n2grams_got_in_article
            del sentDF
            
        del token_dic
        del n2grams_dic
        del token_DF
        del n2gram_DF



    def filter_2grams(self, n2gram_mim_delta_val = 0.5):
        
        print('\n( Function: filter_2grams )')

        print('Abrindo a DF com a contagem de tokens...')
        #abrindo a contagem de tokens e a quantidade total de documentos
        token_DF = pd.read_csv(self.diretorio + '/Outputs/ngrams/appearence/tokens_appereance_counts.csv', index_col=0)
        #token_DF.dropna(inplace=True)

        print('Abrindo a DF com a contagem de bigrams...')
        n2grams_DF = pd.read_csv(self.diretorio + '/Outputs/ngrams/appearence/2grams_appereance_counts.csv', index_col=0)
        #n2grams_DF.dropna(inplace=True)

        n2gram_scores_dic = {}
        print('Existem ', n2grams_DF.shape[0], ' bigrams para processar.')

        #checando a existência das pastas
        if not os.path.exists(self.diretorio + '/Outputs/ngrams/filtered_scores'):
            os.makedirs(self.diretorio + '/Outputs/ngrams/filtered_scores')
        
        if not os.path.exists(self.diretorio + '/Outputs/ngrams/filtered_scores/n2grams_scores.csv'):
            print('Calculando os scores dos bigrams...')
            counter = 1
            for bigram in n2grams_DF.index:
                
                try:
                    #encontrando os tokens a partir dos bigrams
                    match = re.search(r'([\w\-\+]+)_([\w\-\+]+)', bigram)
                    try:
                        token_1 = match.group(1)
                        token_2 = match.group(2)
                    #caso haja caracteres especiais nos bigramas (ex: ag$_cl@)
                    except AttributeError:
                        continue
                    
                    deltas = []
                    for token in (token_1, token_2):
                        delta = ( ( token_DF.loc[ token , 'total'] - n2grams_DF.loc[ bigram , 'total' ] ) / token_DF.loc[ token , 'total'] )
                        deltas.append(delta)
                    
                    if min(deltas) <= n2gram_mim_delta_val:
                        n2gram_scores_dic[bigram] = {}
                        n2gram_scores_dic[bigram]['min_delta'] = min(deltas)
                        n2gram_scores_dic[bigram]['token_1'] = token_1
                        n2gram_scores_dic[bigram]['token_2'] = token_2
                        #n2gram_scores_dic[bigram]['Score'] = ( n2grams_DF.loc[bigram , 'total'] - threshold_2gram ) / ( token_DF.loc[token_0 , 'total'] * token_DF.loc[token_1 , 'total'] )
                    
                    #print('token_0: ', token_0, ' ; token_1: ', token_1)
                    #print(n2gram_scores_dic[bigram])
                    #time.sleep(0.1)
                    
                    if counter % 100000 == 0:
                        print('n2gram processed ', counter)
                    counter += 1
                
                except KeyError:
                    #print('Erro de reconhecimento do bigram: ', bigram, ' pelo regex. Ignorando...')
                    continue

            
            print('Salvando ~/Outputs/ngrams/filtered_scores/n2grams_scores.csv')
            n2gram_scores_DF = pd.DataFrame.from_dict(n2gram_scores_dic, orient='index')
            #n2gram_scores_series.sort_values(ascending=False, inplace=True)                
            n2gram_scores_DF.to_csv(self.diretorio + '/Outputs/ngrams/filtered_scores/n2grams_scores.csv')
    
        else:
            n2gram_scores_DF = pd.read_csv(self.diretorio + '/Outputs/ngrams/filtered_scores/n2grams_scores.csv', index_col = 0)


        print('Filtering n2gram DF...')
        n2gram_scores_DF.sort_values(by=['min_delta'], ascending=True, inplace=True)            
        n2grams_concat = pd.concat([n2grams_DF, n2gram_scores_DF], axis = 1)
        n2grams_concat.dropna(inplace = True)
        n2grams_concat.index.name = 'index'
        n2grams_concat.to_csv(self.diretorio + '/Outputs/ngrams/filtered_scores/n2grams_filtered.csv')
        print('Saving to ~/Outputs/ngrams/filtered_scores/n2grams_filtered.csv')
            
        del token_DF
        del n2grams_DF
        del n2gram_scores_dic
        del n2gram_scores_DF
        del n2grams_concat
    
                
        
    def find_idf(self, min_token_appereance_in_corpus = 4):
        
        print('\n( Function: find_IDF )')
        
        if os.path.exists(self.diretorio + f'/Outputs/tfidf/idf.csv'):
            print('O IDF já foi calculado.')
            print('> Abortando função: text_objects.find_IDF')
            return

        #criando a TFIDF
        if not os.path.exists(self.diretorio + '/Outputs/tfidf'):
            os.makedirs(self.diretorio + '/Outputs/tfidf')
            print('Criando a pasta ~/Outputs/tfidf')

        #carregando a lista de arquivos
        sent_file_list = sorted(os.listdir(self.diretorio + '/Outputs/sents_filtered')) #lista de arquivos
        print('Total de artigos encontrados: ', len(sent_file_list))
        #testar se há arquivos no diretório ~/Outputs/sents_filtered
        if len(sent_file_list) == 0:
            print('ERRO!')
            print('Não há arquivos no self.diretorio ~/Outputs/sents_filtered.')
            print('Usar a função "text_objects.get_text_objects()"')
            return

        #montando a lista de arquivos de sentenças extraídas
        n_articles = len( get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv') )
                
        #abrindo a contagem de ngrams e a quantidade total de documentos
        n1gram_count_appereance = pd.read_csv(self.diretorio + '/Outputs/ngrams/appearence/tokens_appereance_counts.csv')
        n1gram_count_appereance.dropna(inplace=True)
        n1gram_count_appereance.set_index('index', inplace=True)
        print('Shape n1gram DF: ', n1gram_count_appereance.shape)
            
        n_sentences = load_log_info(log_name = 'sentence_counter', logpath = self.diretorio + '/Outputs/log/filtered_sents.json')
        print('Calculando o IDF para: ', n_sentences, ' documentos (sentenças).')
                
        #calculando o IDF
        IDF = {}
        for token in n1gram_count_appereance.index:
            #só serão tomados tokens que aparecem mais de n vez
            if n1gram_count_appereance.loc[token, 'total'] >= min_token_appereance_in_corpus:
            
                #cálculo do IDF para sentenças e artigos
                IDF[token] = {}
                IDF[token]['idf_sent'] = round ( np.log( ( ( n_sentences + 1) / ( n1gram_count_appereance.loc[token, 'sent'] + 1 ) ) + 1 ) , 8 ) 
                IDF[token]['idf_article'] = round ( np.log( ( ( n_articles + 1) / ( n1gram_count_appereance.loc[token, 'article'] + 1) ) + 1 ) , 8 )

        #cálculo do TF normalizado pelo número total de tokens da coleção e número de artigos da coleção
        #OBS1: esse TF calculado abaixo será usado no SUBSAMPLING durante a determinação dos word vectors
        #OBS2: o TF para determinação das matrizes TF-IDF da coleção é normaliado pelo número de tokens de cada sentença
        for token in IDF.keys():
            IDF[token]['tf_token_norm'] = n1gram_count_appereance.loc[token, 'total'] / len(IDF.keys())

        #salvando o IDF
        IDF = pd.DataFrame.from_dict(IDF, orient='index')
        IDF.to_csv(self.diretorio + f'/Outputs/tfidf/idf.csv')
        print('Total de tokens no IDF DF: ', len(IDF.index), '')
        
        del n1gram_count_appereance
        del IDF
        

    
    def set_tfidf_log_and_sent_stats(self, file_batch_size = 1000):

        print('\n( Function: set_TFIDF_log_and_sent_stats )')

        #checando o LOG file
        if os.path.exists(self.diretorio + '/Outputs/log/tfidf_batches_log.json'):
            print('O TFIDF_LOG e sents_index já foram extraídos.')
            print('> Abortando função: text_objects.set_TFIDF_log_and_sent_stats')
            return

        sent_index_dic = {}
        log_batches = {}    
        
        #carregando a lista de arquivos processados
        print('Carregando os nomes dos arquivos .csv filtrados...')
        filtered_sents_filenames = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')
        
        #número de documentos
        n_documents = len(filtered_sents_filenames)
        
        #caso o valor de file_batch_size seja incorreto
        if file_batch_size > n_documents:
            print('ERRO!')
            print('O valor inserido para o "file_batch_size" é maior que o número total de arquivos em ~/Outputs/sents_filtered')
            return
        
        #criando os batches de arquivos
        batch_indexes = get_file_batch_index_list(n_documents, file_batch_size)
        
        print('Getting LOG and stats (TFIDF_log.json; sents_index.csv)...')
        #varrendo todos os slices
        #contador de sentença no total
        count_sents_total = 0
        #número do batch
        batch_counter_number = 0
        for sl in batch_indexes:

            #dicionário para cada batch
            batch_counter_number += 1
            c_batch_counter_number = get_tag_name(batch_counter_number, prefix = '')
            log_batches[c_batch_counter_number] = {}
            print('Processing batch: ', c_batch_counter_number, '; indexes: ', sl)
            
            #contador de sentenças do batch
            count_sents_batch = 0
            for i in range(sl[0], sl[1]+1):
                
                filename = filtered_sents_filenames[i]
                #print('Processing ', filename, '...')
                DFsents = pd.read_csv(self.diretorio + f'/Outputs/sents_filtered/{filename}.csv', index_col = 0)

                count_sents_batch += len(DFsents.index)
                count_sents_total += len(DFsents.index)
                #print('Total sents: ', count_sents_total)
                
                #obtendo os indexes dos documentos para cada file
                sent_index_dic[filename] = {}
                sent_index_dic[filename]['initial_sent'] = DFsents.index.values[0]
                sent_index_dic[filename]['last_sent'] = DFsents.index.values[-1]
                
                del DFsents
            
            log_batches[c_batch_counter_number]['first_file_index'] = sl[0]
            log_batches[c_batch_counter_number]['last_file_index'] = sl[1]
            log_batches[c_batch_counter_number]['n_sents'] = count_sents_batch
                                                    
        #salvando o log de slices para calcular a matriz TFIDF
        save_dic_to_json(self.diretorio + f'/Outputs/log/tfidf_batches_log.json', log_batches)
        #salvando o index de sentença
        pd.DataFrame.from_dict(sent_index_dic, orient='index').to_csv(self.diretorio + '/Outputs/log/sents_index.csv')
        print('Total sentence number processed: ', count_sents_total)
        
        del log_batches
        del sent_index_dic



    def find_tfidf(self):

        print('\n( Function: find_TFIDF )')

        #checar o arquivo log
        if not os.path.exists(self.diretorio + f'/Outputs/log/tfidf_batches_log.json'):
            print('ERRO!')
            print(f'O arquivo LOG (TFIDF_batches_log.json) não foi encontrado.')
            print('Executar a função text_objects.set_TFIDF_log_and_sent_stats.')
            return
        else:
            #abrindo o TFIDF log file com os slices
            TFIDF_batches_log = load_dic_from_json(self.diretorio + f'/Outputs/log/tfidf_batches_log.json')
            last_batch = int(sorted(list(TFIDF_batches_log.keys()))[-1])

        #checando se há o diretorio npz
        if not os.path.exists(self.diretorio + '/Outputs/tfidf/tfidf_sents_npz'):
            os.makedirs(self.diretorio + '/Outputs/tfidf/tfidf_sents_npz')
        
        #checando se há arquivos no ~/Outputs/tfidf/tfidf_sents_npz
        npz_files_saved = get_filenames_from_folder(self.diretorio + '/Outputs/tfidf/tfidf_sents_npz', file_type = 'npz')
        if npz_files_saved is not None:
            last_npz_file = sorted(npz_files_saved)[-1]
            last_batch_saved = int(re.search(r'[0-9]+', last_npz_file).group())
            print('\nÚltimo batch encontrado na pasta ~/Outputs/tfidf/tfidf_sents_npz: ', last_batch_saved, '\n')

        #caso não tenha nenhum arquivo npz salvo
        else:
            last_batch_saved = 0

        #caso todas as sentenças já tenham sido processadas
        if last_batch == last_batch_saved:
            if os.path.exists(self.diretorio + '/Outputs/tfidf/tfidf_sents_batch.h5'):
                os.remove(self.diretorio + '/Outputs/tfidf/tfidf_sents_batch.h5')
            
            print('O TFIDF de todos os documentos já foi calculado')
            print('> Abortando função: text_objects.find_TFIDF')
            return

        #checando se os IDFs foram calculados
        if not os.path.exists(self.diretorio + f'/Outputs/tfidf/idf.csv'):
            print('Não há IDFs calculados em ~/Outputs/tfidf')
            print('Usar a função "text_objects.find_IDF()"')
            print('> Abortando função: text_objects.find_TFIDF')
            return
        else:
            #carregando os IDFs
            IDF_df = pd.read_csv(self.diretorio + f'/Outputs/tfidf/idf.csv', index_col = 0)
            n_tokens = len(IDF_df.index)
        
            #caso não tenha nenhum token no IDF
            if n_tokens == 0:
                print('Erro! Não há tokens no IDF DF.')
                print('Diminuir o valor de min_token_appereance.')
                print('> Abortando função: text_objects.find_TFIDF')
                return

        #carregando a lista de arquivos processados
        print('Carregando os nomes dos arquivos .csv...')
        filtered_sents_filenames = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')

        #varrendo os batches estabelecidos no LOG
        for batch in range(last_batch_saved + 1, last_batch + 1):

            #gerando o nome do arquivo para o batch
            c_batch_number = get_tag_name(batch, prefix = '')
            
            tfidf_sents_h5_file, tfidf_sents_h5_matrix = create_h5_matrix(shape = (TFIDF_batches_log[str(c_batch_number)]['n_sents'], n_tokens), filepath = self.diretorio + '/Outputs/tfidf/tfidf_sents_batch.h5', dtype=np.float64)
                
            first_file_index_in_batch = TFIDF_batches_log[str(c_batch_number)]['first_file_index']
            last_file_index_in_batch = TFIDF_batches_log[str(c_batch_number)]['last_file_index']
            print('Arquivos do batch - ', filtered_sents_filenames[first_file_index_in_batch], ' a ', filtered_sents_filenames[last_file_index_in_batch])
            print('Indexes do batch - ', first_file_index_in_batch, ' a ', last_file_index_in_batch)
            
            
            #posição do vetor doc na matriz tfidf_sents
            row_number = 0

            #varrendo os files do batch
            for file_index  in range(first_file_index_in_batch, last_file_index_in_batch + 1):
                
                #checando a matriz TFIDF dos artigos
                if not os.path.exists(self.diretorio + '/Outputs/tfidf/tfidf_articles.h5'):
                    tfidf_articles_h5_file, tfidf_articles_h5_matrix = create_h5_matrix(shape = (len(filtered_sents_filenames), n_tokens), filepath = self.diretorio + '/Outputs/tfidf/tfidf_articles.h5', dtype=np.float64)
                
                else:
                    tfidf_articles_h5_file, tfidf_articles_h5_matrix = load_h5_matrix(self.diretorio + '/Outputs/tfidf/tfidf_articles.h5', mode = 'a')

                #abrindo o csv com as sentenças do artigo
                filename = filtered_sents_filenames[file_index]
                sentDF = pd.read_csv(self.diretorio + '/Outputs/sents_filtered/' + f'{filename}.csv', index_col = 0)
                print(f'\nProcessando CSV (~/Outputs/sents_filtered/{filename}.csv)')
                print('Primeira sentença do artigo (row number): ', row_number)

                #abrindo um dicionário para coletar os tokens do artigo
                articles_tokens_dic = {}
                total_tokens_in_article = 0

                #analisando cada sentença
                for index in sentDF.index:
                    
                    sent_tokens_dic = {}
                    sent = sentDF.loc[index, 'Sentence']
    
                    #splitando a sentença em tokens
                    sent_tokens = get_tokens_from_sent(sent, stopwords_list_to_remove = self.stopwords_list, spacy_tokenizer = nlp)
                    n_tokens_sent = len(sent_tokens)
                    
                    #checando a presença do token na sentença
                    for token in sent_tokens:

                        #o valor da coluna (POS X) é a posição do token no IDF_df (index da IDF_df)
                        pos_token_in_IDF = np.where(IDF_df.index.values == token)

                        #caso o token exista no IDF_df (lembre-se que o IDF_df foi filtrado)
                        if len(pos_token_in_IDF[0]) == 1:

                            pos_x = pos_token_in_IDF[0]
                            
                            total_tokens_in_article += 1

                            #contabilizando os tokens para a matriz tfidf_articles
                            try:
                                articles_tokens_dic[token]['counts'] += 1
                            
                            except KeyError:
                                articles_tokens_dic[token] = {}
                                articles_tokens_dic[token]['counts'] = 1
                                articles_tokens_dic[token]['pos_x'] = pos_x

                            #contabilizando os tokens para a matriz tfidf_sents
                            try:
                                sent_tokens_dic[token]['counts'] += 1
                            
                            except KeyError:
                                sent_tokens_dic[token] = {}
                                sent_tokens_dic[token]['counts'] = 1
                                sent_tokens_dic[token]['pos_x'] = pos_x

                    #varrendo os tokens da sentença
                    for token in sent_tokens_dic.keys():
                        pos_x = sent_tokens_dic[token]['pos_x']
                        TF = np.log10( (sent_tokens_dic[token]['counts'] / n_tokens_sent) + 1)
                        IDF = IDF_df.loc[token, 'idf_sent']
                        
                        #calculando o TFIDF
                        TFIDF = TF * IDF
                        tfidf_sents_h5_matrix[row_number, pos_x] = TFIDF
                        #print('> row_number: ', row_number, ' ; token: ', token, '; pos_x: ', pos_x, ' ; TFIDF: ', TFIDF)
                                    
                    #descendo as linhas da matriz tfidf_sents
                    row_number += 1
                
                #varrendo os tokens do artigo
                for token in articles_tokens_dic.keys():
                    pos_x = articles_tokens_dic[token]['pos_x']
                    TF = np.log10( (articles_tokens_dic[token]['counts'] / total_tokens_in_article) + 1 )
                    IDF = IDF_df.loc[token, 'idf_article']
                    
                    #calculando o TFIDF
                    TFIDF = TF * IDF
                    tfidf_articles_h5_matrix[ file_index, pos_x ] = TFIDF
            
                tfidf_articles_h5_file.close()
                del tfidf_articles_h5_matrix
            
            #salvando a sparse matrix do batch da tfidf_sents
            sm = sparse.csr_matrix(tfidf_sents_h5_matrix, dtype = np.float64)
            sparse.save_npz(self.diretorio + f'/Outputs/tfidf/tfidf_sents_npz/sparse_csr_{c_batch_number}.npz', sm, compressed=True)

            print(f'\nArquivo npz salvo (sparse_csr_{batch}.npz).')
            print('Arquivos salvos no npz - ', filtered_sents_filenames[first_file_index_in_batch], ' a ', filtered_sents_filenames[last_file_index_in_batch])

            tfidf_sents_h5_file.close()
            del tfidf_sents_h5_matrix
        
        #salvando a sparse matrix do tfidf_articles
        tfidf_articles_h5_file, tfidf_articles_h5_matrix = load_h5_matrix(self.diretorio + '/Outputs/tfidf/tfidf_articles.h5', mode = 'r')
        am = sparse.csr_matrix(tfidf_articles_h5_matrix, dtype = np.float64)
        sparse.save_npz(self.diretorio + f'/Outputs/tfidf/tfidf_articles_sparse_csr.npz', am, compressed=True)
        tfidf_articles_h5_file.close()
        
        os.remove(self.diretorio + '/Outputs/tfidf/tfidf_sents_batch.h5')
        os.remove(self.diretorio + '/Outputs/tfidf/tfidf_articles.h5')



    def find_sent_stats(self, mode = 'raw_sentences'):

        print('\n( Function: find_sent_stats )')
        print('mode: ', mode)
        
        fig, axes = plt.subplots(3, 1, figsize=(12,18), dpi=300)
        
        if mode.lower() == 'raw_sentences':
            term_name = 'not_filtered'
            file_list = get_filenames_from_folder(self.diretorio + '/Outputs/sents_raw', file_type = 'csv')
            folder_name = 'sents_raw'
        elif mode.lower() == 'filtered_sentences':
            term_name = 'filtered'
            file_list = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')
            folder_name = 'sents_filtered'
        
        #checando os LOGs
        cond1 = os.path.exists(self.diretorio + f'/Outputs/log/stats_sents_tokens_len_{term_name}.json')
        cond2 = os.path.exists(self.diretorio + f'/Outputs/log/stats_articles_sents_len_{term_name}.json')
        if False not in (cond1, cond2):
            print(f'O stats_SENT_TOKENS{term_name} e o stats_article_SENT{term_name} já foram extraídos.')
            print('> Abortando função: text_objects.find_sent_stats')
            return

        sent_token_len = {}
        sent_token_len['with_SW_True'] = []
        sent_token_len['with_SW_False'] = []        
        sent_token_stats = {}
        sent_token_stats['with_SW_True'] = {}
        sent_token_stats['with_SW_False'] = {}
        article_sent_len = []
        article_sent_stats = {}
        
        print(f'Varrendos os documentos em ~/Outputs/{folder_name}')
        for filename in file_list:
            
            print('Processing ', filename, '...')            
            DFsents = pd.read_csv(self.diretorio + f'/Outputs/{folder_name}/{filename}.csv', index_col = 0)

            #guardando o número de sentenças por documento para fazer a estatística
            article_sent_len.append(len(DFsents.index))
            
            #varrendo cada sentença de cada arquivo para determinação da SENT_TOKEN_STATS (estatística de token por sentença)
            for j in DFsents.index:
                
                sent = DFsents.loc[ j, 'Sentence']
                len_with_SW_True = len( get_tokens_from_sent(sent) )
                len_with_SW_False = len([token for token in get_tokens_from_sent(sent) if (token.lower() not in self.stopwords_list and token[ : -1].lower() not in self.stopwords_list)])
                
                sent_token_len['with_SW_True'].append( len_with_SW_True )
                sent_token_len['with_SW_False'].append( len_with_SW_False )
                
        #calculando os parâmetros estatísticos de token por sentenças
        for i in range(len(['False', 'True'])):
            
            boolean = ('False', 'True')[i]
            
            sent_token_len_array = np.array(sent_token_len[f'with_SW_{boolean}'])
            
            sent_token_stats[f'with_SW_{boolean}']['min'] = int(np.min(sent_token_len_array))
            sent_token_stats[f'with_SW_{boolean}']['max'] = int(np.max(sent_token_len_array))
            sent_token_stats[f'with_SW_{boolean}']['avg'] = round(np.mean(sent_token_len_array), 10)
            sent_token_stats[f'with_SW_{boolean}']['std'] = round(np.std(sent_token_len_array), 10)
            sent_token_stats[f'with_SW_{boolean}']['median'] = round(np.median(sent_token_len_array), 10)
            sent_token_stats[f'with_SW_{boolean}']['0_quantile'] = sent_token_len_array.min().astype(float)
            sent_token_stats[f'with_SW_{boolean}']['0.05_quantile'] = round(np.quantile(sent_token_len_array, 0.05), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.1_quantile'] = round(np.quantile(sent_token_len_array, 0.1), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.2_quantile'] = round(np.quantile(sent_token_len_array, 0.2), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.25_quantile'] = round(np.quantile(sent_token_len_array, 0.2), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.3_quantile'] = round(np.quantile(sent_token_len_array, 0.3), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.4_quantile'] = round(np.quantile(sent_token_len_array, 0.4), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.5_quantile'] = round(np.quantile(sent_token_len_array, 0.5), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.6_quantile'] = round(np.quantile(sent_token_len_array, 0.6), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.7_quantile'] = round(np.quantile(sent_token_len_array, 0.7), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.75_quantile'] = round(np.quantile(sent_token_len_array, 0.7), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.8_quantile'] = round(np.quantile(sent_token_len_array, 0.8), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.9_quantile'] = round(np.quantile(sent_token_len_array, 0.9), 10)
            sent_token_stats[f'with_SW_{boolean}']['0.95_quantile'] = round(np.quantile(sent_token_len_array, 0.95), 10)
            sent_token_stats[f'with_SW_{boolean}']['1_quantile'] = sent_token_len_array.max().astype(float)

            axes[i].set_title(f'sent_token_len with_SW_{boolean}')
            axes[i].hist(sent_token_len_array, bins = 50, range=(0,150), color='gray', alpha=0.5)
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['avg'], color='green', label='mean', alpha=0.5)
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['avg'] + sent_token_stats[f'with_SW_{boolean}']['std'], color='red', alpha=0.5, label='+std')
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['avg'] - sent_token_stats[f'with_SW_{boolean}']['std'], color='red', alpha=0.5, label='-std')
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['median'], color='blue', alpha=0.5, label='median')
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['0.25_quantile'], color='orange', alpha=0.5, label='0.25_quantile')
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['0.75_quantile'], color='orange', alpha=0.5, label='0.75_quantile')
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['0.05_quantile'], color='black', alpha=0.5, label='0.05_quantile')
            axes[i].axvline(sent_token_stats[f'with_SW_{boolean}']['0.95_quantile'], color='black', alpha=0.5, label='0.95_quantile')
            axes[i].legend()

        #calculando os parâmetros estatísticos de sentença por artigo
        article_sent_len_array = np.array(article_sent_len)
        
        article_sent_stats['min'] = int(np.min(article_sent_len))
        article_sent_stats['max'] = int(np.max(article_sent_len))
        article_sent_stats['avg'] = round(np.mean(article_sent_len), 10)
        article_sent_stats['std'] = round(np.std(article_sent_len), 10)
        article_sent_stats['median'] = round(np.median(article_sent_len), 10)
        article_sent_stats['0.25_quantile'] = round(np.quantile(article_sent_len, 0.25), 10)
        article_sent_stats['0.75_quantile'] = round(np.quantile(article_sent_len, 0.75), 10)
        article_sent_stats['0.05_quantile'] = round(np.quantile(article_sent_len, 0.05), 10)
        article_sent_stats['0.95_quantile'] = round(np.quantile(article_sent_len, 0.95), 10)
        
        axes[2].set_title('article_sent_len')
        axes[2].hist(article_sent_len_array, bins = 50, color='gray', alpha=0.5)
        axes[2].axvline(article_sent_stats['avg'], color='green', label='mean', alpha=0.5)
        axes[2].axvline(article_sent_stats['avg'] + article_sent_stats['std'], color='red', alpha=0.5, label='+std')
        axes[2].axvline(article_sent_stats['avg'] - article_sent_stats['std'], color='red', alpha=0.5, label='-std')
        axes[2].axvline(article_sent_stats['median'], color='blue', alpha=0.5, label='median')
        axes[2].axvline(article_sent_stats['0.25_quantile'], color='orange', alpha=0.5, label='1_quartile')
        axes[2].axvline(article_sent_stats['0.75_quantile'], color='orange', alpha=0.5, label='3_quartile')
        axes[2].axvline(article_sent_stats['0.05_quantile'], color='black', alpha=0.5, label='0.05_quantile')
        axes[2].axvline(article_sent_stats['0.95_quantile'], color='black', alpha=0.5, label='0.95_quantile')
        axes[2].legend()       
                
        #salvando o sent_stats (estatística de token por sentença)
        save_dic_to_json(self.diretorio + f'/Outputs/log/stats_sents_tokens_len_{term_name}.json', sent_token_stats)
        #salvando o sent_stats (estatística de token por sentença)
        save_dic_to_json(self.diretorio + f'/Outputs/log/stats_articles_sents_len_{term_name}.json', article_sent_stats)
        #salvando a figura com os article_sent_token stats
        fig.savefig(self.diretorio + f'/Outputs/log/article_sent_token_hist_{term_name}.png')
        
        del sent_token_len
        del sent_token_stats
        del article_sent_len
        del article_sent_stats
        
        

    def looking_for_ref_section(self, sentDF, section_marker = 'references', min_ref_to_find_for_section = 10, check_function = False):

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
        pattern2 = r'[A-Z][a-z\-\^~]+(\s|\,|and)+[A-Z\-\s\.\,]+[\,\.]'
        #identificador citações de referências (bibliografia) padrão: X-Y. Zhang,
        pattern3 = r'[A-Z\-\s\.\,]+[A-Z][a-z\-\^~]+[\s\,]'    
        
        #encontrando os indexes das referências
        found_section = False
        found_ref_section_delimiter = False
        reference_section_index_list = []
        refs_section_index = None
    
        #varrendo as sentenças
        for i in sentDF.index:
            
            #abrindo a sentença
            sent = sentDF.loc[i, 'Sentence']
                
            #print('looking for references: ', i,' ; ', sent)
            #time.sleep(1)
                            
            #caso a seção já tenha sido encontrada com um número mínimo de citações
            if len(reference_section_index_list) >= min_ref_to_find_for_section:
                found_section = True
                refs_section_index = reference_section_index_list[0]
                break
            
            elif found_ref_section_delimiter is True:
                
                #encontrar a citação dos nomes do tipo Zhang X-Y.,
                if re.match(pattern2, sent):
                    reference_section_index_list.append(i)
                    
                    if check_function is True:
                        print('Ref found: ', sent)
                        pass

                #encontrar a citação dos nomes do tipo X-Y. Zhang,
                elif re.match(pattern3, sent):
                    reference_section_index_list.append(i)
                    
                    if check_function is True:
                        print('Ref found: ', sent)
                        pass
            
            #primeira etapa: encontrar o termo References ou Bibliography                    
            else:
                if re.match(pattern1, sent):
                    #print(re.match(pattern1, sent))
                    found_ref_section_delimiter = True
                    reference_section_index_list.append(i)
                    
                    if check_function is True:
                        print('> Ref section begin: ', i, ' ; ', sent)
                            
        return found_section, refs_section_index
