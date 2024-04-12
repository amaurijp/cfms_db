#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import pandas as pd
import regex as re
import webbrowser as wb

from FUNCTIONS import error_incompatible_strings_input
from FUNCTIONS import create_article_file_list_with_filename
from FUNCTIONS import error_print_abort_class
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import save_dic_to_json

from functions_PARAMETERS import list_numerical_parameter
from functions_PARAMETERS import list_textual_parameter        
from functions_PARAMETERS import regex_patt_from_parameter
from functions_PARAMETERS import get_physical_units_converted_to_SI

from functions_TOKENS import get_nGrams_list

class DataFrames(object):
    
    def __init__(self, mode = 'select_parameters_manual', diretorio = None):
        
        print('\n( Class: DataFrames )')

        self.diretorio = diretorio
        self.class_name = 'DataFrames'
        self.mode = mode.lower()

        #testando os erros de inputs
        self.abort_class = error_incompatible_strings_input('mode', mode, ('collect_parameters_automatic', 'collect_parameters_manual', 'select_sentences'), class_name = self.class_name)
                
        print(f'Mode chosen: {mode}')            


    def set_settings(self, 
                     input_DF_name = 'frags_extracted', 
                     output_DF_name = 'paramater', 
                     parameter = 'temperature',
                     hold_filenames = False, 
                     hold_instances_number = False,
                     ngram_for_textual_search = 2,
                     min_ngram_appearence = 5,
                     lower_sentence_in_textual_search = False,
                     numbers_extraction_mode = 'all',
                     filter_unique_results = False,
                     cluster_min_max_num_vals = False,
                     send_to_consolidated_DF = False):
        
        print('( Function: set_files )')
        print('Setting Files...')
        
        self.input_DF_name = input_DF_name
        self.output_DF_name = output_DF_name
        self.input_parameter = parameter
        self.hold_filenames = hold_filenames
        self.hold_instances_number = hold_instances_number
        self.min_ngram_appearence = min_ngram_appearence
        self.lower_sentence_in_textual_search = lower_sentence_in_textual_search
        self.ngram_for_textual_search = ngram_for_textual_search
        self.numbers_extraction_mode = numbers_extraction_mode
        self.filter_unique_results = filter_unique_results
        self.cluster_min_max_num_vals = cluster_min_max_num_vals
        self.send_to_consolidated_DF = send_to_consolidated_DF


    
    def get_data(self, article_file_type = ''):

        print('( Function: get_data )')
        
        #checando erros de instanciação/inputs
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return

        #checando se existe um Data Frame de fragmentos
        print(f'Procurando... {self.diretorio}/Outputs/extracted/' + f'{self.input_DF_name}.csv')
        if os.path.exists(self.diretorio + '/Outputs/extracted/' + f'{self.input_DF_name}.csv'):
            self.extracted_sents_DF = pd.read_csv(self.diretorio + '/Outputs/extracted/' + f'{self.input_DF_name}.csv', index_col=[0,1], dtype=object)
            print(f'Carregando o DataFrame de INPUT com as sents extraidas (~/Outputs/extracted/{self.input_DF_name}.csv)')

                
            #dic para salvar as SI PUs padronizadas para cada features (ou parâmetros)
            if not os.path.exists(self.diretorio + '/Outputs/dataframes/SI_PUs.json'):
                self.SI_PUs_dic_to_record = {}
            else:
                self.SI_PUs_dic_to_record = load_dic_from_json(self.diretorio + '/Outputs/dataframes/SI_PUs.json')
            
            #definindo os inputs de parâmetros
            for filename in self.extracted_sents_DF.index.levels[0]:

                self.filename = filename

                #caso seja para extrair os valores
                if self.send_to_consolidated_DF is False:
                    #checando se já existe um output Data Frame para esses parâmetros
                    if os.path.exists(self.diretorio + f'/Outputs/dataframes/{self.output_DF_name}_mode_{self.numbers_extraction_mode}.csv'):
                        self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/{self.output_DF_name}_mode_{self.numbers_extraction_mode}.csv', index_col=[0,1], dtype=object)
                        self.output_DF.index.names = ['Filename', 'Counter']
                        #print(f'Carregando o DataFrame de OUTPUT (~/Outputs/extracted/{self.output_DF_name}.csv)')
                        
                        try:
                            last_article_processed = self.output_DF.index.levels[0][-1]
                        except IndexError:
                            last_article_processed = 'ATC00000'
                    else:
                        print(f'Output DF {self.output_DF_name}_mode_{self.numbers_extraction_mode}.csv não encontrado.')
                        print(f'Criando o output_DF data frame: {self.output_DF_name}_mode_{self.numbers_extraction_mode}.csv')
                        
                        #caso tenha que ser gerada a output_DF
                        self.output_DF = pd.DataFrame(columns=['Filename', 'Counter'], dtype=object)
                        self.output_DF.set_index(['Filename', 'Counter'], inplace=True)
                        last_article_processed = 'ATC00000'
                
                    article_extracted_list = create_article_file_list_with_filename(last_article_processed)

                    #abrindo o search-extract report
                    if os.path.exists(self.diretorio + f'/Outputs/log/se_report.json'):
                        #carregando o dicionário
                        self.search_report_dic = load_dic_from_json(self.diretorio + f'/Outputs/log/se_report.json')
                        if self.search_report_dic['search'][self.input_DF_name]['searching_status'] != 'finished':
                            print(f'Erro! O processo de extração para o search_input {self.input_DF_name} ainda não terminou.' )
                            return                        
                        
                        elif last_article_processed == 'ATC00000':
                            self.search_report_dic['export'] = {}
                            self.search_report_dic['export'][self.input_DF_name] = {}
                            self.search_report_dic['export'][self.input_DF_name]['last_article_processed'] = None
                            self.search_report_dic['export'][self.input_DF_name]['total_finds'] = 0
                            self.search_report_dic['export'][self.input_DF_name]['article_finds'] = 0
                            
                    else:
                        print('Erro! LOG counter_se_report não encontrado em ~/outputs/log' )
                        print(f'Erro! O processo de extração para o search_input {self.input_DF_name} não foi feito.' )
                        return

                #caso seja para consolidar a DF
                elif self.send_to_consolidated_DF is True:
                    
                    if os.path.exists(self.diretorio + f'/Outputs/log/consolidated_files.json'):
                        
                        self.consolidated_files = load_dic_from_json(self.diretorio + f'/Outputs/log/consolidated_files.json')

                        try:
                            lastfile_consolidated_for_parameter = self.consolidated_files[self.input_parameter]
                            article_extracted_list = create_article_file_list_with_filename(lastfile_consolidated_for_parameter)
                        
                        except KeyError:
                            self.consolidated_files[self.input_parameter] = 'ATC00000'
                            article_extracted_list = create_article_file_list_with_filename('ATC00000')

                    else:
                        self.consolidated_files = {}
                        self.consolidated_files[self.input_parameter] = 'ATC00000'
                        article_extracted_list = create_article_file_list_with_filename('ATC00000')
                
                #varrendo os arquivos
                if self.filename not in article_extracted_list:
                    
                    print(f'\n------------------------------------------')
                    print(f'Extracting parameters from {self.filename}')
                    
                    #dicionário para coletar os parâmetros numéricos extraídos
                    self.parameters_extracted = {}
                    self.parameters_extracted['filename'] = self.filename
                    self.parameters_extracted['selected_sent_index'] = None
                    self.parameters_extracted['param_type'] = None

                    print('\nParameter: ( ', self.input_parameter, ' )') 
                    print('Fragments extracted from: ', self.filename)

                    #varrendo as linhas com as sentenças para esse artigo (input_DF)
                    for i in self.extracted_sents_DF.loc[ (self.filename , ) ,  ].index:

                        #sentença
                        sent = self.extracted_sents_DF.loc[ (self.filename , i ) , self.input_DF_name ]
                        
                        #index de sentença
                        sent_index = int( self.extracted_sents_DF.loc[ (self.filename , i ) , self.input_DF_name + '_index' ] )

                        #coletando a sentença e o sent_index
                        self.parameters_extracted[sent_index] = {}
                        self.parameters_extracted[sent_index]['sent'] = sent
                        self.parameters_extracted[sent_index]['got_parameter_from_sent'] = False
                                                                
                        #Mostrando as sentenças a serem processadas para esse artigo
                        print(f'\nIndex {i} (sent_index {sent_index}):', sent, '\n')
                        
                        #só entramos na sequência abaixo para coletar os parâmetros
                        if self.mode != 'select_sentences':
                            
                            #checando se o parâmetro irá para a extração numérica
                            if self.input_parameter in list_numerical_parameter():
                                
                                #extraindo os parâmetros numéricos com as unidades físicas
                                numerical_params_extracted_from_sent = self.extract_numerical_parameters(sent, sent_index, self.input_parameter, extract_mode = self.numbers_extraction_mode)
                                
                                #caso tenha sido extraído algum output numérico corretamente
                                if numerical_params_extracted_from_sent['total_num_outputs_extracted'] > 0 and numerical_params_extracted_from_sent['extract_error'] is False:
                                    
                                    print('> Extracted numerical outputs - n_num_outputs: ', numerical_params_extracted_from_sent['total_num_outputs_extracted'], ' ; SI_units: ', numerical_params_extracted_from_sent['PUs_extracted'])
                                    print('>', numerical_params_extracted_from_sent['all_num_output'] )

                                    self.parameters_extracted['param_type'] = 'numerical'

                                    #adicionando as SI PUs que serão exportadas para a consolidated_DF
                                    self.SI_PUs_dic_to_record[self.input_parameter] = numerical_params_extracted_from_sent['PUs_extracted']
                                    
                                    #caso o método de coleta seja manual
                                    if self.mode == 'collect_parameters_manual':
                                        
                                        while True:
                                            
                                            user_entry = str(input('\nConfirme o(s) valor(es) extraidos (yes/y): '))

                                            if user_entry[0].lower() == 'y':
                                                #coletando os parâmetros extraidos da sentença
                                                self.parameters_extracted[sent_index]['param_captured'] = numerical_params_extracted_from_sent['all_num_output']
                                                self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True
                                                break
                                            
                                            elif user_entry == 'exit':
                                                print('> Abortando função: DataFrames.get_data')
                                                return
                                            
                                            elif user_entry[0].lower() == 'n':
                                                break

                                            else:
                                                print('ERRO > Input inválido (digite "yes", "no" ou "exit")\n')
                                        
                                    elif self.mode == 'collect_parameters_automatic':
                                        #coletando os parâmetros extraidos da sentença [ counter , sent_index , val ]
                                        self.parameters_extracted[sent_index]['param_captured'] = numerical_params_extracted_from_sent['all_num_output']
                                        self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True
                                    
                                    #time.sleep(2)
                                
                                #caso nenhum output numérico tenha sido exportado
                                else:
                                    print(f'> Nenhum parâmetro numérico foi extraído para o parameter: {self.input_parameter}')

                            #checando se o parâmetro irá para a extração textual                                                
                            elif self.input_parameter in list_textual_parameter(diretorio=self.diretorio):
                                
                                #caso a sentença seja procurada em lower
                                sent_modified = sent
                                if self.lower_sentence_in_textual_search is True:
                                    sent_modified = sent.lower()

                                textual_params_extracted_from_sent = self.extract_textual_parameters(sent_modified, sent_index, self.input_parameter)
                                
                                #caso tenha sido extraído algum output textual
                                if textual_params_extracted_from_sent['total_textual_outputs_extracted'] > 0:
                                    print('> Extracted textual outputs - n_textual_outputs: ', textual_params_extracted_from_sent['total_textual_outputs_extracted'], ' )')
                                    print('> Parâmetros textuais extraídos: ', self.input_parameter)
                                    print('> ', textual_params_extracted_from_sent['all_textual_output'] )
                                    
                                    self.parameters_extracted['param_type'] = 'textual'

                                    #caso o método de coleta seja manual
                                    if self.mode == 'collect_parameters_manual':

                                        while True:

                                            user_entry = str(input('\nConfirme o(s) valor(es) extraidos (yes/y): '))

                                            if user_entry.lower() in ('y', 'yes'):
                                                #coletando os parâmetros extraidos da sentença
                                                self.parameters_extracted[sent_index]['param_captured'] = textual_params_extracted_from_sent['all_textual_output']
                                                self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True
                                                break
                                            
                                            elif user_entry == 'exit':
                                                print('> Abortando função: DataFrames.get_data')
                                                return
                                            
                                            elif user_entry.lower() in ('n', 'no'):
                                                break

                                            else:
                                                print('ERRO > Input inválido (digite "yes", "no" ou "exit")\n')
                                    
                                    elif self.mode == 'collect_parameters_automatic':
                                        #coletando os parâmetros extraidos da sentença
                                            self.parameters_extracted[sent_index]['param_captured'] = textual_params_extracted_from_sent['all_textual_output']
                                            self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True

                                    #time.sleep(2)
                                    
                                else:
                                    print(f'> Nenhum parâmetro textual foi extraído para o parameter: {self.input_parameter}')
                                
                            #caso o parâmetro introduzido (input_parameter) não esteja na lista do functions_PARAMETERS
                            else:
                                #listar os parâmetros disponíveis para extração
                                available_inputs = list_numerical_parameter() + list_textual_parameter(diretorio=self.diretorio)
                                abort_class = error_incompatible_strings_input('input_parameter', self.input_parameter, available_inputs, class_name = self.class_name)
                                if abort_class is True:
                                    return                                        
                                                            
                    while True: 
                        try:
                            #escolhendo entre modo de seleção de sentenças ou modo de coleta de parâmetros  
                            if self.mode == 'select_sentences':
                                print('\nDigite o(s) index(es) das sentenças de interesse (digite valores inteiros)')
                                print('Outros comandos: "+" para ignorar esse artigo e ir para o próximo; "open" para abrir o artigo; e "exit" para sair.')
                                self.param_val = str(input('Index: '))
                                
                            elif self.mode in ('collect_parameters_automatic', 'collect_parameters_manual'):
                                self.param_val = '*'
                            
                            #processando o input                                        
                            if self.param_val.lower() == 'open':
                                wb.open_new(self.diretorio + '/DB/' + self.filename + f'.{article_file_type}')
                                continue

                            elif self.param_val.lower() == '*':
                                break
                            
                            elif self.param_val.lower() == '+':
                                break
                            
                            elif self.param_val.lower() == 'exit':
                                print('> Abortando função: DataFrames.get_data')
                                return
                            
                            else:
                                #caso o modo seja para coletar as sentenças para treinamento ML                                                                                
                                if self.mode == 'select_sentences':
                                    try:
                                        selected_sent_key = int(self.param_val)
                                        #verificando se esse parametro inserido é relativo ao valor de key das sentenças coletadas
                                        self.parameters_extracted['selected_sent_index'] = selected_sent_key
                                        break
                                    except (TypeError, KeyError):
                                        print('Erro! Inserir um index válido para coletar a sentença.')
                                        continue                                                                                            
                    
                        except ValueError:
                            print('--------------------------------------------')
                            print('Erro!')
                            print('Inserir valor válido')
                            break                                                                                        
            
                    #se o último input introduzido não foi "+" (que é usado para passar para o proximo artigo)   
                    if self.param_val != '+':
                                        
                        #caso seja modo de coleta de dados e algum parametro foi extraído
                        if self.mode != 'select_sentences':
                            
                            #print(self.parameters_extracted)

                            #definindo uma lista para colocar todos os parâmetros coletados
                            all_values_collected_list = []

                            #definindo um dicionário para colocar os outputs modificados
                            self.parameters_extracted[self.input_parameter] = {}
                            self.parameters_extracted[self.input_parameter]['outputs'] = []
                            self.parameters_extracted[self.input_parameter]['extracted_sent_index'] = []
                            #o len_outputs conta quantos valores diferentes foram capturados na extração
                            self.parameters_extracted[self.input_parameter]['len_outputs'] = 0
                            
                            #indicador se algum parametro foi coletado
                            got_any_parameter = False

                            #caso os resultados numéricos sejam clusterizados (só entra o menor e o maior valor encontrado no artigo)
                            if (self.parameters_extracted['param_type'] == 'numerical') and (self.cluster_min_max_num_vals is True):

                                #varrendo todas as sentenças que tiveram dados extraidos
                                for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:
                                    
                                    #caso os parametros não foram coletados na sentença (isso acontece somente no modo de coleta manual)
                                    if self.parameters_extracted[sent_index]['got_parameter_from_sent'] is False:
                                        continue
                                    else:
                                        got_any_parameter = True

                                    #varrendo os outputs no formato: [ counter , sent_index , val ]
                                    for val in self.parameters_extracted[sent_index]['param_captured']:
                                        
                                        all_values_collected_list.append( round(float( val ), 10) )                                        
                                        self.parameters_extracted[self.input_parameter]['len_outputs'] += 1
                                        
                                        if sent_index not in self.parameters_extracted[self.input_parameter]['extracted_sent_index']:
                                            self.parameters_extracted[self.input_parameter]['extracted_sent_index'].append(sent_index)

                                if len(all_values_collected_list) > 0:
                                    min_val = round(min(all_values_collected_list), 10)
                                    max_val = round(max(all_values_collected_list), 10)                                    
                                    self.parameters_extracted[self.input_parameter]['outputs'].append( str(min_val) + ', ' + str(max_val) )
                                
                            #caso todos os resultados entrem
                            elif (self.parameters_extracted['param_type'] == 'numerical') and (self.cluster_min_max_num_vals is False):

                                #varrendo todas as sentenças que tiveram dados extraidos
                                clustered_numbers = ''
                                for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:
                                    
                                    #caso os parametros não foram coletados na sentença (isso acontece somente no modo de coleta manual)
                                    if self.parameters_extracted[sent_index]['got_parameter_from_sent'] is False:
                                        continue
                                    else:
                                        got_any_parameter = True
                                    
                                    #varrendo os outputs no formato: [ counter , sent_index , val ]
                                    for val in self.parameters_extracted[sent_index]['param_captured']:
                                        
                                        if self.filter_unique_results == True:
                                            if val not in all_values_collected_list:
                                                clustered_numbers += str( round(float( val ), 10) ) + ', '
                                                all_values_collected_list.append(val)
                                            
                                        else:
                                            clustered_numbers += str( round(float( val ), 10) ) + ', '
                                        
                                        self.parameters_extracted[self.input_parameter]['len_outputs'] += 1
                                        
                                        if sent_index not in self.parameters_extracted[self.input_parameter]['extracted_sent_index']:
                                            self.parameters_extracted[self.input_parameter]['extracted_sent_index'].append(sent_index)
                                
                                if len(clustered_numbers) > 0:
                                    
                                    #tirando o último ', '
                                    clustered_numbers = clustered_numbers[ : -2]
                                    self.parameters_extracted[self.input_parameter]['outputs'].append( clustered_numbers )
                                    
                            #caso todos os resultados textuais entrem
                            elif (self.parameters_extracted['param_type'] == 'textual'):

                                #varrendo todas as sentenças que tiveram dados extraidos
                                for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:

                                    #caso os parametros não foram coletados na sentença (isso acontece somente no modo de coleta manual)
                                    if self.parameters_extracted[sent_index]['got_parameter_from_sent'] is False:
                                        continue
                                    else:
                                        got_any_parameter = True

                                    #varrendo os outputs no formato: [ counter , sent_index , val ]
                                    for val in self.parameters_extracted[sent_index]['param_captured']:
                                        
                                        #caso os parâmetros sejam extraidos somente um vez
                                        if self.filter_unique_results == True:
                                            if val not in self.parameters_extracted[self.input_parameter]['outputs']:
                                                
                                                self.parameters_extracted[self.input_parameter]['outputs'].append( val )
                                                self.parameters_extracted[self.input_parameter]['len_outputs'] += 1
                                        
                                                if sent_index not in self.parameters_extracted[self.input_parameter]['extracted_sent_index']:
                                                    self.parameters_extracted[self.input_parameter]['extracted_sent_index'].append(sent_index)
                                        
                                        #caso todos os parâmetros textuais de todas as sentenças sejam extraídos
                                        else:
                                            self.parameters_extracted[self.input_parameter]['outputs'].append( val )
                                            self.parameters_extracted[self.input_parameter]['len_outputs'] += 1
                                            
                                            if sent_index not in self.parameters_extracted[self.input_parameter]['extracted_sent_index']:
                                                self.parameters_extracted[self.input_parameter]['extracted_sent_index'].append(sent_index)
                                
                            #apagando as keys de cada sentença
                            for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:
                                del(self.parameters_extracted[sent_index])

                            #gerando a DF com os dados colletados
                            print('\nSummary: parameters extracted from file: ', self.filename)
                            print(self.parameters_extracted)
                            #time.sleep(5)
                            
                            if got_any_parameter is True:
                                self.convert_parameters_extracted_to_DF()


            #consolidando o report na DF caso seja o ultimo arquivo de procura        
            if self.filename == self.extracted_sents_DF.index.levels[0][-1] and self.send_to_consolidated_DF is False:
                self.generate_search_report()


        else:
            print('Erro! Não foi encontrado um DF de fragamentos de artigos.')
            print('> Abortando a classe: DataFrames')
            return



    def convert_parameters_extracted_to_DF(self):
                
        
        

        if not os.path.exists(self.diretorio + '/Outputs/dataframes'):
            os.makedirs(self.diretorio + '/Outputs/dataframes')

        #número de outputs
        self.number_new_instances_extracted = len( self.parameters_extracted[self.input_parameter]['outputs'] )
        
        #caso o modo seja de selecionar sentenças
        if self.mode == 'select_sentences':
            
            #gerando e salvando as DF com as sentenças
            if self.number_new_instances_extracted >= 1:
                self.export_to_DF()
        
        else:
            #gerando e salvando as DF com os parâmetros
            if self.number_new_instances_extracted >= 1:
                self.export_to_DF()



    def check_conditions_to_consolidated_DF(self):
        

        checked = True

        #procurando se o index de artigo já tem sample number
        filename = self.parameters_extracted['filename']

        print(f'Procurando... {self.diretorio}/Outputs/dataframes/consolidated_DF.csv')
        if not os.path.exists(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv'):
            print(f'Criando a DF consolidada... (~/Outputs/dataframes/consolidated_DF.csv)')
            self.consolidated_DF = pd.DataFrame(index=[[],[]], dtype=object)
            self.consolidated_DF.index.names = ['Filename', 'Counter']
            self.consolidated_DF['instances_counter'] = ''
            self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv')

        #carregando a DF consolidada
        else:
            print(f'Abrindo a consolidated DF: {self.diretorio}/Outputs/dataframes/consolidated_DF.csv')
            self.consolidated_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv', index_col=[0,1], dtype=object)


        #se já houver um instance_counter na consolidated_DF
        try:        
            self.current_instances_number = int( float(self.consolidated_DF.loc[ ( filename , 0 ), 'instances_counter' ]) )
        
        #exceção caso o index não exista na DF ou caso o valor de sample_counter seja None
        except KeyError:
            self.consolidated_DF.loc[ ( filename , 0 ), 'instances_counter' ] = 0
            self.current_instances_number = 0

        print('current_instance_number = ', self.current_instances_number)
        print('number_new_instances_to_add = ', self.number_new_instances_extracted)
        
        #condições para fazer a consolidação do FULL DF
        #para atualização no número de amostras
        cond_match_instances_number = False
        #só se atualiza quando o número de amostra atualizao for igual àquele que está na DF consolidada
        if self.hold_instances_number is True:
            if self.number_new_instances_extracted == self.current_instances_number:
                cond_match_instances_number = True        
        else:        
            #quando o número de amostra atualizado for igual àquele que está na DF consolidada
            if self.number_new_instances_extracted == self.current_instances_number:
                cond_match_instances_number = True
            #ou quando um valor seja 1 e outro maior ou igual a 1
            elif self.number_new_instances_extracted == 1 and  self.current_instances_number >= 1:
                cond_match_instances_number = True
            #ou quando um valor seja maior ou igual a 1 e o outro 0 ou 1
            elif self.number_new_instances_extracted >= 1 and  self.current_instances_number in (0, 1):
                cond_match_instances_number = True

        #quando o atributo de hold_samples estive ligado, o algoritmo só adiciona parâmetros nas amostras que já tem sample_counter > 0        
        cond_hold_samples = True
        if self.hold_filenames is True:
            #só atualizará se o já houver um número de amostra na DF consolidada
            if (self.current_instances_number == 0):
                cond_hold_samples = False
        else:
            pass                
                
        #checando os resultados
        print('cond_match_instances_number: ', cond_match_instances_number)
        print('cond_hold_samples: ', cond_hold_samples)
        #time.sleep(2)

        #se alguma das condições falhou
        if False in (cond_match_instances_number, cond_hold_samples):
            checked = False

        return checked



    def export_to_DF(self):

        print('\nExporting to DF...')
        
        filename = self.parameters_extracted['filename']

        #trabalhando na DF de extração
        if self.send_to_consolidated_DF is False:

            #trabalhando no output_DF
            for i in range( self.number_new_instances_extracted ):
                
                #salvando no output_DF
                output_val = self.parameters_extracted[self.input_parameter]['outputs'][i]
                
                self.output_DF.loc[ ( filename , i ) , self.input_parameter ] = output_val
                self.output_DF.loc[ ( filename , i ) , self.input_parameter + '_index' ] = str( self.parameters_extracted[self.input_parameter]['extracted_sent_index'] )
            
            #salvando a output_DF
            self.output_DF.to_csv(self.diretorio + f'/Outputs/dataframes/{self.output_DF_name}_mode_{self.numbers_extraction_mode}.csv')
            print(f'\nSalvando output_DF para {filename}...')

            #contadores para exportar nas DFs de report
            self.search_report_dic['export'][self.input_DF_name]['last_article_processed'] = self.parameters_extracted['filename']
            self.search_report_dic['export'][self.input_DF_name]['total_finds'] += self.number_new_instances_extracted
            self.search_report_dic['export'][self.input_DF_name]['article_finds'] += 1
            
            #salvando o DF report
            save_dic_to_json(self.diretorio + f'/Outputs/log/se_report.json', self.search_report_dic)


        #trabalhando no consolidated_DF
        elif self.send_to_consolidated_DF is True:
            
            if self.check_conditions_to_consolidated_DF() is True:

                #caso o número de instancias a ser adicionado seja igual ao número de instancias que já estão no DF consolidado
                if ( self.number_new_instances_extracted == self.current_instances_number ) or ( self.number_new_instances_extracted >= 1 and self.current_instances_number in (0, 1) ):
                    
                    for i in range( self.number_new_instances_extracted ):
                        
                        #salvando no consolidated_DF
                        output_val = self.parameters_extracted[self.input_parameter]['outputs'][i]
                        
                        self.consolidated_DF.loc[ ( filename , i ) , self.input_parameter ] = output_val
                        self.consolidated_DF.loc[ ( filename , i ) , self.input_parameter + '_index' ] = str( self.parameters_extracted[self.input_parameter]['extracted_sent_index'] )
                        self.consolidated_DF.loc[ ( filename , i ) , 'instances_counter' ] = self.number_new_instances_extracted
                
                #outras situações
                elif self.number_new_instances_extracted == 1 and self.current_instances_number >= 1:

                    for i in range( self.current_instances_number ):
                        
                        #salvando no consolidated_DF
                        output_val = self.parameters_extracted[self.input_parameter]['outputs'][0]

                        self.consolidated_DF.loc[ ( filename , i ) , self.input_parameter ] = output_val
                        self.consolidated_DF.loc[ ( filename , i ) , self.input_parameter + '_index' ] = str( self.parameters_extracted[self.input_parameter]['extracted_sent_index'] )
                        self.consolidated_DF.loc[ ( filename , i ) , 'instances_counter' ] = self.current_instances_number
                
                self.consolidated_files[self.input_parameter] = self.filename
                save_dic_to_json(self.diretorio + '/Outputs/dataframes/SI_PUs.json', self.SI_PUs_dic_to_record)
                save_dic_to_json(self.diretorio + '/Outputs/log/consolidated_files.json', self.consolidated_files)
                self.consolidated_DF.sort_index(level=[0,1], inplace=True)
                self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv')
                print('> adicionando na consolidated DF : ', self.consolidated_DF.loc[ ( filename , ) ,  ])
                print(f'Atualizando DataFrame consolidado para {filename}...')
                    
            else:
                print('\nHá incompatibilidade entre os outputs (função: check_conditions_to_consolidated_DF).')
                print(f'DataFrame para o {filename} não foi consolidada.')
                return
    
    

    def extract_textual_parameters(self, text, text_index, parameter):

        
        extracted_dic = {}
        extracted_dic['text_index'] = text_index
        extracted_dic['parameter'] = parameter
        extracted_dic['all_textual_output'] = []
        
        str_list = []
        term_list = get_nGrams_list( [ parameter ], ngrams = self.ngram_for_textual_search, min_ngram_appearence = self.min_ngram_appearence, diretorio=self.diretorio)
        
        #varrendo os termos
        for term in term_list:
                
            #coletando o termo
            if re.search(term, text):

                term_to_get = re.search(term, text).captures()[0]

                #adicionando o termo na lista
                str_list.append(term_to_get)
    
        str_list.sort()
        extracted_dic['all_textual_output'] = str_list
        
        #exportando o número de resultados
        extracted_dic['total_textual_outputs_extracted'] = len(str_list)
        
        #print(extracted_dic)
        #time.sleep(10)
        return extracted_dic


    def extract_numerical_parameters(self, text, text_index, parameter, extract_mode = 'all'):

        #obtendo os padrões regex para encontrar e para não encontrar
        parameters_patterns = regex_patt_from_parameter(parameter)
        
        #print('\nPU_to_find_regex:\n', parameters_patterns['PU_to_find_regex'])
    
        extracted_dic = {}
        extracted_dic['total_num_outputs_extracted'] = 0
        extracted_dic['PUs_extracted'] = None
        extracted_dic['text_index'] = text_index
        extracted_dic['parameter'] = parameter
        extracted_dic['extract_error'] = False
        extracted_dic['all_num_output'] = []
        
        #encontrandos os parâmetros numéricos
        counter = 1
        if re.finditer(parameters_patterns['PU_to_find_regex'], text ):
            for match in re.finditer(parameters_patterns['PU_to_find_regex'], text ):
                
                PU_found = match.groups()[-2]

                extracted_dic[counter] = {}
                extracted_dic[counter]['nums_not_coverted'] = []
                extracted_dic[counter]['PU_not_converted'] = PU_found

                collected_nums = []
                for find in match.groups():
                    
                    #o padrão regex é para pegar as entradas que só tenham números (sem outros caracteres)
                    if (find is not None) and (find not in collected_nums) and not re.search(r'[\s]', find):

                        #checando se o parâmetro é numérico ou a unidade PU
                        try:
                            float(find)
                            extracted_dic[counter]['nums_not_coverted'].append( float(find) )
                            extracted_dic['total_num_outputs_extracted'] += 1
                            collected_nums.append( find )
                        
                        except ValueError:
                            continue
                
                counter += 1

        #varrendo os findings
        counter = 0
        for key in extracted_dic.keys():
            
            #encontrando as entradas numéricas com os números e as PUs
            try:
                float(key)
                
                if key > 1 and extract_mode == 'one':
                    extracted_dic['extract_error'] = True
                    break

                #fazendo a conversão das unidades
                factor_to_multiply , factor_to_add , PU_in_SI = get_physical_units_converted_to_SI( extracted_dic[key]['PU_not_converted'].split() )
                
                if None not in (factor_to_multiply , factor_to_add , PU_in_SI):

                    if extracted_dic['PUs_extracted'] is None:
                        extracted_dic['PUs_extracted'] = PU_in_SI
                    
                    extracted_dic[key]['PU'] = PU_in_SI
                    extracted_dic[key]['nums'] = []
                    
                    for num in extracted_dic[key]['nums_not_coverted']:
                        num_converted = round( ( float(num) * factor_to_multiply) + factor_to_add , 9)
                        extracted_dic[key]['nums'].append(num_converted)
                        #nesta lista se coloca todos os números extraidos
                        extracted_dic['all_num_output'].append( num_converted )

                    del extracted_dic[key]['PU_not_converted']
                    del extracted_dic[key]['nums_not_coverted']
                
                else:
                    print('ERRO de conversão das PUs. Checar se as unidades da PU: ', extracted_dic[key]['PU_not_converted'].split(), ' estão na função "get_physical_units_converted_to_SI".')
                    time.sleep(10)


            except ValueError:
                continue

        #print('num_extraction_dic:', extracted_dic)

        #time.sleep(1)
        return extracted_dic
    


    def generate_search_report(self):
        
        #abrindo o SE report            
        if os.path.exists(self.diretorio + '/Settings/SE_inputs.csv'):
            search_report_DF = pd.read_csv(self.diretorio + '/Settings/SE_inputs.csv', index_col = 0)

            search_report_DF.loc[self.input_DF_name, 'Total Extracted'] = self.search_report_dic['export'][self.input_DF_name]['total_finds']
            search_report_DF.loc[self.input_DF_name, 'Articles Extracted'] = self.search_report_dic['export'][self.input_DF_name]['article_finds']
            search_report_DF.loc[self.input_DF_name , 'export_status' ] = 'finished'

            search_report_DF.sort_index(inplace=True)
            search_report_DF.to_csv(self.diretorio + '/Settings/SE_inputs.csv')
            print('Salvando o SE report em ~/Settings/SE_inputs.csv')
                
