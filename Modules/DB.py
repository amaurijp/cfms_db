#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import pandas as pd

from PDF import PDFTEXT
from XML import XMLTEXT

from FUNCTIONS import get_tag_name
from FUNCTIONS import filename_gen

from functions_TEXTS import save_text_to_TXT

class database(object):
    
    def __init__(self, diretorio = None, article_file_type = ''):
        
        print('\n( Class: database )')
        
        self.diretorio = diretorio
        self.article_file_type = article_file_type.lower()
        
        #caso não haja o diretorio ~/Outputs
        if not os.path.exists(self.diretorio + '/Outputs'):
            os.makedirs(self.diretorio + '/Outputs')
        #caso não haja o diretorio ~/Articles_to_add
        if not os.path.exists(self.diretorio + '/Articles_to_add'):
            os.makedirs(self.diretorio + '/Articles_to_add')
        #caso não haja o diretorio ~/DB
        if not os.path.exists(self.diretorio + '/DB'):
            os.makedirs(self.diretorio + '/DB')

        print('Iniciando o módulo DB...')
        #carregando o DB atual
        if os.path.exists(diretorio + '/Outputs/DB_ID.csv'):
            print('DataBase de artigos encontrado (~/Outputs/DB_ID.csv)')
            self.DB_ID = pd.read_csv(diretorio + '/Outputs/DB_ID.csv', index_col=0)
            self.last_index = int(len(self.DB_ID.index))
            #print('\n', self.DB_ID)
            print('Index atual: ', self.last_index)
        else:        
            print('DataBase de artigos não encontrado.')
            print('Criando DB...\n')
            self.last_index = 0
            self.DB_ID = pd.DataFrame(columns=['article_doi', 'Title'], dtype=object)

        
    def update(self):
        
        print('( Function: update )')
        
        print('Atualizando o DB...')
        file_list = os.listdir(self.diretorio + '/Articles_to_add')

        if len(file_list) == 0:
            print('\nNão existem arquivos na pasta ~/Articles_to_add para atualizar o DB.\nAbortando a função...')
            return

        articleID_dict_temp = {} #dic de identificação dos arquivos dos documentos
        article_doi = {} #dic article_doi
        article_title = {} #dic título do artigo
        article_language = {} #dic da língua do artigo
        article_address = {} #dic do endereço do artigo
        article_year = {} #dic do ano do artigo
        file_error_list = [] #lista de artigos que não foram abertos
        counterfiles = 0
        
        #contadores de artigos extraidos e não extraidos
        counter_extracted = 0
        counter_n_extracted = 0
        for filename in sorted(file_list):

            if self.article_file_type.lower() == 'pdf':
                
                #checando o basename do arquivo. não pode ser 'ATC'
                checked_filename = self.check_rename_files(filename, file_extension = 'pdf')
                
                PDF = PDFTEXT(f'{checked_filename[:-4]}', 
                            folder_name = 'Articles_to_add', 
                            language = 'english',
                            min_len_text = 10000,
                            diretorio=self.diretorio)
                PDF.process_document(apply_filters = False)
                PDF.find_meta_data()
            
                #checando se há erro na extração do PDF
                if PDF.proc_error_type is None:
                    
                    #aqui o ID é um código calculado em função do número de palavras do PDF convertido para plain text
                    articleID_dict_temp[f'{checked_filename}'] = PDF.PDF_ID
                    article_doi[f'{checked_filename}'] = PDF.doi
                    article_title[f'{checked_filename}'] = PDF.title
                    article_language[f'{checked_filename}'] = 'English'
                    article_address[f'{checked_filename}'] = str( PDF.countries_found ) #os paises estão em uma lista
                    article_year[f'{checked_filename}'] = PDF.publication_date
                    
                    counter_extracted += 1
                    counterfiles += 1
                    print(f'Summary - PDF extracted: {counter_extracted} ; PDF non-extracted: {counter_n_extracted}')
                
                else:
                    file_error_list.append([filename, PDF.proc_error_type])
                    #print('Exceção de character encontrado: ', repr(string_text[char_index]))
                    proc_error_file_name = self.rename_proc_error_files(filename, file_extension = 'pdf')
                    save_text_to_TXT(PDF.filtered_text, '_' + PDF.proc_error_type + '_' + proc_error_file_name, diretorio=self.diretorio)                
                    counter_n_extracted += 1
                    counterfiles += 1
                    print(f'Summary - PDF extracted: {counter_extracted} ; PDF non-extracted: {counter_n_extracted}')
                    continue

            elif self.article_file_type.lower() == 'xml':
                
                #checando o basename do arquivo. não pode ser 'ATC'
                checked_filename = self.check_rename_files(filename, file_extension = 'xml')
                
                XML = XMLTEXT(f'{checked_filename[:-4]}', 
                              folder_name = 'Articles_to_add', 
                              language = 'english',
                              min_len_text = 10000,
                              diretorio=self.diretorio)
                XML.process_document(apply_filters = False)
                XML.find_meta_data()

                #checando se há erro na extração do XML
                if XML.proc_error_type is None:
                    
                    #aqui o ID é um código calculado em função do número de palavras do XML convertido para plain text
                    articleID_dict_temp[f'{checked_filename}'] = XML.XML_ID

                    article_doi[f'{checked_filename}'] = XML.doi
                    article_title[f'{checked_filename}'] = XML.title
                    article_language[f'{checked_filename}'] = 'English'
                    article_address[f'{checked_filename}'] =  str( XML.countries_found ) #os paises estão em uma lista
                    article_year[f'{checked_filename}'] = XML.publication_date
                    
                    counter_extracted += 1
                    counterfiles += 1
                    print(f'Summary - XML extracted: {counter_extracted} ; XML non-extracted: {counter_n_extracted}')
                else:
                    file_error_list.append([filename, XML.proc_error_type])
                    #print('Exceção de character encontrado: ', repr(string_text[char_index]))
                    proc_error_file_name = self.rename_proc_error_files(filename, file_extension = 'xml')
                    save_text_to_TXT(XML.filtered_text, '_' + XML.proc_error_type + '_' + proc_error_file_name, diretorio=self.diretorio)                
                    counter_n_extracted += 1
                    counterfiles += 1
                    print(f'Summary - XML extracted: {counter_extracted} ; XML non-extracted: {counter_n_extracted}')
                    continue

            elif self.article_file_type.lower() == 'webscience_csv_report':

                #abrindo o arquivo csv com o report
                wbsci_report = pd.read_csv(self.diretorio + '/Articles_to_add/' + filename, index_col=0)
                
                #varrendo os indexes da DF
                indexes_to_collect = []
                for i in wbsci_report.index:    
                    
                    #condições para coletar o artigo
                    cond1 = type(wbsci_report.loc[ i, 'DOI' ]) == str and len(wbsci_report.loc[ i, 'DOI' ]) > 0
                    cond2 = type(wbsci_report.loc[ i, 'Abstract' ]) == str
                    cond3 = 'retracted' not in wbsci_report.loc[ i, 'Article Title' ].lower()
                    cond4 = wbsci_report.loc[ i, 'Language' ].lower() == 'english'
                    if all((cond1, cond2, cond3, cond4)) == True:
                        
                        indexes_to_collect.append(i)
                        counter_extracted += 1
                        counterfiles += 1
                        print(f'Summary - Abstracts extracted: {counter_extracted} ; Abstracts non-extracted: {counter_n_extracted}')
                    
                    else:
                        counter_n_extracted += 1
                        counterfiles += 1
                        print(f'Summary - Abstracts extracted: {counter_extracted} ; Abstracts non-extracted: {counter_n_extracted}')
                
                #filtrando os indexes a coletar da DF
                wbsci_report_filtered = wbsci_report.iloc[indexes_to_collect]
                wbsci_report_filtered.reset_index( drop=True, inplace=True )
                
                article_title = dict( zip( wbsci_report_filtered.index, wbsci_report_filtered['Article Title'].values ))
                articleID_dict_temp = dict( zip( wbsci_report_filtered.index, wbsci_report_filtered['DOI'].values ))
                #para o web of science report o ID é o DOI
                article_doi = articleID_dict_temp

        print('Total: ', counterfiles)
        
        #caso haja artigos para tentar adicionar do DB
        if len(articleID_dict_temp) > 0:
        
            #adicionado os números índices dos arquivos a serem adicionados no DataBase
            files_to_add = [] #essa lista contem o número índice do artigo a ser adicionado no folder DataBase
            article_ID_added = [] #essa lista contem o article_ID do documento que será adicionado. Essa lista é para que não haja duplicatas na adição

            #checando duplicatas
            list_docs_filenames, list_docs_ID = zip( *list(articleID_dict_temp.items()) )
            for ID_N in range(len(list_docs_ID)):
                cond1 = list_docs_ID[ID_N] not in self.DB_ID['article_doi'].values
                cond2 = list_docs_ID[ID_N] not in article_ID_added
                if all([cond1, cond2]) == True:
                    files_to_add.append(list_docs_filenames[ID_N])
                    article_ID_added.append(list_docs_ID[ID_N])
                
            #adicionando os novos arquivos no folder 'DB' e o article_ID no DataBase
            tag_list = range(self.last_index, len(files_to_add) + self.last_index)

            if len(files_to_add) > 0:
                for file_N in range(len(files_to_add)):
                    
                    tag_name = get_tag_name(tag_list[file_N])

                    #colocando o artigo na pasta DB
                    file_name = files_to_add[file_N]
                    if self.article_file_type.lower() == 'pdf':
                        self.rename_move_save_files(file_name, tag_name, file_extension = 'pdf')
                        self.DB_ID.loc[tag_name, 'article_doi'] =  article_doi[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Title'] = article_title[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Addresses'] = article_address[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Publication Year'] = int( article_year[ files_to_add[file_N] ] )
                        self.DB_ID.loc[tag_name, 'Language'] = article_language[ files_to_add[file_N] ]

                    elif self.article_file_type.lower() == 'xml':
                        self.rename_move_save_files(file_name, tag_name, file_extension = 'xml')
                        self.DB_ID.loc[tag_name, 'article_doi'] =  article_doi[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Title'] = article_title[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Addresses'] = article_address[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Publication Year'] = int( article_year[ files_to_add[file_N] ] )
                        self.DB_ID.loc[tag_name, 'Language'] = article_language[ files_to_add[file_N] ]

                    elif self.article_file_type.lower() == 'webscience_csv_report':
                        #coletando o endereço e o ano de publicação e o abstract
                        self.DB_ID.loc[tag_name, 'article_doi'] =  article_doi[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Title'] = article_title[ files_to_add[file_N] ]
                        self.DB_ID.loc[tag_name, 'Addresses'] = wbsci_report_filtered.loc[ file_N, 'Addresses']
                        self.DB_ID.loc[tag_name, 'Publication Year'] = int( wbsci_report_filtered.loc[ file_N, 'Publication Year'] )
                        self.DB_ID.loc[tag_name, 'Language'] = wbsci_report_filtered.loc[ file_N, 'Language']
                        
                        #exportando o abstract para txt
                        abstract = wbsci_report_filtered.loc[ file_N, 'Abstract']
                        self.export_str_to_file(abstract, tag_name, file_extension = 'txt')
                
                self.DB_ID.to_csv(self.diretorio + '/Outputs/DB_ID.csv')
        
        print()
        print(self.DB_ID)
            
        print('Error report')
        print('--------------------------------------------------')
        print('Os arquivos seguintes não puderam ser processados:')
        for filename in file_error_list:
            print(filename)
        print('Total: ', len(file_error_list))
        print('--------------------------------------------------')

            
    def rename_move_save_files(self, filename, tag_name, file_extension = ''):
        
        print('( Function: rename_move_save_files )')
        
        old_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{filename}')
        new_file_name = os.path.join(self.diretorio + '/DB', f'{tag_name}.{file_extension}')
        os.rename(old_file_name, new_file_name)
        print(f'Arquivo: {filename} renomeado para {tag_name}.{file_extension} e movido com sucesso para a pasta ~/DB.')
    

    def export_str_to_file(self, text, tag_name, file_extension = ''):

        if file_extension.lower() == 'txt':
            print('Saving txt file ' + tag_name + ' ...' )
            with open(self.diretorio + '/DB/' + f'{tag_name}.{file_extension}', 'w') as doc:
                doc.write(text)
                doc.close()

        
    def check_rename_files(self, filename, file_extension = ''):
        
        #print('( Function: check_rename_files )')
        
        #o prefixo ATC (de article) será usado para nomear os documetnos na pasta /DB
        if filename[ : 3] == 'ATC':        
            old_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{filename}')
            new_filename = filename_gen()
            new_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{new_filename}.{file_extension}')
            os.rename(old_file_name, new_file_name)
            #print(old_file_name, new_file_name)
            print('Basename de arquivo incompatível')
            print(f'{filename} renomeado para {new_filename}.{file_extension}')
            return f'{new_filename}.{file_extension}'
        else:
            return filename


    def rename_proc_error_files(self, filename, file_extension = ''):
        
        #print('( Function: rename_proc_error_files )')
        
        old_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{filename}')
        new_filename = filename_gen()
        new_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'_Not_processed_{new_filename}.{file_extension}')
        os.rename(old_file_name, new_file_name)
        return f'_Not_processed_{new_filename}.{file_extension}'
