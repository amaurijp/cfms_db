#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functions_TEXTS import break_text_in_sentences
from functions_TEXTS import filter_chars
from functions_TEXTS import break_text_in_sentences
import time
import regex as re


class TXTTEXT:

    def __init__(self, txt_file_name, folder_name = 'Articles_to_add', language = 'english', min_len_text = 100, diretorio = None):

        print('\n( Class: TXTTEXT )')
        
        self.diretorio = diretorio
        self.txt_file_name = txt_file_name
        self.filtered_text = ''
        self.folder_name = folder_name
        self.match_category = True
        self.language = language
        self.min_len_text = min_len_text
        self.proc_error_type = None

    
    def process_document(self, apply_filters = True, DB_ID_df = None):
        
        #print('\n( Function: process_text )')

        print(f'\nProcessing: {self.txt_file_name}...')
        with open(self.diretorio + '/' + self.folder_name + '/' + self.txt_file_name + '.txt', 'r') as file:
            #texto raw
            self.raw_text = file.read()
            file.close()

        #print((self.raw_text))
        #texto filtrado 1 X        
        if apply_filters is True:
            self.filtered_text, self.len_text = filter_chars(self.raw_text, diretorio = self.diretorio, filename = self.txt_file_name)
        else:
            self.filtered_text = self.raw_text
            self.len_text = len(self.raw_text)
        
        #checando a lingua
        print('> ', self.txt_file_name)
        if DB_ID_df.loc[self.txt_file_name, 'Language'].lower() != 'english':
            self.proc_error_type = 'Not_english'
            print(f'O Arquivo {self.txt_file_name} não bate com a língua determinada ({self.language}).')
            print('> Abortando função: XMLTEXT.process_text')
            return

        #checando o tamanho mínimo do texto
        if len( self.raw_text ) < self.min_len_text:
            self.proc_error_type = 'Not_min_length'
            print(f'O Arquivo {self.txt_file_name} extraído não possui o tamanho mínimo (self.min_len_text).')
            print('> Abortando função: XMLTEXT.process_text')
            return
        
        #quebrando o texto em sentenças
        self.sentences = break_text_in_sentences(self.filtered_text)        
        self.token_list = self.filtered_text.split()
        self.n_tokens = len(self.token_list)
        print('Char counter: ', self.len_text)
        print('Token counter: ', self.n_tokens)


    def find_country(self, filename, DB_ID_dataframe, country_list):
        
        countries_found = []
        
        addresses = DB_ID_dataframe.loc[filename, 'Addresses']
        
        if type(addresses) == str and len(addresses) > 0:
            for country_name in country_list:
                match = re.search(country_name, addresses)
                if match:
                    if country_name in ('Arab Emirates', 'United Arab Emirates'):
                        if 'United Arab Emirates' not in countries_found:
                            countries_found.append('United Arab Emirates')                
                    elif country_name in ('Cote dIvoire', 'Ivory Coast'):
                        if 'Ivory Coast' not in countries_found:
                            countries_found.append('Ivory Coast')
                    elif country_name in ('Guinea-Bissau', 'Guinea Bissau'):
                        if 'Guinea Bissau' not in countries_found:
                            countries_found.append('Guinea Bissau')
                    elif country_name in ('Timor-Leste', 'Timor Leste'):
                        if 'Timor Leste' not in countries_found:
                            countries_found.append('Timor Leste')
                    elif country_name in ('United States of America', 'United States', 'USA', 'US', 'U\.S\.A\.', 'U\.S\.'):
                        if 'USA' not in countries_found:
                            countries_found.append('USA')
                    elif country_name in ('United Kingdom', 'UK', 'U\.K\.'):
                        if 'UK' not in countries_found:
                            countries_found.append('UK')
                    else:
                        if country_name not in countries_found:
                            countries_found.append(country_name)
            
        return countries_found


    def find_meta_data(self, filename, DB_ID_dataframe):

        self.title = DB_ID_dataframe.loc[filename, 'Title' ]
        self.doi = DB_ID_dataframe.loc[filename, 'article_ID' ]
        self.publication_date = DB_ID_dataframe.loc[filename, 'Publication Year' ]

        print('Doc Year: ', self.publication_date, '; DOI: ', self.doi)