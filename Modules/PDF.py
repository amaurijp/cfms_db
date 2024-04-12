#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import time
import os
import regex as re

from tika import parser

from requests.exceptions import ReadTimeout

from functions_TEXTS import filter_chars
from functions_TEXTS import check_text_language
from functions_TEXTS import exist_term_in_string
from functions_TEXTS import break_text_in_sentences


class PDFTEXT(object):
    
    def __init__(self, pdf_file_name, folder_name = 'Articles_to_add', language = 'english', min_len_text = 1000, diretorio = None):

        print('\n( Class: PDFTEXT )')
        
        self.diretorio = diretorio
        self.pdf_file_name = pdf_file_name
        self.filtered_text = ''
        self.folder_name = folder_name
        self.match_category = True
        self.language = language
        self.min_len_text = min_len_text
        self.proc_error_type = None
    
    
    def process_document(self, apply_filters = True):
        
        #print('\n( Function: process_text )')
        
        print(f'\nProcessing: {self.pdf_file_name}...')
        
        #extraindo o texto com o tika parser
        try: 
            self.pdf = parser.from_file(self.diretorio + '/' + self.folder_name + '/' + self.pdf_file_name + '.pdf')
        except (FileNotFoundError, ReadTimeout):
            try: 
                self.pdf = parser.from_file(self.diretorio + '/' + self.folder_name + '/' + self.pdf_file_name + '.PDF')
            except (FileNotFoundError, ReadTimeout):
                self.proc_error_type = 'Not_found_file'
                print(f'O Arquivo {self.pdf_file_name} não foi encontrado.')
                print('> Abortando função: PDFTEXT.process_text')
                return                       
            
        #coletando metadados com pypdf
        self.PDFinfo = self.pdf['metadata']
        #print('PDF Meta Dados:', self.PDFinfo)                
        #texto raw
        self.raw_text = self.pdf['content']
        #testando se o texto foi extraído
        try:
            len(self.raw_text)
        except TypeError:
            self.proc_error_type = 'Not_char_extracted'
            print(f'O Arquivo {self.pdf_file_name} não teve o texto extraído.')
            print('> Abortando função: PDFTEXT.process_text')
            return        
        
        #texto filtrado 1 X        
        if apply_filters is True:
            self.filtered_text, self.len_text = filter_chars(self.raw_text, diretorio = self.diretorio, filename = self.pdf_file_name)
        else:
            self.filtered_text = self.raw_text
            self.len_text = len(self.raw_text)
        
        #checando a lingua
        if check_text_language( self.raw_text , self.language) is False:
            self.proc_error_type = 'Not_english'
            print(f'O Arquivo {self.pdf_file_name} não bate com a língua determinada ({self.language}).')
            print('> Abortando função: PDFTEXT.process_text')
            return
        
        #checando o tamanho mínimo do texto
        if len( self.raw_text ) < self.min_len_text:
            self.proc_error_type = 'Not_min_length'
            print(f'O Arquivo {self.pdf_file_name} extraído não possui o tamanho mínimo (self.min_len_text).')
            print('> Abortando função: PDFTEXT.process_text')
            return            
        
        #quebrando o texto em sentenças
        self.sentences = break_text_in_sentences(self.filtered_text)
        self.token_list = self.filtered_text.split()
        self.n_tokens = len(self.token_list)
        #usando o número de chars + o de tokens
        self.PDF_ID =  str(self.len_text) + '_' + str(self.n_tokens)
        print('Char counter: ', self.len_text)
        print('Token counter: ', self.n_tokens)
    
    
    #------------------------------
    def find_country(self):
        
        country_list = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Arab Emirates', 
                        'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 
                        'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 
                        'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cote dIvoire', 'Cabo Verde', 
                        'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 
                        'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Czech Republic', 
                        'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 
                        'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 
                        'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 
                        'Guinea', 'Guinea-Bissau', 'Guinea Bissau', 'Guyana', 'Haiti', 'Holy See', 'Honduras', 'Hungary', 
                        'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 
                        'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 
                        'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 
                        'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 
                        'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 
                        'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 
                        'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine State', 'Panama', 'Papua New Guinea', 
                        'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 
                        'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 
                        'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 
                        'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan', 
                        'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 
                        'Tanzania', 'Thailand', 'Timor Leste', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 
                        'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'UK', 
                        'U.K.', 'United States of America', 'United States', 'US', 'U.S.', 'USA', 'U.S.A.', 
                        'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']
        
        countries_found = []
        for country_name in country_list:
            match = re.search(country_name, self.filtered_text[ : 6000])
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
        
        self.countries_found = countries_found
    

    #------------------------------
    def find_meta_data(self):
        
        #print('\n( Function: find_meta_data )')
        
        self.title = ''
        self.doi = ''
        self.publication_date = 0

        self.find_country()

        #procurando o título no METADATA
        #print(self.PDFinfo)
        for key in self.PDFinfo.keys():
            cond1 = exist_term_in_string(string = key, terms = ['title', 'name'])
            if cond1 == True:
                self.title = self.PDFinfo[key]

        #procurando a data de publicação no METADATA
        found_publication = False
        for key in self.PDFinfo.keys():
            cond1 = exist_term_in_string(string = key, terms = ['date'])
            cond2 = 'crossmark' not in key.lower()
            if cond1 == True and cond2 == True:
                #print(self.PDFinfo[key])
                for char_N in range(len(self.PDFinfo[key])):
                    try:
                        year = int(self.PDFinfo[key][char_N : char_N + 4 ])
                        if year in range(2000, 2020+1, 1):
                            self.publication_date = self.PDFinfo[key][char_N : char_N + 4 ]
                            found_publication = True
                            break
                    except:
                        continue

                if (found_publication is True):
                    break
                else:
                    continue

        found_meta_DOI = False        
        #procurando o DOI no METADATA
        for key in self.PDFinfo.keys():
            cond1 = exist_term_in_string(string = key, terms = ['doi'])
            if cond1 == True:
                self.doi = self.PDFinfo[key]
                found_meta_DOI = True
                
        if found_meta_DOI == True:
            pass
        
        #procurando o DOI no texto
        else:
            
            cond1=False
            cond2=False            
            for char_index in range(len(self.filtered_text)):        
                if self.filtered_text[char_index : char_index + len('doi') ].lower() == 'doi':
                    if self.filtered_text[char_index] == '/':
                        cond1 = True
                        bar_char = char_index
                        continue                   
                    if self.filtered_text[char_index] == '/' and cond1 == True:
                        cond2 = True
                        break
                
            if cond1 == True and cond2 == True:
                self.doi = self.filtered_text[ bar_char : bar_char + 40 ]        

        print('Doc Year: ', self.publication_date, '; DOI: ', self.doi)
        #print(self.title)