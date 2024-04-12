#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functions_TEXTS import break_text_in_sentences
from bs4 import BeautifulSoup as bs
from bs4 import element
from functions_TEXTS import filter_chars
from functions_TEXTS import check_text_language
from functions_TEXTS import break_text_in_sentences
import regex as re
import time


class XMLTEXT:

    def __init__(self, xml_file_name, folder_name = 'Articles_to_add', language = 'english', min_len_text = 1000, diretorio = None):

        print('\n( Class: XMLTEXT )')

        self.diretorio = diretorio
        self.xml_file_name = xml_file_name
        self.filtered_text = ''
        self.folder_name = folder_name
        self.match_category = True
        self.language = language
        self.min_len_text = min_len_text
        self.proc_error_type = None


    def process_document(self, apply_filters = True):
        
        #print('\n( Function: process_text )')

        
        print(f'\nProcessing: {self.xml_file_name}...')
        with open(self.diretorio + '/' + self.folder_name + '/' + self.xml_file_name + '.xml', 'r', encoding='utf-8') as file:
            self.soup_xml = bs(file, 'lxml-xml', from_encoding="utf-8")
            file.close()

        #texto raw
        self.raw_text = ''

        #varrendo as seções
        sections_tag = self.soup_xml.find_all('sec')
        last_section_name = 'None'
        print('Looking for sections division')
        for sec in sections_tag:
            try:
                #checando se o tag não é de uma subseção
                if re.search(r'[0-9](?=\.[0-9])', sec.attrs['id']) is None:
                
                    #coletando o nome da seção
                    section_name = sec.title.get_text()

                    #a seção de introdução sempre aparece como INTRODUCION
                    if 'introduction' in section_name.lower():
                        section_name = 'introduction'
                    
                    #a parte de resultados e discussão pode vir misturada
                    elif ( 'results' in section_name.lower() ) and ( 'discussion' in section_name.lower() ):
                        section_name = 'results-discussion'
                    
                    #quando os resultados vem separado
                    elif 'results' in section_name.lower():
                        section_name = 'results'
                    
                    #quando a discussão vem separada
                    elif 'discussion' in section_name.lower():
                        section_name = 'discussion'

                    #quando a conclusão vem separada
                    elif 'conclusion' in section_name.lower():
                        section_name = 'conclusion'
                    
                    #a parte de methodologia tem muitas definições diferentes na coleção
                    else:
                        section_name = 'methodology'

                    #caso não seja a mesma seção
                    if section_name != last_section_name:
                        
                        #caso esteja fechando uma seção
                        if last_section_name != 'None':               
                            self.raw_text += f'The END of the section is there (separated from the XML file). '
                            print('  Closing section: ', last_section_name)

                        print('> Section', sec.attrs['id'], sec.title.get_text())
                        print('  New section name given: ', section_name, ' ; last_section_name:', last_section_name)
                        #print('Concatenating section to text: ', section_name)
                        self.raw_text += f'The BEGIN of the section is there (separated from the XML file) {section_name}. '
            
            #se não for possível definir o nome da seção, passa-se para a próxima
            except (AttributeError, KeyError):
                continue
            
            #varrendo os parágrafos
            paragraphs = sec.find_all('p')
            for p in paragraphs:
                self.raw_text += p.get_text() + ' '

            last_section_name = section_name
        
        self.raw_text += f'The END of the section is there (separated from the XML file).'
        print('  Closing section: ', last_section_name)
        
        #print((self.raw_text))
        #texto filtrado 1 X        
        if apply_filters is True:
            self.filtered_text, self.len_text = filter_chars(self.raw_text, diretorio = self.diretorio, filename = self.xml_file_name)
        else:
            self.filtered_text = self.raw_text
            self.len_text = len(self.raw_text)
        
        #checando a lingua
        if check_text_language( self.raw_text , self.language) is False:
            self.proc_error_type = 'Not_english'
            print(f'O Arquivo {self.xml_file_name} não bate com a língua determinada ({self.language}).')
            print('> Abortando função: XMLTEXT.process_text')
            return
        
        #checando o tamanho mínimo do texto
        if len( self.raw_text ) < self.min_len_text:
            self.proc_error_type = 'Not_min_length'
            print(f'O Arquivo {self.xml_file_name} extraído não possui o tamanho mínimo (self.min_len_text).')
            print('> Abortando função: XMLTEXT.process_text')
            return
        
        #quebrando o texto em sentenças
        self.sentences = break_text_in_sentences(self.filtered_text)        
        self.token_list = self.filtered_text.split()
        self.n_tokens = len(self.token_list)
        #usando o número de chars + o de tokens
        self.XML_ID =  str(self.len_text) + '_' + str(self.n_tokens)
        print('Char counter: ', self.len_text)
        print('Token counter: ', self.n_tokens)


    def find_country(self):
        
        countries_found = []
        
        affs = self.soup_xml.front.find_all('aff')
        for aff in affs:
            try:
                countries_found.append( aff.get_text() )
            except AttributeError:
                continue

        self.countries_found = countries_found


    def find_meta_data(self):
        
        self.find_country()

        #o título do artigo geralmente vem com "\n" no texto
        self.title = re.sub(r'\n', '', self.soup_xml.find('article-title').get_text() )
        self.doi = self.soup_xml.find('article-id').get_text()
        self.publication_date = self.soup_xml.find('pub-date').year.get_text()

        print('Doc Year: ', self.publication_date, '; DOI: ', self.doi)