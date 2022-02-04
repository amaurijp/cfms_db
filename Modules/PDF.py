#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class PDFTEXT(object):
    
    def __init__(self, pdf_file_name, folder_name = 'Articles', language = 'english', replace_numbers = False, min_len_text = 10000, text_break_function = 'function_text', diretorio = None):

        print('\n=============================\nPDF...\n=============================\n')
        print('( Class: PDFTEXT )')
        
        #import PyPDF2 o PyPDF2 foi substituido pelo tika
        import os        
        self.diretorio = diretorio
        self.pdf_file_name = pdf_file_name
        self.filtered_text = ''
        self.folder_name = folder_name
        self.text_break_function = text_break_function.lower()
        self.match_category = True
        self.replace_numbers = replace_numbers
        self.language = language
        self.min_len_text = min_len_text
        self.proc_error_type = None


    def check_category(self, category):
        
        #print('\n( Function: check_category )')
        
        import os
        import pandas as pd
        if os.path.exists(self.diretorio + '/Outputs/TFIDF/classification_Articles_Categorized.csv'):
            DF_categories = pd.read_csv(self.diretorio + '/Outputs/TFIDF/classification_Articles_Categorized.csv', index_col=0)         
        else:
            print('DF de Categorias não encontrado (~/Outputs/TFIDF/classification_Articles_Categorized.csv)...')
            print('Rodar o modulo Classifier')
            return
        if DF_categories.loc[self.pdf_file_name, 'Category'] != category:        
            self.match_category = False
    
    
    def process_text(self, apply_filters = True, min_tokens_per_sent = 2, min_tokens_to_break_sent = 2):
        
        #print('\n( Function: process_text )')

        from tika import parser
        from functions_TEXTS import filter_chars
        from functions_TEXTS import process_sentences
        from functions_PARAMETERS import replace_numerical_parameters
        from functions_TEXTS import check_text_language
        #from FUNCTIONS import correct_period_and_spaces
    
        #PyPDF não está sendo utilizado mais (foi substituido pelo tika-parser)
        '''
        self.char_counter = 0 #contador de caracteres válidos
        
        #extraindo o texto
        with open(self.diretorio + '/' + folder_name + '/' + pdf_file_name + '.pdf', 'rb') as pdf_file_rb:
            #coletando metadados com pypdf
            self.pdf = PyPDF2.PdfFileReader(pdf_file_rb)            
            info = self.pdf.getDocumentInfo()
            self.PDFinfo = info
            #print('PDF Meta Dados:', info)
            number_of_pages = self.pdf.numPages
            for page_number in range(number_of_pages):
                page_objt = self.pdf.getPage(page_number)                
                processed_text = remove_special_chars(page_objt.extractText())
                self.filtered_text += processed_text[0]
                self.char_counter += processed_text[1]
            pdf_file_rb.close()
        
        self.len_text = len(self.filtered_text)
        self.PDF_ID = self.filtered_text[900:950] #usando 30 caracteres para o ID
        #print('Char counter: ', self.char_counter)
        '''
        
        print(f'\nProcessing: {self.pdf_file_name}...')
        
        #extraindo o texto com o tika parser
        try: 
            self.pdf = parser.from_file(self.diretorio + '/' + self.folder_name + '/' + self.pdf_file_name + '.pdf')
        except FileNotFoundError:
            try: 
                self.pdf = parser.from_file(self.diretorio + '/' + self.folder_name + '/' + self.pdf_file_name + '.PDF')
            except FileNotFoundError:
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
            self.filtered_text, self.len_text = filter_chars(self.raw_text, diretorio = self.diretorio)
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
        #texto com parametros numéricos substituidos para encontrar as sentenças (logo abaixo)
        if ( self.replace_numbers is True ):
            self.filtered_text_replaced = replace_numerical_parameters(self.filtered_text, char_range_number = 20, check = False)
        
        '''
        #texto filtrado 2 X (para salvar)
        self.final_text = correct_period_and_spaces(self.filtered_text)
        #texto filtrado 2 X com com parametros numéricos substituidos (para salvar e extrair tokens)
        self.final_text_replaced = replace_numerical_parameters(self.final_text, char_range_number = 20, check = False)
        '''
        
        #tipos de tokenizers
        if (self.text_break_function == 'spacy'):
            import spacy
            nlp = spacy.load('en_core_web_sm')
            #quebrando o texto em sentenças
            text_nlp1 = nlp(self.filtered_text)
            #filtrando as sentenças
            self.sentences = process_sentences(text_nlp1.sents, self.text_break_function, min_tokens_per_sent = min_tokens_per_sent)
            if ( self.replace_numbers is True ):
                text_nlp2 = nlp(self.filtered_text_replaced)
                self.sentences_replaced = process_sentences(text_nlp2.sents, self.text_break_function, min_tokens_per_sent = min_tokens_per_sent)
        elif (self.text_break_function == 'function_text'):
            #quebrando o texto em sentenças
            from functions_TEXTS import break_text_in_sentences
            text_broken1 = break_text_in_sentences(self.filtered_text, min_tokens_to_break_sent = min_tokens_to_break_sent)
            #filtrando as sentenças
            self.sentences = process_sentences(text_broken1, self.text_break_function, min_tokens_per_sent = min_tokens_per_sent)
            if ( self.replace_numbers is True ):
                text_broken2 = break_text_in_sentences(self.filtered_text_replaced, min_tokens_to_break_sent = min_tokens_to_break_sent)
                self.sentences_replaced = process_sentences(text_broken2, self.text_break_function, min_tokens_per_sent = min_tokens_per_sent)
        else:
            self.sentences = []        

        self.token_list = self.filtered_text.split()
        self.n_tokens = len(self.token_list)
        #usando o número de chars + o de tokens
        self.PDF_ID =  str(self.len_text) + '_' + str(self.n_tokens)
        print('Char counter: ', self.len_text)
        print('Token counter: ', self.n_tokens)
                    

    #------------------------------
    def find_class_TF(self, TF_term_list):
                
        import regex as re
        
        #print('\n( Function: find_class_TF )')
        
        self.class_TF = {}
        for term in TF_term_list:
            find_counter = len( re.findall( term , self.filtered_text ) ) 
            #print('Foram encontrados ', find_counter, ' para o termo ', TF_term_list)
            self.class_TF = find_counter / self.n_tokens
            #print('TF: ', self.class_TF)


    #------------------------------
    def find_TFIDF(self, term_list):

        #print('\n( Function: find_TFIDF )')
                
        self.TF , self.check_term_existence = find_term_in_text(self.filtered_text, term_list, self.n_tokens)

        
    #------------------------------
    def extract_fragments(self, text_len_to_extract = 20):

        #print('\n( Function: extract_fragments )')
        
        import regex as re
        import numpy as np
        import pandas as pd
        #import time
        import os        
        from FUNCTIONS import generate_term_list
        from FUNCTIONS import split_operation_term_list
        from FUNCTIONS import create_PDFFILE_list
        
        if (self.replace_numbers is True):
            full_text_extracted = self.final_text_replaced
        else:
            full_text_extracted = self.filtered_text
        
        #checar se o PDF está na categoria
        if (self.match_category is True):
            
            #gerando a lista de termos a serem procurados nos PDFs
            with open(self.diretorio + '/Settings/search_in_text.txt','r') as search_terms_file:
                term_list = generate_term_list(search_terms_file)
                search_terms_file.close()
            #print(term_list)
    
            #checando se esse PDF já foi extraido seus fragmentos
            extracted_PDF_list = []
            #checar a existência de DF de fragmentos
            ID_columns = ['Filename', 'Index']
            if os.path.exists(self.diretorio + '/Outputs/Extracted/frags_extracted_from_fulltext.csv'):
                DF_existent = pd.read_csv(self.diretorio + '/Outputs/Extracted/frags_extracted_from_fulltext.csv', index_col=[0,1])  
                last_pdf_file_processed = DF_existent.index.levels[0].values[-1]
                extracted_PDF_list = create_PDFFILE_list(last_pdf_file_processed)
                
            else:
                #criando a coluna de indexação do DF de fragmentos
                columns = ID_columns + term_list
                DF_existent = pd.DataFrame(columns=columns)
                DF_existent.set_index(['Filename', 'Index'], inplace=True)
            
            #checar se o PDF já não teve os termos extraidos
            if self.pdf_file_name not in extracted_PDF_list:  
                
                #separando os termos a partir do arquivo TXT
                term_list_dic = split_operation_term_list(term_list)
    
                #print()
                print('Prim. terms: ', term_list_dic['primary'])
                print('Sec. terms: ', term_list_dic['secondary'])
                print('Sec. terms operations: ', term_list_dic['operations'])
                print('Regex patterns: ', term_list_dic['regex'])
                #print()            
    
                DF_line_number = 0 #número de linhas (começa com zero)
                indexed_results = {}
                results_counter = 1            
                            
                #checa se todas os termos de procura (linhas do arquivo TXT) foram encontrados no documento
                columns_terms_found = 0 
                            
                #cada term_set_N será uma coluna que será adicionada no DataFrame
                for term_set_N in range(len(term_list_dic['primary'])):
    
                    #numero da linha da entrada (as linhas separam o número de vezes que os termos foram encontrados no artigo)
                    line = 0
                    #numero da coluna da entrada (as colunas separam os termos que foram procurados)
                    column = term_set_N + len(ID_columns)
                                                   
                    #checando se há termos primários
                    if len(term_list_dic['primary'][term_set_N][0]) > 0:
                        #print(term_list_dic['primary'][term_set_N])                                                            
                        #procurando pelos termos primários no texto
                        for term_to_search in term_list_dic['primary'][term_set_N]:
                            #procurando os termos exatos no texto
                            matches = re.finditer( term_to_search , full_text_extracted )
                            for match in matches:
                                #se foi encontrado algum termo primário
                                if match.group:
                                    #print('Encontrado o termo primario: ', term_to_search)
                                    match_start = match.span()[0]
                                    match_end = match.span()[1]
                                    indexes_found = [match_start , match_end]
                                    #parte do texto extraida para cada match
                                    extrated_part = full_text_extracted[ match_start - text_len_to_extract : match_end + text_len_to_extract ]                                    
                                    #dicionário para coletar as operações + e - feitas com os termos secundários
                                    check_sec_term_operation = {}
                                    #procurando os termos secundários                                
                                    for j in range(len(term_list_dic['secondary'][term_set_N])):
                                        #assumindo primariamente que nenhum termo foi encontrado para esse conjunto de termos secundários [j]
                                        check_sec_term_operation[j] = '-'
                                        #procurando cada termo secundário dentro do grupo de secundários
                                        for sec_term in term_list_dic['secondary'][term_set_N][j]:
                                            if re.search(sec_term , extrated_part):
                                                #print('Encontrado o termo secundário: ', sec_term)
                                                check_sec_term_operation[j] = '+'
                                                break
                                    
                                    #caso as operações de termos secundários sejam as mesmas inseridas
                                    if list(check_sec_term_operation.values()) == term_list_dic['operations']:
                                        #testado se tem expressão regex para encontrar
                                        if len(term_list_dic['regex'][term_set_N]) > 0:
                                            terms_match_regex = re.search(term_list_dic['regex'][term_set_N], extrated_part)
                                            if terms_match_regex:
                                                indexed_results[results_counter] = ([line, column], indexes_found, extrated_part)
                                                line += 1
                                                results_counter +=1                    
                                                print('*TERMS FOUND (Com Padrão REGEX)*')
                                                #print('Extracted text:')
                                                #print('*** ', repr(extrated_part), '***')
                                                #print('Indexes collected: ', indexes_found)
                                        
                                        else:    
                                            indexed_results[results_counter] = ([line, column], indexes_found, extrated_part)
                                            line += 1
                                            results_counter +=1                    
                                            print('*TERMS FOUND*')
                                            #print('Extracted text:')
                                            #print('*** ', repr(extrated_part), '***')
                                            #print('Indexes collected: ', indexes_found)
                
                    #caso não tenha termos primários, mas só o padrão regex para testar
                    else:
                        if len(term_list_dic['regex'][term_set_N]) > 0: #testado se tem expressão regex para encontrar
                            matches = re.finditer( term_list_dic['regex'][term_set_N] , full_text_extracted )
                            for match in matches:
                                #se foi encontrado algum regex
                                if match.group:
                                    #print('Encontrado o termo primario: ', term_to_search)
                                    match_start = match.span()[0]
                                    match_end = match.span()[1]                                
                                    indexes_found = [match_start , match_end]
                                    #parte do texto extraida para cada match
                                    extrated_part = full_text_extracted[ match_start - text_len_to_extract : match_end + text_len_to_extract ]                                        
                                    indexed_results[results_counter] = ([line, column], indexes_found, extrated_part)
                                    line += 1
                                    results_counter +=1                    
                                    print('*TERMS FOUND (Com Padrão REGEX)*')
                                    #print('Extracted text:')
                                    #print('*** ', repr(extrated_part), '***')
                                    #print('Indexes collected: ', indexes_found)
                                            
                        else:
                            print('Erro! Fornecer o padrão REGEX após o indentificador & (ex: & ([0-9]{3,4})[°]*C )')
                            return
                                            
                    #se o termo foi encontrado, o line será maior que o found_column_term (que foi igualado ao line acima)
                    if line > columns_terms_found:
                        columns_terms_found += 1
                    
                    #definindo o número de linhas e colunas que formarão o DATAFRAME
                    if line > DF_line_number:
                        DF_line_number = line
                    DF_column_number = term_set_N + len(ID_columns) + 1
    
                #checando se todas as linhas do Terms_Search foram encontradas no arquivo PDF
                if columns_terms_found == len(term_list_dic['primary']):
                    print('*ALL TERMS FOUND*')
                    #adicionando as entradas na numpy array
                    extracted_text_len = 2*text_len_to_extract + int((2*text_len_to_extract)*0.2)
                    array = np.zeros([DF_line_number, DF_column_number], dtype=np.dtype(f'U{extracted_text_len}'))
                    for results_found in sorted(indexed_results.keys()):
                        pos_X = indexed_results[results_found][0][0]
                        pos_Y = indexed_results[results_found][0][1]
                        #print(indexed_results[results_found][2])
                        array[pos_X, pos_Y] = repr(indexed_results[results_found][2])
                    #adicionando o titulo e o DOI no numpy array
                    for line in range(array.shape[0]):
                        array[line, 0] = self.pdf_file_name
                        array[line, 1] = line
                    #transformando a array em dataframe
                    DF_to_concat = pd.DataFrame(array, columns=ID_columns + term_list)
                    DF_to_concat.set_index(['Filename', 'Index'], inplace=True)
                    #concatenando a DF
                    DF_concatenated = pd.concat([DF_existent, DF_to_concat])
                    #salvando a DataFram em CSV
                    DF_concatenated.to_csv(self.diretorio + '/Outputs/Extracted/frags_extracted_from_fulltext.csv')
    
                else:
                    return
    
            else:
                print(f'{self.pdf_file_name} já teve os fragmentos extraidos.')
                return

        else:
            print(f'{self.pdf_file_name} não é da categoria inserida.')
            return
    
    
    #------------------------------
    def find_country(self, country_list):
    
        #import time
        import regex as re
        
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
        
        return countries_found
    
    #------------------------------
    def find_meta_data(self):
        
        #print('\n( Function: find_meta_data )')
        
        from functions_TEXTS import exist_term_in_string
        
        self.title = 'Not Found'
        self.doi = ''
        self.publication_date = 'Not Found'

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
    

#------------------------------------------------
def find_term_in_text(text, term_list, n_tokens):
    
    #print('\n( Function: find_term_in_text )')
    import regex as re
    
    #definindo o modo que a função irá operar
    try:
        len(term_list[0][0])
        mode = 'classify'
    except TypeError:
        mode = 'normal'
    
    #O mode 'classify' usa um conjunto de termos dados pelo arquivo ~/Outputs/TFIDF/Terms_to_classify_Docs.txt
    #para classificar os documentos existentes. Nesse modo, a lista 'term_list' é uma lista de lista
    if mode == 'classify':
        TF = {}
        check_term_existence = {}
        for term_N in range(len(term_list)):        
            #print('Procurando: ', term_list[term_N])
            find_counter = 0
            for term in term_list[term_N]:
                #procurando o termo no texto
                find_counter = len( re.findall( term , text ) )
            
            print('Matches: ', find_counter, ' para o termo ', term_list[term_N])
            if find_counter >= 1:
                check_term_existence[str(term_list[term_N])] = True
            else:
                check_term_existence[str(term_list[term_N])] = False
            TF[str(term_list[term_N])] = find_counter / n_tokens

    #O mode 'normal' procura os valores de frequencia de termo (TF) e a existência dos termos na lista simples 'term_list'.
    #Nesse modo, a 'term_list' é a lista de tokens do corpo de documentos (considerando todos os documentos)
    if mode == 'normal':
        TF = {}
        check_term_existence = {}
        for term in term_list[term_N]:
            find_counter = 0
            #procurando o termo no texto
            find_counter = len( re.findall( term , text ) )
            
            print('Foram encontrados ', find_counter, ' para o termo ', term_list[term_N])
            if find_counter >= 1:
                check_term_existence[term_list[term_N]] = True
            else:
                check_term_existence[term_list[term_N]] = False
            TF[term_list[term_N]] = find_counter / n_tokens
                
    return TF , check_term_existence