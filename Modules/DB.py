#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DB(object):
    
    def __init__(self, diretorio = None):
        
        print('( Class: DB )')
        
        import pandas as pd
        import os  
        self.diretorio = diretorio
        
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
            self.DB_ID = pd.DataFrame(columns=['PDF_ID', 'Title'], dtype=object)

        
    def update(self, save_to_GoogleDrive = False, min_PDF_len = 5000):
        
        print('( Function: update )')
        
        print('Atualizando o DB...')
        from PDF import PDFTEXT
        import time
        import os
        from FUNCTIONS import get_tag_name
        from functions_TEXTS import save_text_to_TXT
        from GoogleDriveAPI import gDrive
        file_list = os.listdir(self.diretorio + '/Articles_to_add')
        PDFID_dict_temp = {} #dic PDF_ID
        PDF_title = {} #dic título do PDF
        file_error_list = [] #lista de PDFs que não foram abertos
        counterfiles = 0
        
        #fazendo o carregamneto no Google Drive
        if save_to_GoogleDrive == True:
            self.drive = gDrive()
        
        #contadores de PDF extraidos e não extraidos
        counter_extracted = 0
        counter_n_extracted = 0
        for filename in sorted(file_list):
            #checando o basename do arquivo. não pode ser 'PDF'
            checked_filename = self.check_rename_files(filename)
                
            PDF = PDFTEXT(f'{checked_filename[:-4]}', 
                          folder_name = 'Articles_to_add', 
                          language = 'english',
                          replace_numbers = False, 
                          min_len_text = 10000,
                          diretorio=self.diretorio)
            PDF.process_text(apply_filters = False)
            
            #checando se há erro na extração do PDF
            if PDF.proc_error_type is None:
                PDF.find_meta_data()
                PDF_title[f'{checked_filename}'] = PDF.title
                PDFID_dict_temp[f'{checked_filename}'] = PDF.PDF_ID
                counter_extracted += 1
                counterfiles += 1
                print(f'Summary - PDF extracted: {counter_extracted} ; PDF non-extracted: {counter_n_extracted}')
            else:
                file_error_list.append([filename, PDF.proc_error_type])
                #print('Exceção de character encontrado: ', repr(string_text[char_index]))
                proc_error_file_name = self.rename_PDF_proc_error_files(filename)
                save_text_to_TXT(PDF.filtered_text, '_' + PDF.proc_error_type + '_' + proc_error_file_name, diretorio=self.diretorio)                
                counter_n_extracted += 1
                counterfiles += 1
                print(f'Summary - PDF extracted: {counter_extracted} ; PDF non-extracted: {counter_n_extracted}')
                continue

        print('Total: ', counterfiles)
        #adicionado os números índices dos arquivos a serem adicionados no DataBase
        files_to_add = [] #essa lista contem o número índice do artigo a ser adicionado no folder DataBase
        PDFID_added = [] #essa lista contem o PDF_ID do documento que será adicionado. Essa lista é para que não haja duplicatas na adição
        for ID_N in range(len(PDFID_dict_temp.values())):
            cond1 = list(PDFID_dict_temp.values())[ID_N] not in list(self.DB_ID['PDF_ID'].values)
            cond2 = list(PDFID_dict_temp.values())[ID_N] not in PDFID_added
            if all([cond1, cond2]) == True:
                files_to_add.append(list(PDFID_dict_temp.keys())[ID_N])
                PDFID_added.append(list(PDFID_dict_temp.values())[ID_N])
        #adicionando os novos arquivos no folder 'DB' e o PDF_IF no DataBase
        tag_list = range(self.last_index, len(files_to_add) + self.last_index)        
        if len(files_to_add) > 0:
            for file_N in range(len(files_to_add)):
                tag_name = get_tag_name(tag_list[file_N])
                self.DB_ID.loc[tag_name] =  (PDFID_dict_temp[files_to_add[file_N]],
                                             PDF_title[files_to_add[file_N]])
                #colocando o artigo na pasta DB
                file_name = files_to_add[file_N]
                self.rename_move_save_files(file_name, tag_list[file_N], save_to_GoogleDrive = save_to_GoogleDrive)
            self.DB_ID.to_csv(self.diretorio + '/Outputs/DB_ID.csv')
            #salvando no Google Drive
            if save_to_GoogleDrive == True: 
                self.drive.upload('DB_ID.csv', self.diretorio + '/Outputs', folderID='1UkqulFBvGwdpUTQm896MTopKGKhk31gO', fileType = 'text/csv')
        
        print()
        print(self.DB_ID)
            
        print('Error report')
        print('--------------------------------------------------')
        print('Os arquivos seguintes não puderam ser processados:')
        for filename in file_error_list:
            print(filename)
        print('Total: ', len(file_error_list))
        print('--------------------------------------------------')

            
    def rename_move_save_files(self, filename, file_list_number, basename = 'PDF', save_to_GoogleDrive = False):
        
        print('( Function: rename_move_save_files )')
        
        from FUNCTIONS import get_tag_name
        import os
        
        old_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{filename}')
        tag_name = get_tag_name(file_list_number)
        new_file_name = os.path.join(self.diretorio + '/DB', f'{tag_name}.pdf')
        os.rename(old_file_name, new_file_name)
        print(f'Arquivo: {filename} renomeado para {tag_name}.pdf e movido com sucesso para a pasta ~/DB.')
        #salvando no GooDrive
        if save_to_GoogleDrive == True: 
            self.drive.upload(f'{tag_name}.pdf', self.diretorio + '/DB', folderID='1jzl2n0T-E3vAfzhEsMu4TZH2rFKP0_uG', fileType = 'application/pdf')

        
    def check_rename_files(self, filename):
        
        #print('( Function: check_rename_files )')
        
        from FUNCTIONS import filename_gen
        import os
        
        if filename[ : 3] == 'PDF':        
            old_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{filename}')
            new_filename = filename_gen()
            new_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{new_filename}.pdf')
            os.rename(old_file_name, new_file_name)
            #print(old_file_name, new_file_name)
            print('Basename de arquivo incompatível')
            print(f'{filename} renomeado para {new_filename}.pdf.')
            return f'{new_filename}.pdf'
        else:
            return filename


    def rename_PDF_proc_error_files(self, filename):
        
        #print('( Function: rename_PDF_proc_error_files )')
        
        from FUNCTIONS import filename_gen
        import os
        
        old_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'{filename}')
        new_filename = filename_gen()
        new_file_name = os.path.join(self.diretorio + '/Articles_to_add', f'_Not_processed_{new_filename}.pdf')
        os.rename(old_file_name, new_file_name)
        return f'_Not_processed_{new_filename}.pdf'
