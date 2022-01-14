#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class download(object):
    
    def __init__(self, diretorio = None):

        self.diretorio = diretorio
        print('diretorio: ', self.diretorio)
        
    def download_articles_from_DOI(self, year = '0000', Gdrive_folder_ID = ''):

        print('year: ', year)        

        import time
        import os        
        import numpy as np
        import regex as re
        import pandas as pd
        import urllib.request
        from selenium.common import exceptions
        from urllib.error import HTTPError
        from urllib.error import URLError
        from seleniumwire import webdriver    
        from FUNCTIONS import get_filenames_from_folder
        from GoogleDriveAPI import gDrive

        self.drive = gDrive()

        if not os.path.exists(self.diretorio + f'/Inputs/DOI/{year}/PDFs'):
            os.makedirs(self.diretorio + f'/Inputs/DOI/{year}/PDFs')

        print('Procurando DF em: ', self.diretorio + f'/Inputs/DOI/{year}/concat_DF.csv')
        if os.path.exists(self.diretorio + f'/Inputs/DOI/{year}/concat_DF.csv'):
            concat_DF = pd.read_csv(self.diretorio + f'/Inputs/DOI/{year}/concat_DF.csv', index_col=[0,1])
            
        else:
            concat_DF = pd.DataFrame([], index=[[],[]])
            
            filenames = get_filenames_from_folder(self.diretorio + f'/Inputs/DOI/{year}', file_type = 'xls')        
            for filename in filenames:        
                temp_df = pd.read_excel(self.diretorio + f'/Inputs/DOI/{year}/{filename}.xls')
                temp_df = temp_df[['Publication Type', 'Article Title', 'Source Title', 'Language', 'Publisher', 'Publication Year', 'DOI']].copy()
                concat_DF = pd.concat([concat_DF, temp_df], axis=0, ignore_index=True)
    
            concat_DF['Status'] = len(concat_DF.index) * ['to_try']
            concat_DF['Counter'] = np.arange(0, len(concat_DF.index))
            concat_DF.set_index(['Publication Type', 'Counter'], inplace=True)
            concat_DF.to_csv(self.diretorio + f'/Inputs/DOI/{year}/concat_DF.csv')
        

        for index in concat_DF.index:
            #caso seja um artigo
            if index[0].lower() == 'j':
                if concat_DF.loc[ index , 'Status'].lower() == 'to_try' and concat_DF.loc[ index , 'Language'].lower() == 'english':

                    doi_number = concat_DF.loc[ index , 'DOI']
                    print('--------------------------------------------')
                    print('Index: ', index[1])
                    print('Carregando o documento: ', doi_number)
                    print('Doc Title: ', concat_DF.loc[ index , 'Article Title'])
                    print('Source: ', concat_DF.loc[ index , 'Source Title'])
                    print('Publisher: ', concat_DF.loc[ index , 'Publisher'])

                    #localizando o doi com o crossref                    
                    print('Building opener via urllib')
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('Accept', 'application/vnd.crossref.unixsd+xml')]
                    
                    try:
                        r = opener.open(f'http://dx.doi.org/{doi_number}')                    
                        link_to_FT = re.search(r'(?<=rel\="canonical",\s*\<).+?(?=\>)', r.info()['Link']).captures()[0]
                        print('Link to FULL_TEXT found: ', link_to_FT)
    
                        # Use firefox dowmloader to get file
                        fp = webdriver.FirefoxProfile()
                        fp.set_preference("browser.download.folderList", 2)
                        fp.set_preference("browser.download.manager.showWhenStarting", False)
                        fp.set_preference("browser.download.dir", self.diretorio + f'/Inputs/DOI/{year}/PDFs')
                        fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
                        fp.set_preference("pdfjs.disabled", "true")
                        fp.set_preference("pdfjs.disabled", True)
                        
                        # disable Adobe Acrobat PDF preview plugin
                        fp.set_preference("plugin.scan.plid.all", "false")
                        fp.set_preference("plugin.scan.Acrobat", "99.0")
    
                        options = {'proxy': {'http': 'http://0000000:00000@proxy.br:8080', 
                                             'https': 'https://0000000:00000@proxy.br:8080',
                                             'no_proxy': 'localhost,127.0.0.1'}}
                                                
                        print('Abrindo o FIREFOX...')
                        browser = webdriver.Firefox(firefox_profile=fp, seleniumwire_options=options)
                        time.sleep(2)
                        print('Encaminhando para o link com o FULL TEXT...')
                        browser.get(link_to_FT)
    
                        if 'elsevier' in concat_DF.loc[ index , 'Publisher'].lower():
                            new_link = re.search(r'(?<=rel\="self"/\>\<link\shref\=").+?(?="\srel="scidir"/\>)', browser.page_source).captures()[0]
                            print('Opening new link: ', new_link)
                            browser.get(new_link)
                            time.sleep(3)                   
                            button_link = re.search(r'(?<="linkToPdf":").+?(?=")', browser.page_source).captures()[0]
                            browser.get('https://www.sciencedirect.com' + button_link)
                            time.sleep(50)
                        elif 'public library science' in concat_DF.loc[ index , 'Publisher'].lower():
                            browser.find_element_by_link_text('Download PDF').click()
                            time.sleep(50)
                        else:
                            time.sleep(50)
                            
                        print('Closing FIREFOX...')
                        browser.quit()
                        concat_DF.loc[ index , 'Status'] = 'complete'
                        concat_DF.to_csv(self.diretorio + f'/Inputs/DOI/{year}/concat_DF.csv')                        
                        self.drive.upload('concat_DF.csv', self.diretorio + f'/Inputs/DOI/{year}', folderID=Gdrive_folder_ID, fileType = 'text/csv')
                        time.sleep(2)
                        
                    except exceptions.WebDriverException:
                        print('Closing FIREFOX...')
                        concat_DF.loc[ index , 'Status'] = 'complete'
                        concat_DF.to_csv(self.diretorio + f'/Inputs/DOI/{year}/concat_DF.csv')
                        self.drive.upload('concat_DF.csv', self.diretorio + f'/Inputs/DOI/{year}', folderID=Gdrive_folder_ID, fileType = 'text/csv')
                        time.sleep(2)
                        continue
                        
                    except (HTTPError, URLError, AttributeError, TypeError):
                        print('Error! FULL_TEXT not found!')
                        concat_DF.loc[ index , 'Status'] = 'FT_not_found'
                        concat_DF.to_csv(self.diretorio + f'/Inputs/DOI/{year}/concat_DF.csv')
                        self.drive.upload('concat_DF.csv', self.diretorio + f'/Inputs/DOI/{year}', folderID=Gdrive_folder_ID, fileType = 'text/csv')
                        time.sleep(2)
                        continue



###############################################################################################
def main():
    
    import argparse
      
    #Função principal   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-y', '--year', default = 'None', help = 'Introduzir o ano dos documentos que serão baixados.', type = str)
    parser.add_argument('-d', '--diretorio', default = 'None', help ='Introduzir o Master Folder do programa (manipular arquivo em ~/Settings/search_inputs.csv).', type = str)
    parser.add_argument('-f', '--folder_ID', default = 'None', help ='Introduzir o folder ID do Google_Drive (O ID da pasta que será salvo o arquivo concat_DF.csv).', type = str)
    
    args = parser.parse_args()

    process(args.year, args.diretorio, args.folder_ID)


def process(year, diretorio, folder_ID):
    
    dw = download(diretorio = diretorio)
    dw.download_articles_from_DOI(year = year, Gdrive_folder_ID = folder_ID)


###############################################################################################
#executando a função
main()
