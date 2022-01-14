#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class classifier(object):
    
    def __init__(self, master_terms = ['' , ''], save_to_GoogleDrive = False, diretorio = None):
        
        print('\n=============================\nPDF CLASSIFIER...\n=============================\n')
        print('( Class: classifier )')
        
        import pandas as pd
        import os
        self.diretorio = diretorio
        from GoogleDriveAPI import gDrive
        
        self.master_terms = master_terms
        self.save_to_GDrive = save_to_GoogleDrive
        if self.save_to_GDrive == True:
            self.drive = gDrive()
        
        #caso não haja o diretorio ~/Outputs/TFIDF
        if not os.path.exists(self.diretorio + '/Outputs/TFIDF'):
            os.makedirs(self.diretorio + '/Outputs/TFIDF')
        
        country_data = pd.read_csv(self.diretorio + '/Inputs/countries_DB.csv', index_col=0)
        self.country_list = country_data['Name'].values
               
     
    def load_classification_TFIDF(self):
        
        print('( Function: load_classification_TFIDF )')
        
        import pandas as pd

        self.TFIDF_for_classification = pd.read_csv(self.diretorio + '/Outputs/TFIDF/classification_TFIDF.csv', index_col=0)
        print('Carregando o TFIDF (~/Outputs/TFIDF/classification_TFIDF.csv)...')
    
    
    def classify_docs(self, tokenizer = 'python', update_DB = False):

        print('( Function: classify_docs )')
        
        #import time
        import numpy as np
        import os
        import pandas as pd    
        from PDF import PDFTEXT
        from DB import DB
        from FUNCTIONS import get_filenames_from_folder
        from FUNCTIONS import get_tag_name
        from FUNCTIONS import process_search_input_file
        from FUNCTIONS import save_dic_to_json
        import math
        
        
        if update_DB == True:
            DB = DB(diretorio=self.diretorio)
            DB.update(save_to_GoogleDrive = self.save_to_GDrive, min_PDF_len = 20000)
        else:         
            cond1 = os.path.exists(self.diretorio + '/Outputs/TFIDF/classification_TFIDF.csv')
            cond2 = os.path.exists(self.diretorio + '/Outputs/DB_TF_YEAR.csv')
            cond3 = os.path.exists(self.diretorio + '/Outputs/doc_location.csv')
            cond4 = os.path.exists(self.diretorio + '/Outputs/DB_TF_stats.json')
            if False not in (cond1, cond2, cond3, cond4):
                print('CSV de classificação encontrado em ~/Outputs/TFIDF/classification_TFIDF.csv')
                print('CSV de TF e Ano encontrado em ~/Outputs/DB_TF_YEAR.csv')
                print('CSV de localização dos Docs encontrado em ~/Outputs/doc_location.csv')
                print('JSON de TF_stats encontrado em ~/Outputs/DB_TF_stats.json')
                try:
                    pd.read_csv(self.diretorio + '/Outputs/TFIDF/classification_TFIDF.csv', index_col=0).index.values[-1]
                    pd.read_csv(self.diretorio + '/Outputs/DB_TF_YEAR.csv', index_col=0).index.values[-1]
                    pd.read_csv(self.diretorio + '/Outputs/doc_location.csv', index_col=0).index.values[-1]
                    print('> Abortando função: classifier.classify_docs')
                    return
                except IndexError:
                    print('ERRO! Os DFs não possuem entradas.')
                    pass
            
        print('Criando novo TFIDF...')
        print('Classificando os documentos pelo TFIDF (~/Settings/Terms_to_classify_Docs.txt)')
        print('Termos comuns a todos os documentos: ', self.master_terms)
        print('Tokenizer: ', tokenizer)
        print('Update DB: ', update_DB)    
                
        #sec e op não estão sendo usados aqui
        term_list_dic = process_search_input_file('Terms_to_classify_Docs.txt', diretorio=self.diretorio)        
        term_list_primary = [ term_list_dic[search_mode]['primary'] for search_mode in term_list_dic.keys() if (search_mode[-1] in '0123456789') and (search_mode[ : len('literal')] == 'literal' ) ]        
        #print(term_list_primary)
        #time.sleep(10)
        
        #Abrindo os documentos para determinação de TFIDF
        TFclass_Year_dic = {}
        TF = {}
        term_existence = {}
        TFIDF = {}
        country_dic = {}
        
        #número de arquivos no diretório DB        
        try:
            n_documents = len( get_filenames_from_folder(self.diretorio + '/DB', file_type = 'pdf') )
        except TypeError:
            n_documents = 0
        
        for file_N in range(n_documents):
                filename = get_tag_name(file_N)
                #criando o dicionário de TFIDF para cada PDF
                TFIDF[f'{filename}'] = {}
                #criando o dicionário para TF de classificação e para o ano
                TFclass_Year_dic[f'{filename}'] = {}
                TFclass_Year_dic[f'{filename}']['TF'] = ''
                TFclass_Year_dic[f'{filename}']['Publication Year'] = ''
                
                PDF = PDFTEXT(f'{filename}', folder_name = 'DB', diretorio = self.diretorio)
                PDF.process_text(apply_filters = False)
                #encontrando o TF e IDF para os termos usados para classificação (no arquivo ~/Settings/Terms_to_classify_Docs.txt)
                PDF.find_TFIDF(term_list_primary)
                TF[f'{filename}'] = PDF.TF
                term_existence[f'{filename}'] = PDF.check_term_existence
                #encontrando os  meta dados
                PDF.find_meta_data()
                #encontrando o TF para classificação
                PDF.find_class_TF(self.master_terms)
                #definindo os termos para um dicionário
                TFclass_Year_dic[f'{filename}']['TF'] = PDF.class_TF
                TFclass_Year_dic[f'{filename}']['Publication Year'] = PDF.publication_date
                #encontrando o pais que foi feito o artigo
                country_dic[f'{filename}'] = PDF.find_country(self.country_list)
        
        #contando o número de documentos nos quais um termo aparece
        docs_containing_term = {}
        for term in term_list_primary:
            docs_containing_term[str(term)] = 0
            for PDF_file_name in TF.keys():
                if term_existence[PDF_file_name][str(term)] == True:
                    docs_containing_term[str(term)] += 1
        
        #Encontrando o TFIDF            
        for PDF_file_name in TF.keys():
            for term in term_list_primary: 
                #print(f'Número de documentos com o termo: {str(term)}, ', docs_containing_term[str(term)])
                try:            
                    IDF = math.log( n_documents / docs_containing_term[str(term)] )
                except ZeroDivisionError:
                    IDF = 0
                #print(f'O documento {PDF_file_name} contém o termo {str(term)} ', TF[PDF_file_name][str(term)], ' vezes')
                TFIDF[PDF_file_name][str(term)] = TF[PDF_file_name][str(term)] * IDF
        
        self.TFIDF_for_classification = pd.DataFrame(TFIDF).T
        self.TFIDF_for_classification.to_csv(self.diretorio + '/Outputs/TFIDF/classification_TFIDF.csv')
        print('salvando... ~/Outputs/TFIDF/classification_TFIDF.csv')
        self.TFclass_Year_dic = pd.DataFrame.from_dict(TFclass_Year_dic, orient = 'index')
        self.TFclass_Year_dic.to_csv(self.diretorio + '/Outputs/DB_TF_YEAR.csv')
        print('salvando... ~/Outputs/DB_TF_YEAR.csv')
        self.doc_location = pd.DataFrame.from_dict(country_dic, orient = 'index')
        self.doc_location.to_csv(self.diretorio + '/Outputs/doc_location.csv')
        print('salvando... ~/Outputs/doc_location.csv')
                
        #exportando as estatísticas do DB_TF
        DB_TF_array = self.TFclass_Year_dic['TF'].values
        DB_TF_stats = {}
        DB_TF_stats['min'] = int(np.min(DB_TF_array))
        DB_TF_stats['max'] = int(np.max(DB_TF_array))
        DB_TF_stats['avg'] = round(np.mean(DB_TF_array), 10)
        DB_TF_stats['std'] = round(np.std(DB_TF_array), 10)
        DB_TF_stats['median'] = round(np.median(DB_TF_array), 10)
        DB_TF_stats['1_quartile'] = round(np.quantile(DB_TF_array, 0.25), 10)
        DB_TF_stats['3_quartile'] = round(np.quantile(DB_TF_array, 0.75), 10)
        DB_TF_stats['0.05_quantile'] = round(np.quantile(DB_TF_array, 0.05), 10)
        DB_TF_stats['0.95_quantile'] = round(np.quantile(DB_TF_array, 0.95), 10)
    
        #salvando o sent_stats (estatística de token por sentença)
        save_dic_to_json(self.diretorio + '/Outputs/DB_TF_stats.json', DB_TF_stats)
    
        if self.save_to_GDrive == True:
            self.drive.upload('classification_TFIDF.csv', self.diretorio + '/Outputs/TFIDF', folderID='16rs7jaL1egbr90bM3k2a0BjMDQAzzfrd', fileType = 'text/csv')

            
    def analyze_TFIDF(self):
        
        print('( Function: analyze_TFIDF )')
        
        import os
        if os.path.exists(self.diretorio + '/Outputs/TFIDF/classification_Articles_Categorized.csv'):
            print('CSV de categorização encontrado em ~/Outputs/TFIDF/classification_Articles_Categorized.csv')
            print('> Abortando função: classifier.analyze_TFIDF')
            return
        
        #analyzando os TFIDFs para classificação    
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        #import numpy as np    
            
        #abrindo a DF TFIDF
        #print('Raw TFIDF (Document Term matrix)')
        #print(self.TFIDF_for_classification)        

        #---------------------------------------------------------------
        #Os topic vectors estão sendo calculados por PCA (abaixo)
        '''
        print('Calculando o SVD pelo numpy...')
        #centralizando os valores da DF
        term_document_matrix = self.TFIDF_for_classification.T - self.TFIDF_for_classification.T.values.mean()
        print('Term-Document Matrix (TFIDF.T)')
        print(term_document_matrix)
        U, S, Vt = np.linalg.svd(term_document_matrix)
        U_term_topic_matrix = pd.DataFrame(U, index=term_document_matrix.index)
        print('U (Term Topic) matrix')
        print(U_term_topic_matrix)
        U_topic_term_matrix = U_term_topic_matrix.T
        print('U.T (Topic Term) matrix')
        print(U_topic_term_matrix)
        topic_document_matrix = U_topic_term_matrix.values.dot(term_document_matrix)
        topicsN = ['topic{}'.format(i) for i in range(U.shape[1])]
        TDM_DF = pd.DataFrame(topic_document_matrix.T, columns=topicsN, index=self.TFIDF_for_classification.index)
        print('SVD Topic Document Matrix (Topic-Term-M . Term-Document-M)')
        print(TDM_DF)
        '''
        
        #---------------------------------------------------------------     
        print('Calculando os Topic Vectors por PCA (sklearn)...')
        pca = PCA(n_components  = len(self.TFIDF_for_classification.columns))
        pca = pca.fit(self.TFIDF_for_classification)
        pca_topic_vectors = pca.transform(self.TFIDF_for_classification)
        topicsN = ['topic{}'.format(i) for i in range(pca.n_components)]
        pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=topicsN, index=self.TFIDF_for_classification.index)
        #print('PCA Topic Vectors')
        #print(pca_topic_vectors)
        
        #encontrando os weights para cada termo do TFIDF
        term_weights_M = pd.DataFrame(pca.components_, columns=self.TFIDF_for_classification.columns, index=topicsN).T
        term_weights_M.to_csv(self.diretorio + '/Outputs/TFIDF/classification_PCA_Topic_Vectors_Weights.csv')
        if self.save_to_GDrive == True:
            self.drive.upload('classification_PCA_Topic_Vectors_Weights.csv', self.diretorio + '/Outputs/TFIDF', folderID='16rs7jaL1egbr90bM3k2a0BjMDQAzzfrd', fileType = 'text/csv')
        #print('PCA Term-Topic Weights Matrix')
        #for index in term_weights_M.index:
        #    print(term_weights_M.loc[index])

        #---------------------------------------------------------------
        #redefinindo as colunas da PCA Document-Topic Matrix de acordo com os termos que
        #melhor definem cada tópico
                
        #A função classifica as colunas em função das linhas
        def find_topic_classification(dataframe):
            #encontrando o termo que mais se associa ao tópico e o topic_val (para determinação da qualidade da classificação)
            tag_list = {}
            tag_col = {}
            for column in dataframe.columns:
                col_vals = dataframe[column].values
                max_val_pos = np.where( col_vals == col_vals.max())[0][0]
                tag = dataframe.index.values[max_val_pos]
                topic_val = dataframe[column].iloc[max_val_pos]
                if tag not in list(tag_list.keys()):
                    tag_list[tag] = topic_val
                    tag_col[tag] = column
                else:
                    if topic_val > tag_list[tag]:
                        tag_list[tag] = topic_val
                        tag_col[tag] = column
            
            counter = 1
            for column in dataframe.columns:
                if column not in tag_col.values():
                    tag_list['Not Classified ' + str(counter) ] = 0
                    counter += 1
               
            return pd.DataFrame(np.array([list(tag_list.keys()), list(tag_list.values())]).T, columns=['Classification', 'Topic Val'])

        #encontrando a classificação para os tópicos
        categories_classification = find_topic_classification(term_weights_M)
        categories_classification.set_index('Classification', inplace=True)
        #print('CATEGORIES FOUND FOR TOPICS')
        #print(categories_classification)
        
        #Encontrando a classificação para os documentos
        #Primeiro se dá o nome dos termos de classificação para os tópicos
        pca_topic_vectors.columns = categories_classification.index
        #print('PCA Topic Vectors (categorized)')
        #print(pca_topic_vectors)
        
        #A função classifica as colunas em função das linhas
        def find_document_classification(dataframe):
            #encontrando o termo que mais se associa ao tópico e o topic_val (para determinação da qualidade da classificação)
            topic_val_list = {}
            tag_list = {}
            for index in dataframe.index:
                line_vals = dataframe.loc[index].values
                max_val_pos = np.where( line_vals == line_vals.max())[0][0]
                tag = dataframe.columns[max_val_pos]
                tag_list[index] = tag
                topic_val = dataframe[tag].loc[index]
                topic_val_list[index] = topic_val
                
            return pd.DataFrame(np.array([list(tag_list.values()), list(topic_val_list.values())]).T, columns=['Classification', 'Topic Val'], index=range(len(tag_list.values())))
        
        #Como a função encontra a classificação da coluna em função da linha, usa-se a matrix transposta
        documents_classification = find_document_classification(pca_topic_vectors)
        #print('PDFs CATEGORIZED BY TOPIC')
        #print(documents_classification)        

        #avaliando a qualidade da classificação
        def func_quality_classification(dataframe_to_classify, dataframe_topic_class_quality):
            import numpy as np
            qual_vals = []
            for index in dataframe_to_classify.index:
                try:
                    doc_topic_val = dataframe_to_classify['Topic Val'].loc[index]
                    classification = dataframe_to_classify['Classification'].loc[index]
                    class_topic_qual_val = dataframe_topic_class_quality.loc[classification]
                    final_qual_val = float(doc_topic_val) * float(class_topic_qual_val)
                    qual_vals.append(final_qual_val)
                except KeyError:
                    qual_vals.append(0)
            qual_vals_array = np.array(qual_vals)
            quart_qual_vals = np.array(pd.cut(qual_vals_array, 4, labels = ['poor', 'moderate', 'good', 'excelent']))
            array_mask = (qual_vals_array <= 0)
            for index_N in range(len(array_mask)):
                if array_mask[index_N] == True:
                    quart_qual_vals[index_N] = 'NC'
            return quart_qual_vals

        quality_quartils = func_quality_classification(documents_classification, categories_classification)

        #montando a DF final
        pca_topic_vectors['Category'] = documents_classification['Classification'].values
        pca_topic_vectors['Class. Quality'] = quality_quartils 
        pca_topic_vectors['Articles'] = pca_topic_vectors.index
        
        #print('Processed DF')
        pca_topic_vectors.set_index(['Articles', 'Category', 'Class. Quality'], inplace=True)
        pca_topic_vectors.sort_index(level=1, inplace=True)
        print('FINAL DF')
        print(pca_topic_vectors)
        
        #Salvando o DF
        pca_topic_vectors.to_csv(self.diretorio + '/Outputs/TFIDF/classification_Articles_Categorized.csv')
        if self.save_to_GDrive == True:
            self.drive.upload('Articles_Categorized.csv', self.diretorio + '/Outputs/TFIDF', folderID='16rs7jaL1egbr90bM3k2a0BjMDQAzzfrd', fileType = 'text/csv')

        ''' 
        sns.set(font_scale=0.5)
        #plotando e exportando a PCA DOCUMENT TOPIC MATRIX (RAW)
        fig = sns.pairplot(pca_topic_vectors, diag_kind='kde', plot_kws={'alpha': 0.5})
        fig.savefig(self.diretorio + '/Outputs/TFIDF/classification_PCA_pairplot.png', dpi=200)
        if self.save_to_GDrive == True:
            self.drive.upload('classification_PCA_pairplot.png', self.diretorio + '/Outputs/TFIDF', folderID='16rs7jaL1egbr90bM3k2a0BjMDQAzzfrd', fileType = 'image/png')
            '''
            
        
    def plot_DB(self):
        
        print('( Function: plot_DB )')
        
        import pandas as pd
        import numpy as np
        from FUNCTIONS import load_dic_from_json
        from matplotlib import pyplot as plt
        import os

        #abrindo o DF TF YEAR        
        if os.path.exists(self.diretorio + '/Outputs/DB_TF_YEAR.csv'):
            self.DF_TF_Year = pd.read_csv(self.diretorio + '/Outputs/DB_TF_YEAR.csv', index_col = 0)
            try:
                self.DF_TF_Year.index.values[-1]
            except IndexError:
                print('ERRO! DF TF_Year não possui entradas.')
                print('> Abortando função: classifier.plot_DB')
                return            
        else:
            print('ERRO! DF TF_Year não encontrado.')
            print('> Abortando função: classifier.plot_DB')
            return

        #abrindo o json com o DB_TF_stats
        if os.path.exists(self.diretorio + '/Outputs/DB_TF_stats.json'):
            self.DB_TF_stats = load_dic_from_json(self.diretorio + '/Outputs/DB_TF_stats.json')
        else:
            print('ERRO! Dicionário TF_stats não encontrado.')
            print('> Abortando função: classifier.plot_DB')
            return

        #abrindo o DF de doc_location
        if os.path.exists(self.diretorio + '/Outputs/doc_location.csv'):
            self.doc_location = pd.read_csv(self.diretorio + '/Outputs/doc_location.csv', index_col = 0)
            try:
                self.doc_location.index.values[-1]
            except IndexError:
                print('ERRO! DF de doc_location não possui entradas.')
                print('> Abortando função: classifier.plot_DB')
                return            
        else:
            print('ERRO! DF de doc_location não encontrado.')
            print('> Abortando função: classifier.plot_DB')
            return
        
        #abrindo o DF de classificação
        if os.path.exists(self.diretorio + '/Outputs/TFIDF/classification_Articles_Categorized.csv'):
            self.DF_class = pd.read_csv(self.diretorio + '/Outputs/TFIDF/classification_Articles_Categorized.csv', index_col = 0)
            try:
                self.DF_class.index.values[-1]
            except IndexError:
                print('ERRO! DF de classificação não possui entradas.')
                print('> Abortando função: classifier.plot_DB')
                return            
        else:
            print('ERRO! DF de classificação não encontrado.')
            print('> Abortando função: classifier.plot_DB')
            return
        
        
        #plotando o TF-YEAR DF
        print('Plotando o TF-YEAR-Class plot...')
        
        fig, axes = plt.subplots(2, 2, figsize=(35,16))
        #plotando o TF
        axes[0,0].hist(self.DF_TF_Year['TF'].values, bins=100)
        axes[0,0].set_title(f'TF para {self.master_terms}', fontsize=18)
        axes[0,0].tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, colors='k', width=2, length=3.5, labelsize=14)
        axes[0,0].tick_params(axis="x", labelbottom=True, bottom=True, labeltop=False, top=False, colors='k', width=2, length=3.5, labelsize=14)
        axes[0,0].set_xlabel('TF', labelpad=5, fontsize=17)
        axes[0,0].set_ylabel('Articles', labelpad=5, fontsize=17)
        
        axes[0,0].axvline(self.DB_TF_stats['avg'], color='green', label='mean', alpha=0.5)
        axes[0,0].axvline(self.DB_TF_stats['avg'] + self.DB_TF_stats['std'], color='red', alpha=0.5, label='+std')
        axes[0,0].axvline(self.DB_TF_stats['avg'] - self.DB_TF_stats['std'], color='red', alpha=0.5, label='-std')
        axes[0,0].axvline(self.DB_TF_stats['median'], color='blue', alpha=0.5, label='median')
        axes[0,0].axvline(self.DB_TF_stats['1_quartile'], color='orange', alpha=0.5, label='1_quartile')
        axes[0,0].axvline(self.DB_TF_stats['3_quartile'], color='orange', alpha=0.5, label='3_quartile')
        axes[0,0].axvline(self.DB_TF_stats['0.05_quantile'], color='black', alpha=0.5, label='0.05_quantile')
        axes[0,0].axvline(self.DB_TF_stats['0.95_quantile'], color='black', alpha=0.5, label='0.95_quantile')
        axes[0,0].legend()
        
        #plotando o ano de publicação
        year_list_str, year_count_str = np.unique(self.DF_TF_Year['Publication Year'].values, return_counts=True)
        #print(year_list_str, year_count_str)
        year_list = []
        year_count = []
        year_not_extracted = 0
        for index in range(len(year_list_str)):
            try:
                year_list.append(int(year_list_str[index]))
                year_count.append(int(year_count_str[index]))
            except:
                year_not_extracted = int(year_count_str[index])
                continue
                    
        axes[0,1].bar(year_list, year_count)
        axes[0,1].set_title(f'Ano de publicação dos artigos (Not counted: {year_not_extracted})', fontsize=18)
        axes[0,1].tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, colors='k', width=2, length=3.5, labelsize=14)
        axes[0,1].tick_params(axis="x", labelbottom=True, bottom=True, labeltop=False, top=False, colors='k', width=2, length=3.5, labelsize=14)
        axes[0,1].set_xlabel('Ano', labelpad=5, fontsize=17)
        axes[0,1].set_ylabel('Artigos', labelpad=5, fontsize=17)
        axes[0,1].set_xticks(list(range(min(year_list), max(year_list)+1)))
        
        axes[0,1].set_title('Classificação dos documentos', fontsize=18)
        axes[0,1].tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, colors='k', width=2, length=3.5, labelsize=14)
        axes[0,1].tick_params(axis="x", labelbottom=True, bottom=True, labeltop=False, top=False, colors='k', width=2, length=3.5, labelsize=14)
        axes[0,1].set_ylabel('Artigos', labelpad=5, fontsize=17)
        axes[0,1].set_xlabel('Categorias', labelpad=5, fontsize=17)
        axes[0,1].set_xticks(list(range(min(year_list), max(year_list)+1)))
    
        #plotando o doc_location
        #print(self.doc_location.groupby(['Category']).nunique())
        flat_doc_location_array = self.doc_location.values.reshape(-1)
        not_nan_bool = ~pd.isnull(flat_doc_location_array)
        doc_locations_list = flat_doc_location_array[not_nan_bool]    
        locations , loc_counts = np.unique(doc_locations_list, return_counts=True)
        axes[1,0].bar(locations, loc_counts)
        axes[1,0].set_title('Onde foram produzidos os documentos', fontsize=18)
        axes[1,0].tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, colors='k', width=2, length=3.5, labelsize=14)
        axes[1,0].tick_params(axis="x", labelrotation=90, labelbottom=True, bottom=True, labeltop=False, top=False, colors='k', width=2, length=3.5, labelsize=8)
        axes[1,0].set_xlabel('País', labelpad=5, fontsize=17)
        axes[1,0].set_ylabel('Counts', labelpad=5, fontsize=17)
        
        #plotando o Class DF
        #print(self.DF_class.groupby(['Category']).nunique())
        self.DF_class.groupby('Category').nunique().iloc[ : , 2].plot(kind='bar', ax=axes[1,1])

                
        #salvando as figuras
        fig.tight_layout(pad=1.08, h_pad=5, w_pad=None, rect=None)
        fig.savefig(self.diretorio + '/Outputs/TF_YEAR_Class.png', dpi=100)
        #salvando no Google Drive
        if self.save_to_GDrive == True: 
            self.drive.upload('TF_YEAR_Class.png', self.diretorio + '/Outputs', folderID='1UkqulFBvGwdpUTQm896MTopKGKhk31gO', fileType = 'image/png')        