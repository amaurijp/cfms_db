#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
diretorio = os.getcwd()
import sys
sys.path.append(diretorio + '/Modules')

from DB import database
from TEXTOBS import text_objects
from CLASSIFIER import lda, lsa
#from CLASSIFIER import gibbs_sampling
#from FUNCTIONS import run_func_in_parallel
from ML import train_machine
from ML import stats
from WV import WV


def main():

    time_begin=time.time()
    
    article_file_type = 'webscience_csv_report' #('webscience_csv_report', 'pdf', 'xml')
    
    #atualização dos documentos
    datab = database(diretorio=diretorio, article_file_type = article_file_type)
    datab.update()
    
    #processamento dos documentos
    t_objects = text_objects(article_file_type = article_file_type, diretorio = diretorio)
    t_objects.get_text_objects()
    t_objects.find_sent_stats(mode = 'raw_sentences')
    t_objects.filter_sentences()
    t_objects.find_sent_stats(mode = 'filtered_sentences')
    t_objects.find_text_sections()
    t_objects.get_ngrams_appearance()
    t_objects.filter_2grams()
    t_objects.find_idf(min_token_appereance_in_corpus = 10)
    t_objects.set_tfidf_log_and_sent_stats(file_batch_size = 1000)
    t_objects.find_tfidf()
    t_objects.convert_ner_dic_to_nGrams_df()

    #obtendo as matrizes doc_tokens, doc_topicos e treinando o WV com a SVD (LSA)
    words_to_test_wv_models = ['nanoparticles', 'nanosheets', 'aunps', 'gold', 'nps', 'cu', 'copper', 'molybdenum']
    wv = WV(wv_model = 'svd', diretorio = diretorio) #(w2vec or gensim or svd))
    wv.set_svd_parameters(mode = 'truncated_svd')
    wv.get_LSA_wv_matrixes(n_dimensions = 100, word_list_to_test = words_to_test_wv_models)
    wv.get_LSA_sent_topic_matrix()
    wv.get_LSA_article_topic_matrix()
    wv.get_LSA_topic_vector_stats()
    wv.get_wv_stats()

    #classificando os documentos via LDA
    lda_class = lda(diretorio = diretorio)
    lda_class.set_classifier(doc_type = 'articles', n_topics = 50, alpha = 0.01, beta = 0.01, file_batch_size = 100)
    lda_class.start_lda()
    lda_class.run_lda(iterations = 50)

    #classificando os documentos via LSA
    lsa_class = lsa(diretorio = diretorio)
    lsa_class.export_token_topics_to_csv()

    #treinando o WV com as DNNs e deterinando os termos similares (similaridade semântica)
    wv = WV(wv_model = 'w2vec', diretorio = diretorio) #(w2vec or gensim or svd))
    wv.set_w2vec_parameters(mode = 'cbow')
    wv.get_W2Vec(wv_dim = 300, word_list_to_test = words_to_test_wv_models)
    wv.get_wv_stats()
    #inserir a combinação de procura nos arquivos /Settings/input_{semantic}.txt (lembrete: todos o termos devem estar em letra miníscula no arquivo .txt)
    wv.find_terms_sem_similarity()
    
    #treinando as máquinas para reconhecer seções/sentenças usando CNN
    mc = train_machine(machine_type = 'conv1d', wv_matrix_name ='wv_sents_w2vec_cbow_WS_3', diretorio = diretorio)
    mc.set_train_sections(section_name = 'introduction',
                          article_batch_size = 30,
                          sent_batch_size = 4,
                          sent_stride = 1,
                          n_epochs = 100,
                          load_architecture_number = False)
    mc.train_on_sections()

    mc = train_machine(machine_type = 'conv1d', wv_matrix_name ='wv_sents_w2vec_cbow_WS_3', diretorio = diretorio)
    mc.set_train_sections(section_name = 'methodology',
                          article_batch_size = 30,
                          sent_batch_size = 4,
                          sent_stride = 1,
                          n_epochs = 100,
                          load_architecture_number = False)
    mc.train_on_sections()

    mc = train_machine(machine_type = 'conv1d', wv_matrix_name ='wv_sents_w2vec_cbow_WS_3', diretorio = diretorio)
    mc.set_train_sections(section_name = 'results',
                          article_batch_size = 30,
                          sent_batch_size = 4,
                          sent_stride = 1,
                          n_epochs = 100,
                          load_architecture_number = False)
    mc.train_on_sections()
      
    time_end = time.time()
    print('\nTempo de processamento: ', round(time_end - time_begin, 2))



if __name__ == '__main__':
    main()