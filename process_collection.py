#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def main():
    import time
    import sys
    import os
    diretorio = os.getcwd()
    sys.path.append(diretorio + '/Modules')
    from tika import tika
    tika.TikaClientOnly = True
    tika.TikaStartupMaxRetry = 0
    tika.startServer(diretorio + '/Modules/tika-server-1.24.1.jar')
    time.sleep(0)
    time_begin=time.time()
    from CLASSIFIER import classifier
    from TEXTOBS import text_objects
    from ML import train_machine
    from ML import stats
    from SCHENG import search_engine
    from WV import WV
    import warnings
    warnings.filterwarnings("ignore")
    

    #atualização e classificação dos documentos
    classPDFs = classifier(master_terms = ['natural products'], save_to_GoogleDrive = False, diretorio = diretorio)
    classPDFs.classify_docs(update_DB = False)
    classPDFs.load_classification_TFIDF()
    classPDFs.analyze_TFIDF()
    classPDFs.plot_DB()
    
    #procedimentos sem substituição dos Ngrams
    t_objects = text_objects(text_break_function = 'function_text', tokenizer = 'spacy', replace_Ngrams = False, diretorio = diretorio)
    t_objects.get_text_objects(save_TXT = False, save_JSON = True, filter_by_TF = False, min_tokens_to_break_sent = 2, min_tokens_per_sent = 2)
    t_objects.find_sent_stats(mode = 'raw_sentences')
    t_objects.filter_sentences(cut_quantile_low = 0.01, cut_quantile_high = 0.01, min_sent_tokens = 5, check_function = False)
    t_objects.find_sent_stats(mode = 'filtered_sentences')
    t_objects.find_text_sections(sections_names=[r'([0-9\s\.]*(Materials|MATERIALS)\s(and|And|AND)\s([Mm]ethods|METHODS)\s*([A-Z]|[0-9]\.)|(Experimental|EXPERIMENTAL)\s*[A-Z0-9][\.\sa-z])', 
                                                 r'([0-9\s\.]*(Results?|RESULTS?)\s(and|And|AND)\s([Dd]iscussion|DISCUSSION)\s*([A-Z]|[0-9]\.)|(Results?|RESULTS?)\s*[A-Z0-9][\.\sa-z])'])
    t_objects.get_Ngrams_appearance(min_sent_tokens = 5)
    t_objects.filter_Ngrams(N = 2, threshold_2gram = 10, threshold_3gram = 200, n2gram_mim_score_allowed = 1e-8, n3gram_mim_score_allowed = 1e-16)
    t_objects.find_IDF(min_token_appereance = 10)
    t_objects.set_log_for_TFIDF(n_process = 3)
    t_objects.find_TFIDF(process_number = 4)
    
    #obtendo as matrizes doc_tokens, doc_topicos e treinando o WV com a SVD (LSA)
    wv = WV(wv_model = 'svd', replace_Ngrams = False, diretorio = diretorio) #(w2vec or gensim or svd))
    wv.set_svd_parameters(mode = 'truncated_svd', filter_stopwords = False)
    wv.concat_TFIDF_DFs()
    wv.split_TFIDF_to_SPARSE(batch_size = 200000)
    wv.get_LSA_wv(n_dimensions = 100)
    wv.get_LSA_doc_topic_matrix()
    wv.get_LSA_topic_vector_stats()
    wv.get_matrix_vector_stats()

    #treinando o WV com as DNNs e deterinando os termos similares (similaridade semântica)
    wv = WV(wv_model = 'w2vec', replace_Ngrams = False, diretorio = diretorio) #(w2vec or gensim or svd))
    wv.set_w2vec_parameters(mode = 'cbow', filter_stopwords = False, subsampling = False, words_window_size = 3)
    wv.get_W2Vec(wv_dim = 300, n_epochs = 1, batch_size = 50, pdf_batch_size = 50)
    wv.get_matrix_vector_stats()
    
    #inserir a combinação de procura nos arquivos /Settings/input_{semantic}.txt (lembrete: todos o termos devem estar em letra miníscula no arquivo .txt)
    wv.find_terms_sem_similarity(n_similar_terms_to_overlap = 30, minimum_app_count_to_put_into_2grams = 5)

    #determinando a sobreposição de tópicos usando os termos com similaridade semântica
    wv = WV(wv_model = 'svd', replace_Ngrams = False, diretorio = diretorio) #(w2vec or gensim or svd))
    wv.set_svd_parameters(mode = 'truncated_svd', filter_stopwords = False)
    wv.combine_topic_vectors_by_sem_similarity(n_largest_topic_vals = 30, min_sem_app_count_to_get_topic = 10)
    '''
    #treinando as máquinas para reconhecer seções/sentenças usando CNN
    mc = train_machine(machine_type = 'conv1d', wv_model = 'w2vec', 
                       wv_matrix_name ='w2vec_cbow_SW_False_WS_3_rNgram_False', 
                       filter_stopwords = True, diretorio = diretorio)
    mc.set_train_sections(section_name = 'methodolody',
                          pdf_batch_size = 30,
                          sent_batch_size = 4,
                          sent_stride = 1,
                          n_epochs = 100,
                          vector_type = 'wv')
    mc.train_on_sections()

    #treinando as máquinas para reconhecer seções/sentenças usando Logistic Regressor
    mc = train_machine(machine_type = 'logreg', diretorio = diretorio)
    mc.set_train_sections(section_name = 'methodolody',
                          pdf_batch_size = 1,
                          sent_batch_size = 1,
                          vector_type = 'tv')
    mc.train_on_sections()
    
    #treinando as máquinas estatísticas
    #esse processo é feito após ter sentenças específicas de cada tópico
    feature = 'raw_materials'
    stats = stats(filter_stopwords = True, replace_Ngrams = False, diretorio = diretorio)
    stats.get_topic_prob_func(input_DF_file_names = ['raw_materials_2_extracted_mode_search_with_topic_match',
                                                     'raw_materials_4_extracted_mode_search_with_topic_match',
                                                     'raw_materials_6_extracted_mode_search_with_combined_models',
                                                     'raw_materials_8_extracted_mode_search_with_topic_match'], 
                              feature = feature)
    
    time_end = time.time()
    print('\nTempo de processamento: ', round(time_end - time_begin, 2))
    '''
    
if __name__ == '__main__':
    main()
