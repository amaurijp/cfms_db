#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from SCHENG import search_engine
diretorio = '/media/ext4_storage2/Natural Products/NLP'

search_input = {}


search_input['literal'] = '(Natural Products, Natural Products, Hydrochar, hydrochar)'
search_input['semantic'] = '()'
search_input['topic'] = '()'
search_input['regex'] = 'surface_area'
search_input['bayesian'] = ''


se = search_engine(diretorio = diretorio)
se.set_search_conditions(search_input_dic = search_input,
                         search_mode = 'search_with_combined_models',
                         output_DF_name = 'surface_area_errortest',
                         nGrams_to_get_semantic = 2,
                         lower_sentence_in_semantic_search = True,
                         min_ngram_semantic_appearence = 2
                         )
se.search_with_combined_models()
