#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import regex as re
from FUNCTIONS import extract_inputs_from_csv
from DFs import DataFrames
from SCHENG import search_engine
from FUNCTIONS import load_dic_from_json

def main():
      
    #Função principal   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', '--search_input_index', default = 0, help = 'Introduzir o index com os inputs de SEARCH/EXTRACT (manipular arquivo em ~/Settings/SE_inputs.csv).', type = int)
    parser.add_argument('-r', '--revise_search_cond', default = 'no', help ='Decidir se será feita a checagem de todas as condições de busca.', type = str)
    parser.add_argument('-e', '--extraction_mode', default = 'collect_parameters_automatic', help ='Decidir como será o modo de extração ("collect_parameters_automatic" ou "collect_parameters_manual").', type = str)
    parser.add_argument('-f', '--article_file_type', default = 'txt', help ='Introduzir o tipo do arquivo de texto inicial (xml, pdf ou txt).', type = str)
    parser.add_argument('-d', '--diretorio', default = 'None', help ='Introduzir o Master Folder do programa.', type = str)
    
    args = parser.parse_args()

    process(args.search_input_index, args.revise_search_cond, args.extraction_mode, args.article_file_type, args.diretorio)


    
def process(search_input_index, revise_search_cond, extraction_mode, article_file_type, diretorio):

    print('\n(function: search_extract)')

    SE_inputs = extract_inputs_from_csv(csv_filename = 'SE_inputs', diretorio = diretorio, mode = 'search_extract')
    search_input_index = int(search_input_index)
    print()
    print('> SE_input line index: ', search_input_index)
    
    #definindo as variáveis
    print('> filename: ', SE_inputs[search_input_index]['filename'])
    print('> parameter: ', SE_inputs[search_input_index]['parameter_to_extract'])
    print('> scan_sent_by_sent: ', SE_inputs[search_input_index]['scan_sent_by_sent'])
    for key in SE_inputs[search_input_index]['search_inputs'].keys():
        print('>', key, ': ' , SE_inputs[search_input_index]['search_inputs'][key])
        
    #abrindo a index_list caso haja
    index_list = None
    if SE_inputs[search_input_index]['index_list'] is not None:
        if re.search(r'(?<=P)[0-9]+', SE_inputs[search_input_index]['index_list']) is not None and re.search(r'(?<=_G)[0-9]+', SE_inputs[search_input_index]['index_list']) is not None:
            plot_number = re.search(r'(?<=P)[0-9]+', SE_inputs[search_input_index]['index_list']).group()
            group_number = re.search(r'(?<=_G)[0-9]+', SE_inputs[search_input_index]['index_list']).group()
            print('> Procurando index_list: P', plot_number, 'G', group_number)
            index_list_dic = load_dic_from_json(diretorio + '/Inputs/Index_lists.json')
            index_list = index_list_dic['P'+plot_number][group_number]
            print('> Index_list encontrada.')
        else:
            print('ERRO!')
            print('Inserir adequadamente o nome da index_list para fazer a procura.')
            print('Forma correta: P0000_G1')
            return
    
    #inserir a combinação de procura no arquivo /Settings/SE_inputs.csv
    if SE_inputs[search_input_index]['search_status'].lower() != 'finished':
        print('> Go to search engine...')
        se = search_engine(diretorio = diretorio, index_list = index_list)
        se.set_search_conditions(SE_inputs = SE_inputs[search_input_index],
                                    revise_search_conditions = revise_search_cond
                                    )
        se.search_with_combined_models()


    if SE_inputs[search_input_index]['export_status'].lower() != 'finished':
        
        DF = DataFrames(mode = extraction_mode,
                        diretorio = diretorio)
        DF.set_settings(input_DF_name = SE_inputs[search_input_index]['filename'],
                        output_DF_name = SE_inputs[search_input_index]['filename'],
                        parameter = SE_inputs[search_input_index]['parameter_to_extract'],
                        ngram_for_textual_search = SE_inputs[search_input_index]['ngrams_to_extract'],
                        min_ngram_appearence = SE_inputs[search_input_index]['ngrams_min_app'],
                        lower_sentence_in_textual_search = SE_inputs[search_input_index]['lower_sentence_to_extract_parameters'],
                        hold_filenames = False,
                        hold_instances_number = False,
                        numbers_extraction_mode = 'all',
                        filter_unique_results = True,
                        cluster_min_max_num_vals = False)
        #use o regex para pegar parâmetros numéricos dentro das sentenças
        DF.get_data(article_file_type = article_file_type)



###############################################################################################
#executando a função
main()
