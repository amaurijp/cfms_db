#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def main():
    
    import argparse
      
    #Função principal   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-r', '--revise_search_conditions', default = 'yes', help ='Decidir se será feita a checagem de todas as condições de busca (yes or no; sim ou nao).', type = str)
    parser.add_argument('-s', '--se_index', default = 0, help ='Introduzir o index da procura (ver em "SE_inputs.csv") caso a busca seja manual', type = int)
    
    args = parser.parse_args()

    process(args.revise_search_conditions, args.se_index)


def process(revise_search_conditions, se_index):
    
    import os
    import sys
    local_dir = os.getcwd()
    sys.path.append(local_dir + '/Modules')

    from FUNCTIONS import extract_inputs_from_csv
    SE_inputs = extract_inputs_from_csv(csv_filename = 'SE_inputs', diretorio = local_dir)
    
    #for key in SE_inputs:
    #    print('> ', key, ': ', SE_inputs)

    if revise_search_conditions.lower()[0] == 'n':
        for i in SE_inputs.keys():
            if SE_inputs[i]['search_status'].lower() != 'finished' or SE_inputs[i]['export_status'].lower() != 'finished':
                os.system(f'python {local_dir}/Modules/SEARCH_EXTRACT.py --search_input_index={i} --diretorio={local_dir}')
                print(f'python {local_dir}/Modules/SEARCH_EXTRACT.py --search_input_index={i} --revise_search_cond={revise_search_conditions.lower()} --diretorio={local_dir}')
    else:
        if SE_inputs[se_index]['search_status'].lower() != 'finished' or SE_inputs[se_index]['export_status'].lower() != 'finished':
            os.system(f'python {local_dir}/Modules/SEARCH_EXTRACT.py --search_input_index={se_index} --revise_search_cond={revise_search_conditions.lower()} --diretorio={local_dir}')

if __name__ == '__main__':
    main()