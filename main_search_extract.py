#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/media/ext4_storage2/NaturalProducts/NLP/Modules')
import os
program_folder = '/media/ext4_storage2/NaturalProducts/NLP'
from FUNCTIONS import extract_inputs_from_csv
SE_inputs = extract_inputs_from_csv(csv_filename = 'SE_inputs', diretorio = program_folder)
#for key in SE_inputs:
#    print('> ', key, ': ', SE_inputs)

for i in SE_inputs.keys():
    if SE_inputs[i]['search_status'].lower() != 'finished' or SE_inputs[i]['export_status'].lower() != 'finished':
        os.system(f'gnome-terminal -- python {program_folder}/Modules/SEARCH_EXTRACT.py --search_input_index={i} --diretorio={program_folder}')
        print(f'gnome-terminal -- python {program_folder}/Modules/SEARCH_EXTRACT.py --search_input_index={i} --diretorio={program_folder}')
