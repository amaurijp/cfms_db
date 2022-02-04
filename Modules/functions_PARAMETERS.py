#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_physical_units(mode = 'SI'):

    dic = {}
    
    if mode.lower() == 'si':
        
        dic['areaagro'] = ['ha']
        dic['areametric'] = ['nm2', 'um2', 'mm2', 'cm2', 'dm2', 'm2', 'km2']
        dic['distance'] = ['nm', 'um', 'mm', 'cm', 'dm', 'm' , 'km' ]
        dic['energy'] = ['mJ', 'J', 'kJ', 'MJ']
        dic['electricpotential'] = ['mV', 'V', 'kV']
        dic['force'] = ['N']
        dic['molarity'] = ['umol', 'mmol', 'cmol', 'mol']
        dic['potency'] = ['W']
        dic['ratio'] = ['adimensional']
        dic['percentage'] = ['%', 'wtperc']
        dic['temperature'] = ['C', 'o C', 'oC', 'K']
        dic['time'] = ['s', 'min', 'h', 'day', 'yr']
        dic['volume'] = ['mm3', 'dm3', 'cm3', 'm3', 'ul', 'ml', 'l']
        dic['weight'] = ['ng', 'ug', 'mg', 'g', 'kg', 't']
    
    elif mode.lower() == 'all':
        
        dic['areaagro'] = ['ha', 'Ha', 'HA']
        dic['areametric'] = ['nm2', 'um2', 'µm2', 'mm2', 'cm2', 'dm2', 'm2', 'km2', 'Km2']
        dic['distance'] = ['nm', 'um', 'µm', 'mm', 'cm', 'dm', 'm' , 'km' , 'Km']
        dic['energy'] = ['mJ', 'J', 'kJ', 'KJ', 'MJ']
        dic['electricpotential'] = ['mV', 'V', 'kV', 'KV']
        dic['force'] = ['N', 'Newtons']
        dic['molarity'] = ['umol', 'µmol', 'mmol', 'cmol', 'mol']
        dic['potency'] = ['watts', 'W']
        dic['ratio'] = ['adimensional']
        dic['percentage'] = ['%', 'wtperc']
        dic['temperature'] = ['C', 'o C', 'oC', 'K']
        dic['time'] = ['s', 'sec', 'secs', 'Sec', 'Secs', 'min', 'mins', 'Min', 'Mins', 'h', 'hour', 'hours', 'Hour', 'Hours', 
                       'day', 'days', 'Day', 'Days', 'year', 'years', 'Year', 'Years', 'yr', 'yrs']
        dic['volume'] = ['mm3', 'dm3', 'cm3', 'cc', 'm3', 'ul', 'uL', 'µl', 'µL', 'ml', 'mL',  'l', 'L']
        dic['weight'] = ['ng', 'ug', 'µg', 'mg', 'g', 'kg', 'Kg', 't','ton', 'tonne', 'TON']
    
    return dic


#------------------------------
def get_physical_units_combined(first_parameter = '', second_parameter = None, get_inverse = False, mode = 'si'):
        
    #motando as combinações de unidades físicas
    PU_units_combined = []    

    #coletando todas as unidades físicas
    PU_dic = get_physical_units(mode = mode)

    #varrendo as unidades físicas da classe primária
    for unit1 in PU_dic[first_parameter]:        
        #varrendo todas as classe de unidades físicas
        for key in PU_dic.keys():
            if (key != second_parameter) and (second_parameter is not None):
                continue
            
            elif (second_parameter is None) or (second_parameter == key):
                #fazendo a combinação com todos os parâmetros
                if get_inverse is False:
                    PU_units_combined.extend( [ ( unit1 + ' ' +  unit2 ) for unit2 in PU_dic[key] ] )
                #encontrando as unidades inversas de todas os outros parâmetros e combinando o parâmetro introduzido
                elif get_inverse is True:
                    PU_units_combined.extend( [ ( unit1 + ' ' +  get_physical_unit_inverse(unit2) ) for unit2 in PU_dic[key] ] )
        
    return PU_units_combined


#------------------------------
def get_physical_units_combined_list():

    PU_units_combined_list = []
    
    #weight_volume
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'areaagro', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'areametric', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'distance', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'energy', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'electricpotential', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'force', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'molarity', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'potency', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'ratio', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'percentage', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'temperature', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'time', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'volume', mode = 'all') )
    PU_units_combined_list.extend( get_physical_units_combined(first_parameter = 'weight', mode = 'all') )

    return PU_units_combined_list


#------------------------------
def get_physical_units_converted_to_SI(PUs):

    import time    

    #dicionário para trabalhar com as unidades de entrada (raw)
    units = {}
    units['factor_list'] = []
    units['factor_operation'] = []
    units['raw_unit'] = []
    units['SI_unit'] = []
    

    #obtendo as unidades físicas (PUs)
    PU_units = get_physical_units()

    for PU in PUs:

        units['raw_unit'].append(PU)

        #adimensional
        if PU in PU_units['ratio']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('adimensional')
            #encontrando os fatores de conversão
            units['factor_list'].append(1)
            units['factor_operation'].append('multiply')
        
        #porcentagem
        elif PU in PU_units['percentage']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('%')
            #encontrando os fatores de conversão
            units['factor_list'].append(1)
            units['factor_operation'].append('multiply')
        
        #area_agro
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['areaagro']] + PU_units['areaagro']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('ha')
            #encontrando os fatores de conversão
            units['factor_list'].append(1)
            units['factor_operation'].append('multiply')
                
        #area_metric
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['areametric']] + PU_units['areametric']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('m2')
            #encontrando os fatores de conversão            
            if PU in [ get_physical_unit_inverse(unit) for unit in ['nm2']] + ['nm2']:
                units['factor_list'].append(1e-9 ** 2)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['um2']] + ['um2']:
                units['factor_list'].append(1e-6 ** 2)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mm2']] + ['mm2']:
                units['factor_list'].append(1e-3 ** 2)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['cm2']] + ['cm2']:
                units['factor_list'].append(1e-2 ** 2)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['dm2']] + ['dm2']:
                units['factor_list'].append(1e-1 ** 2)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['m2']] + ['m2']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['km2']] + ['km2']:
                units['factor_list'].append(1e3 ** 2)
                units['factor_operation'].append('multiply')

        #energy
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['energy']] + PU_units['energy']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('J')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['mJ']] + ['mJ']:
                units['factor_list'].append(1e-3)                
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['J']] + ['J']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['kJ']] + ['kJ']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['MJ']] + ['MJ']:
                units['factor_list'].append(1e6)
                units['factor_operation'].append('multiply')

        #distance
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['distance']] + PU_units['distance']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('m')
            #encontrando os fatores de conversão            
            if PU in [ get_physical_unit_inverse(unit) for unit in ['nm']] + ['nm']:
                units['factor_list'].append(1e-9)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['um']] + ['um']:
                units['factor_list'].append(1e-6)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mm']] + ['mm']:
                units['factor_list'].append(1e-3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['cm']] + ['cm']:
                units['factor_list'].append(1e-2)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['dm']] + ['dm']:
                units['factor_list'].append(1e-1)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['m']] + ['m']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['km']] + ['km']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')

        #electric_potential
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['electricpotential']] + PU_units['electricpotential']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('V')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['mV']] + ['mV']:
                units['factor_list'].append(1e-3)                
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['V']] + ['V']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['kV']] + ['kV']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')
                
        #molarity
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['molarity']] + PU_units['molarity']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('mol')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['umol']] + ['umol']:
                units['factor_list'].append(1e-6)                
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mmol']] + ['mmol']:
                units['factor_list'].append(1e-3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mol']] + ['mol']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')            

        #temperature
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['temperature']] + PU_units['temperature']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('C')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['K']] + ['K']:
                units['factor_list'].append(-273)
                units['factor_operation'].append('add')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['C']] + ['C']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')
    
        #time
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['time']] + PU_units['time']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('min')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['s', 'sec', 'Sec', 'secs', 'Secs']] + ['s', 'sec', 'Sec', 'secs', 'Secs']:
                units['factor_list'].append(1/60)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['min', 'min', 'mins', 'Mins']] + ['min', 'min', 'mins', 'Mins']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['h', 'hour', 'Hour', 'hours', 'Hours']] + ['h', 'hour', 'Hour', 'hours', 'Hours']:
                units['factor_list'].append(60)
                units['factor_operation'].append('multiply')            
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['day', 'days', 'Day', 'Days']] + ['day', 'days', 'Day', 'Days']:
                units['factor_list'].append(60*24)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['year', 'years', 'Year', 'Years', 'yr', 'yrs']] + ['year', 'years', 'Year', 'Years', 'yr', 'yrs']:
                units['factor_list'].append(60*24*365)
                units['factor_operation'].append('multiply')

        #volume
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['volume']] + PU_units['volume']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('l')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['mm3']] + ['mm3']:
                units['factor_list'].append( (1e-3 ** 3) * 1000 ) # * 1000 para passar para litro
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['cm3']] + ['cm3']:
                units['factor_list'].append( (1e-2 ** 3) * 1000 ) # * 1000 para passar para litro
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['cc']] + ['cc']:
                units['factor_list'].append( (1e-2 ** 3) * 1000 ) # * 1000 para passar para litro
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['dm3']] + ['dm3']:
                units['factor_list'].append( (1e-1 ** 3) * 1000 ) # * 1000 para passar para litro
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['m3']] + ['m3']:
                units['factor_list'].append(1000) # * 1000 para passar para litro
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['ul', 'uL']] + ['ul', 'uL']:
                units['factor_list'].append(1e-6)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['ml', 'mL']] + ['ml', 'mL']:
                units['factor_list'].append(1e-3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['l', 'L']] + ['l', 'L']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')                

        #weight
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['weight']] + PU_units['weight']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('g')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['ug']] + ['ug']:
                units['factor_list'].append(1e-6)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mg']] + ['mg']:
                units['factor_list'].append(1e-3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['g']] + ['g']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['kg']] + ['kg']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['t','ton', 'tonne', 'TON']] + ['t','ton', 'tonne', 'TON']:
                units['factor_list'].append(1e6)
                units['factor_operation'].append('multiply')            

            
    #caso todas as unidades tenham sido identificadas
    if len(units['factor_list']) == len(units['raw_unit']) == len(units['SI_unit']):
        
        #fator de conversão
        conv_factor_to_multiply = 1
        conv_factor_to_add = 0
        #lista para guardar as unidades no SI
        SI_unis_list = []
        #lista para checar se já duplicadas de PUs caso houve um erro na extração das unidades
        check_duplicate_PUs = []
        
        #varrendo as unidades encontradas (as três listas do dic tem o mesmo length)
        for i in range(len(units['raw_unit'])):
            
            #caso a PU encontrada seja inversa
            if '–' in units['raw_unit'][i]:
                #invertendo a PU
                inverse_PU = get_physical_unit_inverse( units['SI_unit'][i] )
                check_duplicate_PUs.append(inverse_PU)
                if inverse_PU not in SI_unis_list:            
                    #invertendo o factor de conversão
                    conv_factor_to_multiply = round( conv_factor_to_multiply * ( 1 / units['factor_list'][i] ), 9)
                    SI_unis_list.append( inverse_PU )                
                    print('Conversão de unidade: ', units['raw_unit'][i] , ' > ' , inverse_PU, '( fator: ' , ( 1 / units['factor_list'][i] ) , ' )' )
            else:                
                #não precisa inverter a PU
                direct_PU = units['SI_unit'][i]
                check_duplicate_PUs.append(direct_PU)
                if direct_PU not in SI_unis_list:
                    #caso a conversão seja por somatório
                    if units['factor_operation'][i] == 'add':
                        #caso a conversão seja por multiplicação
                        conv_factor_to_add = conv_factor_to_add + units['factor_list'][i]
                        SI_unis_list.append( direct_PU )                                            
                    elif units['factor_operation'][i] == 'multiply':
                        #caso a conversão seja por multiplicação
                        conv_factor_to_multiply = round( conv_factor_to_multiply * units['factor_list'][i], 9)
                        SI_unis_list.append( direct_PU )
                    print('Conversão de unidade: ', units['raw_unit'][i] , ' > ' , direct_PU, '( fator: ' , units['factor_list'][i], ' ; ', units['factor_operation'][i], ' )' )
        
        #montando as PUs no SI
        SI_units = ''
        for i in range(len(SI_unis_list)):
            if i == len(SI_unis_list) - 1:
                SI_units += SI_unis_list[i]
            else:
                SI_units += SI_unis_list[i] + ' '
                
        print('Converted PUs: ', SI_units)
        
        #testando se há erro na conversão (caso haja unidades duplicadas)
        if max( [check_duplicate_PUs.count(i) for i in check_duplicate_PUs] ) > 1:
            print('Erro de extração das PUs: unidades duplicadas: ', check_duplicate_PUs)
            return None , None, None
        
        else:
            return conv_factor_to_multiply , conv_factor_to_add , SI_units 
        
    #caso nenhuma PU tenha sido identificada ou só parcialmente identificadas
    else:
        return None, None, None
        

#------------------------------
def get_physical_unit_replace_special_chars(unit):    
        
    new_unit = ''
    for char_i in range(len(unit)):
        if unit[char_i] == 'K':
            try:
                if unit[char_i + 1] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    new_unit += 'k'
                else:
                    new_unit += 'K'
            except IndexError:
                new_unit += 'K'
        
        elif unit[char_i] == 'µ':
            new_unit += 'u'
        
        elif unit[char_i] == 'L':
            new_unit += 'l'
            
        else:
            new_unit += unit[char_i]
        
    return new_unit
        


#------------------------------
def get_physical_unit_separate_exponent(unit):
    
    if unit[-1] not in ('23'):
        return unit , '1'
        
    else:
        return unit[ : -1 ] , unit[-1]


#------------------------------
def get_physical_unit_inverse(unit):
    
    if unit[-1] not in ('23'):
        return unit + '–1'
        
    else:
        return unit[ : -1 ] + '–' + unit[-1]


#------------------------------
def list_numerical_parameter():

    base_list = ['concentration_elementcontent',
                 'concentration_mass_mass',
                 'concentration_mass_vol',
                 'concentration_molar',
                 'concentration_hhv',
                 'distance_particlesize',
                 'electricpotential_zeta',
                 'rate_temperature_time', 
                 'ratio_element',
                 'percentage_elementcontent',
                 'percentage_watercontent',                 
                 'ratio_weightweight',
                 'surface_area',
                 'temperature_carbonization', 
                 'time_carbonization']

    mod_list1 = [item +'_inc' for item in base_list]
    mod_list2 = [item +'_dec' for item in base_list]
    
    return base_list + mod_list1 + mod_list2


#------------------------------
def list_textual_parameter():

    return ['activation_chemicals',
            'activation_processes',
            'applications',
            'antidiseases',
            'diseases',
            'diseasesregex',
            'microbes',
            'microbesregex',
            'plants',
            'plantsregex',
            'processes',
            'raw_materials', 
            'synthesis_method',
            'techniques']


#------------------------------
def regex_patt_from_parameter(parameter):

    from functions_PARAMETERS import get_physical_units
    from functions_PARAMETERS import get_physical_units_combined
    import regex as re 
    import time

    print('Encontrando padrão regex para o parâmetro: ', parameter)

    pattern_dic = {}
    #esse termo é para indicar se foi encontrado algum parâmetro
    found_parameter = False
    #esse termo é para procurar parâmetros adimensionais ou porcentagens (ex: elemental ratio)
    find_text_parameter = False

    #dicionário com as unidades físicas        
    PU_unit_dic = get_physical_units(mode = 'SI')
    
    #determinando o parâmetro base (temperature, time, etc)
    #cortando o último arg respectivo a o underline ('_')
    base_parameter_to_find = re.match(r'[a-z]+_', parameter)
    
    #checando se o paramêtro será "inc" ou "dec"
    if parameter[ -4 : ] == '_inc':
        parttern_to_complete = 'increas|rais|ris'
        pattern_dic['find_variation_parameter'] = True
        pattern_dic['parameter_suffix'] = '_inc'
        parameter = parameter[:-4]
    elif parameter[ -4 : ] == '_dec':
        parttern_to_complete = 'decreas|reduc|lower'
        pattern_dic['find_variation_parameter'] = True
        pattern_dic['parameter_suffix'] = '_dec'
        parameter = parameter[:-4]
    else:
        parttern_to_complete = 'increas|rais|ris' + '|' + 'decreas|reduc|lower'
        pattern_dic['find_variation_parameter'] = False
        pattern_dic['parameter_suffix'] = ''
    
    pattern_dic['pattern_variation_parameter'] = r'({text})(e|ing|ed)([\-\–\(\)\w\.\,\s]+(of|by|than))?\s(?!(to|from)\s)(?=[0-9]+(\.[0-9]+)?)'.format(text = parttern_to_complete)
    
    #adicionado padrões numéricos que não podem ser encontrados (ex: P 0.05)
    pattern_dic['aditional_numbers_not_to_find'] = r'(\([Pp](\s[\<\>\=])?\s?0\.[0-9]{1,3}\))'
    
    #encontrando o parâmetro base
    if base_parameter_to_find:
        base_parameter_end_index = re.match(r'[a-z]+_', parameter).span()[1] - 1
        base_parameter = parameter[ : base_parameter_end_index]
    else:
        base_parameter = parameter
        
    #determinando se o parâmetro é combinado (ex: concentração mol L-1) ou single (graus celsius)
    if base_parameter in ('aceleration', 'concentration', 'rate', 'surface', 'velocity'):
        parameter_type = 'combined'
    else:
        parameter_type = 'single'

    #determinando as unidades físicas de interesse
    if parameter.lower() == 'concentration_elementcontent':

        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'weight'
        
        #lista de unidades a serem encontradas        
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3
        ndec_min_len, ndec_max_len = 1 , 4
        find_text_parameter = True
        text_parameter_list = ['TOC',
                               'H',
                               'C',
                               'O',
                               'N',
                               'P',
                               'S',
                               'K']
        found_parameter = True
        
    elif parameter.lower() == 'concentration_mass_mass':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'weight'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 6
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
    
    elif parameter.lower() == 'concentration_mass_vol':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 6
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
    
    elif parameter.lower() == 'concentration_molar':

        pattern_dic['first_parameter'] = 'molarity'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos                
        n_min_len , n_max_len = 1 , 6
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True    

    elif parameter.lower() == 'concentration_hhv':
        
        pattern_dic['first_parameter'] = 'energy'
        pattern_dic['second_parameter'] = 'weight'

        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 6
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
    
    elif parameter.lower() == 'distance_particlesize':
        
        pattern_dic['first_parameter'] = 'distance'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
    
    elif parameter.lower() == 'electricpotential_zeta':
        
        pattern_dic['first_parameter'] = 'electric_potential'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        
    elif parameter[ : ].lower() == 'rate_temperature_time':

        pattern_dic['first_parameter'] = 'temperature'
        pattern_dic['second_parameter'] = 'time'

        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len  = 1 , 3
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True

    elif parameter.lower() == 'ratio_element':

        pattern_dic['first_parameter'] = 'ratio'
        pattern_dic['second_parameter'] = None
        
        #lista de unidades a serem encontradas        
        PU_units_to_find = ['']

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 4
        find_text_parameter = True
        text_parameter_list = ['H#C',
                               'C#H',
                               'C#O',
                               'O#C',
                               'N#C',
                               'C#N',
                               'N#H',
                               'H#N']
        found_parameter = True

    elif parameter.lower() == 'percentage_elementcontent':

        pattern_dic['first_parameter'] = 'percentage'
        pattern_dic['second_parameter'] = None
        
        #lista de unidades a serem encontradas        
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3
        ndec_min_len, ndec_max_len = 1 , 4
        find_text_parameter = True
        text_parameter_list = ['TOC',
                               'H',
                               'C',
                               'O',
                               'N',
                               'P',
                               'S',
                               'K']
        found_parameter = True

    elif parameter.lower() == 'percentage_watercontent':
        
        pattern_dic['first_parameter'] = 'percentage'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas        
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3 
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True

    elif parameter.lower() == 'ratio_weightweight':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'weight'

        #lista de unidades a serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True    

    elif parameter.lower() == 'surface_area':

        pattern_dic['first_parameter'] = 'areametric'
        pattern_dic['second_parameter'] = 'weight'        
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True

    elif parameter[ : ].lower() == 'temperature_carbonization':
        
        pattern_dic['first_parameter'] = 'temperature'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 3 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True        
    
    elif parameter.lower() == 'time_carbonization':
        
        pattern_dic['first_parameter'] = 'time'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 4 
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True

    else:
        print(f'Erro! O parâmetro introduzido ({parameter.lower()}) não foi encontrado')
        print('Ver abaixo os parâmetros definidos:')
        for parameter_set in list_numerical_parameter():
            print(parameter_set)
    
    #padrão regex a ser encontrado
    pattern_dic['PU_to_find_regex'] = None
    #textos a serem encontrados
    pattern_dic['parameters_to_find_in_sent'] = None
    pattern_dic['parameters_pattern_to_find_in_sent'] = None
    #unidades físicas a serem encontradas
    pattern_dic['PUs'] = []
    
    #caso o parâmetro tenha sido encontrado
    if found_parameter is True:
        print('Padrão regex encontrado para o parâmetro: ', parameter)

        #caso seja um parâmetro adimensional
        if find_text_parameter is True:
            
            #construindo o padrão regex para procura
            parameters_pattern_to_find = ''
            parameters_to_find = ''
            for i in range(len(text_parameter_list)):
                if i == len(text_parameter_list) - 1:
                    parameters_to_find += text_parameter_list[i]
                    parameters_pattern_to_find += '(?<![0-9])(\s|\,|;|\(|\[|and)+' + text_parameter_list[i] + '[\s\.\,;:\)\]]'
                else:
                    parameters_to_find += text_parameter_list[i] + '|'
                    parameters_pattern_to_find += '(?<![0-9])(\s|\,|;|\(|\[|and)+' + text_parameter_list[i] + '[\s\.\,;:\)\]]|'
            
            #gerando a pattern de texto para encontrar unidades adimensionais
            pattern_dic['parameters_pattern_to_find_in_sent'] = parameters_pattern_to_find
            pattern_dic['parameters_to_find_in_sent'] = parameters_to_find

        #lista de unidades físicas a serem encontradas
        pattern_dic['PUs'] = PU_units_to_find
                
        #montando os padrões que devem ser encontrados
        PU_combination_to_find = ''
        for i in range(len(PU_units_to_find)):
            if i == len(PU_units_to_find) - 1:
                PU_combination_to_find += PU_units_to_find[i]
            else:
                PU_combination_to_find += PU_units_to_find[i] + '|'

        #guardando as unidades físicas (PUs) a serem encontradas na sentença                
        pattern_dic['PU_to_find_in_sent'] = PU_combination_to_find

        #determinando se o número será necessariamente fracionário ou ficará em aberto no padrão regex
        if parameter.lower() == 'ratio_element':
            decimal_operator = ''
        else:
            decimal_operator = '?'


        #gerando a pattern para encontrar        
        pattern_dic['PU_to_find_regex'] = r'((?<![\>\<]\s)((?<=[0-9]\s*)({parameters_to_find})?(&|to|or|and|\s)+|(to|or|and|\(|\;|\,|\s)+)(–?[0-9]{int_min_len},{int_max_len})(\.[0-9]{dec_min_len},{dec_max_len}){decimal_opt}\s*({parameters_to_find})?)+\s*({parameters_to_find})[\s\.\,;:\)\]]'.format(int_min_len = '{' + str(n_min_len),
                                                                                                                                                                                                                                                                                                 int_max_len = str(n_max_len) + '}',
                                                                                                                                                                                                                                                                                                 dec_min_len = '{' + str(ndec_min_len),
                                                                                                                                                                                                                                                                                                 dec_max_len = str(ndec_max_len) + '}',
                                                                                                                                                                                                                                                                                                 parameters_to_find = PU_combination_to_find,
                                                                                                                                                                                                                                                                                                 decimal_opt = decimal_operator)

        #caso o parâmetro seja simples, adiciona-se todas as combinações que não podem ser encontradas
        if parameter_type == 'single':

            #lista de unidades a não serem encontradas
            PU_units_not_find = get_physical_units_combined(first_parameter = base_parameter, second_parameter = None, get_inverse = False)

            #montando os padrões que não devem ser encontrados
            PU_combination_not_to_find = ''
            for i in range(len(PU_units_not_find)):
                if PU_units_not_find[i][ : len('adimensional ') ] == 'adimensional ':
                    PU_units_not_find_modified = PU_units_not_find[i][ len('adimensional ') :  ]
                else:
                    PU_units_not_find_modified = PU_units_not_find[i]
                
                if i == len(PU_units_not_find) - 1:
                    PU_combination_not_to_find += PU_units_not_find_modified
                else:
                    PU_combination_not_to_find += PU_units_not_find_modified + '|'
            
                #print(PU_combination_not_to_find)
                #time.sleep(1)
            #guardando as unidades físicas (PUs) que não podem ser encontradas na sentença
            pattern_dic['PU_not_to_find_in_sent'] = PU_combination_not_to_find
            
            #montando o regex pattern com a combinação de todas as unidades que não deverão ser encontradas
            pattern_dic['PU_not_to_find_regex'] = r'((?<![\>\<]\s)((?<=[0-9]\s*)({parameters_not_to_find})?(&|to|or|and|\s)+|(to|or|and|\(|\;|\,|\s)+)(–?[0-9]{int_min_len},{int_max_len})(\.[0-9]{dec_min_len},{dec_max_len}){decimal_opt}\s*({parameters_not_to_find})?)+\s*({parameters_not_to_find})[\s\.\,;:\)\]]'.format(int_min_len = '{' + str(n_min_len),
                                                                                                                                                                                                                                                                                                                     int_max_len = str(n_max_len) + '}',
                                                                                                                                                                                                                                                                                                                     dec_min_len = '{' + str(ndec_min_len),
                                                                                                                                                                                                                                                                                                                     dec_max_len = str(ndec_max_len) + '}',
                                                                                                                                                                                                                                                                                                                     parameters_not_to_find = PU_combination_not_to_find,
                                                                                                                                                                                                                                                                                                                     decimal_opt = decimal_operator)

        elif parameter_type == 'combined':
            pattern_dic['PU_not_to_find_regex'] = None

    #print('find_variation_parameter: ', pattern_dic['find_variation_parameter'])
    #print('pattern_variation_parameter: ', pattern_dic['pattern_variation_parameter'])
    #print('PU_to_find_regex: ', pattern_dic['PU_to_find_regex'])
    #print('PU_not_to_find_regex: ', pattern_dic['PU_not_to_find_regex'])            
    #print('parameters_to_find_in_sent: ', pattern_dic['parameters_to_find_in_sent'])
    #print('parameters_pattern_to_find_in_sent: ', pattern_dic['parameters_pattern_to_find_in_sent'])
    
    return pattern_dic


#------------------------------
def replace_numerical_parameters(string_text, char_range_number = 10, check = False):
    
    import regex as re
    #import time
    
    #------------------------------
    def replace_char(char):
        new_char = None
        if char in ('0123456789'):
            new_char = 'z'
        elif char == '.':
            new_char = 'x'
        else:
            new_char = char
        return new_char
    
    #substituindo os parametros numéricos
    #------------------------------
    new_text = ''
    char_range = char_range_number
    pattern = r'([\s\(\[:-‐–−—―][0-9]+[\.,][0-9]+|[\s\(\[:-‐–−—―][0-9]+)'
    match_number = False
    char_break = 0
    for char_index in range(len(string_text)):
        #------------------------------
        #testando os limites do loop
        if char_index > len(string_text) - char_range:
            new_text += string_text[char_index]
            continue
        #------------------------------
        if (match_number is True):            
            if char_index == char_break:
                match_number = False
                if (check is True):
                    print(' After: ', repr(new_text[ -char_range : ] + string_text[ char_index ]))
                    #time.sleep(0.1)
                pass
            else:                
                new_text += replace_char(string_text[char_index])
                continue
        #------------------------------
        if re.match(pattern, string_text[ char_index : char_index + char_range ]):
            match = re.match(pattern, string_text[ char_index : char_index + char_range ])
            char_break = match.end() + char_index
            match_number = True
            new_text += replace_char(string_text[char_index])
            if (check is True):
                print('Before: ', repr(string_text[ char_index - int(char_range/2) : char_index + char_range ]))
            continue
            
        new_text += string_text[char_index]
        
    return new_text
