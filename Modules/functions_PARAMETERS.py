#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_physical_units():

    dic = {}
        
    dic['areaagro'] = ['ha', 'Ha', 'HA']
    dic['areametric'] = ['nm2', 'um2', 'mm2', 'cm2', 'dm2', 'm2', 'km2', 'Km2']
    dic['distance'] = ['nm', 'um', 'mm', 'cm', 'dm', 'm' , 'km' , 'Km']
    dic['energy'] = ['mJ', 'J', 'kJ', 'KJ', 'MJ', 'mcal', 'cal', 'kcal', 'Kcal', 'Mcal']
    dic['electricpotential'] = ['mV', 'V', 'kV', 'KV']
    dic['force'] = ['dyn', 'kdyn', 'Kdyn', 'Mdyn', 'dyne', 'kdyne', 'Kdyne', 'Mdyne', 'nN', 'uN', 'mN', 'cN', 'dN', 'N', 'kN']
    dic['log10'] = ['log10', 'log', 'logs']
    dic['molarity'] = ['umol', 'µmol', 'mmol', 'cmol', 'mol']
    dic['potency'] = ['watts', 'W']
    dic['percentage'] = ['%', 'wtperc', 'wtvperc' , 'volperc' ]
    dic['pressure'] = ['Pa', 'kPa', 'KPa', 'MPa', 'Bar', 'bar', 'kBar', 'kbar', 'KBar', 'Kbar', 'MBar', 'Mbar']
    dic['temperature'] = ['C', '°C', 'K']
    dic['time'] = ['s', 'sec', 'secs', 'Sec', 'Secs', 'min', 'mins', 'Min', 'Mins', 'h', 'hour', 'hours', 'Hour', 'Hours', 
                    'day', 'days', 'Day', 'Days', 'year', 'years', 'Year', 'Years', 'yr', 'yrs']
    dic['viscosity'] = ['cP', 'mP']
    dic['volume'] = ['mm3', 'dm3', 'cm3', 'cc', 'm3', 'ul', 'uL', 'ml', 'mL',  'l', 'L']
    dic['weight'] = ['ng', 'ug', 'mg', 'g', 'kg', 'Kg', 't','ton', 'tonne', 'TON']

    dic['all'] = []
    for key in dic.keys():
        dic['all'].extend(dic[key])
    
    return dic


#------------------------------
def get_physical_units_combined(first_parameter = '', second_parameter = None, get_inverse = False, mode = 'all'):
        
    #motando as combinações de unidades físicas
    PU_units_combined = {}
    PU_units_combined['separated'] = []
    PU_units_combined['joint'] = []

    #coletando todas as unidades físicas
    PU_dic = get_physical_units()

    #varrendo as unidades físicas da classe primária
    for unit1 in PU_dic[first_parameter]:        
        
        #varrendo todas as classe de unidades físicas
        for key in PU_dic.keys():

            #caso se queira obter uma unidade combinada específica. EX: mg L
            if (second_parameter is not None) and (key != second_parameter):
                continue
            
            #caso se queria obter todas as combinações de unidades
            elif (second_parameter is None) or (second_parameter == key):
                #fazendo a combinação com todos os parâmetros
                if get_inverse is False:
                    #obtendo as unidades como str
                    PU_units_combined['joint'].extend( [ ( unit1 + ' ' +  unit2 ) for unit2 in PU_dic[key] ] )
                    #obtendo as unidades como tuplas
                    PU_units_combined['separated'].extend( [ ( unit1, unit2 ) for unit2 in PU_dic[key] ] )
                #encontrando as unidades inversas de todas os outros parâmetros e combinando o parâmetro introduzido
                elif get_inverse is True:
                    #obtendo as unidades como str
                    PU_units_combined['joint'].extend( [ ( unit1 + ' ' +  get_physical_unit_inverse(unit2) ) for unit2 in PU_dic[key] ] )
                    #obtendo as unidades como tuplas
                    PU_units_combined['separated'].extend( [ ( unit1, get_physical_unit_inverse(unit2) ) for unit2 in PU_dic[key] ] )
    
    return PU_units_combined


#------------------------------
def get_physical_all_units_combined():

    PU_all_units_combined = {}
    PU_all_units_combined['separated'] = []
    PU_all_units_combined['joint'] = []

    PU_units = get_physical_units()

    for unit in PU_units:
        
        PU_units_combined = get_physical_units_combined(first_parameter = unit)
        
        PU_all_units_combined['separated'].extend( PU_units_combined['separated'] )
        PU_all_units_combined['joint'].extend( PU_units_combined['joint'] )

    return PU_all_units_combined


#------------------------------
#essa função é usada na extração dos dados numéricos das sentenças
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
        
        #porcentagem
        if PU in PU_units['percentage']:
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
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['Km2', 'km2']] + ['Km2', 'km2']:
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
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['KJ', 'kJ']] + ['KJ', 'kJ']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['MJ']] + ['MJ']:
                units['factor_list'].append(1e6)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mcal']] + ['mcal']:
                units['factor_list'].append(1e-3)                
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['cal']] + ['cal']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['Kcal', 'kcal']] + ['Kcal', 'kcal']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['Mcal']] + ['Mcal']:
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
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['Km', 'km']] + ['Km', 'km']:
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
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['KV', 'kV']] + ['KV', 'kV']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')

        #force
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['force']] + PU_units['force']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('N')
            #encontrando os fatores de conversão            
            if PU in [ get_physical_unit_inverse(unit) for unit in ['nN']] + ['nN']:
                units['factor_list'].append(1e-9)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['uN']] + ['uN']:
                units['factor_list'].append(1e-6)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mN']] + ['mN']:
                units['factor_list'].append(1e-3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['cN']] + ['cN']:
                units['factor_list'].append(1e-2)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['dN']] + ['dN']:
                units['factor_list'].append(1e-1)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['N']] + ['N']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['KN', 'kN']] + ['KN', 'kN']:
                units['factor_list'].append(1e3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['Mdyn', 'Mdyne']] + ['Mdyn', 'Mdyne']:
                units['factor_list'].append(1e-5*1e6)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['dyne', 'dyn']] + ['dyne', 'dyn']:
                units['factor_list'].append(1e-5)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['Kdyne', 'kdyne', 'Kdyn', 'kdyn']] + ['Kdyne', 'kdyne', 'Kdyn', 'kdyn']:
                units['factor_list'].append(1e-5*1000)
                units['factor_operation'].append('multiply')

        #log10 - microbe killing PDT
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['log10']] + PU_units['log10']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('log10')
            if PU in [ get_physical_unit_inverse(unit) for unit in ['log10', 'log', 'logs']] + ['log10', 'log', 'logs']:
                units['factor_list'].append(1)
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

        #pressure
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['pressure']] + PU_units['pressure']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('Pa')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['Pa']] + ['Pa']:
                units['factor_list'].append( 1 )
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['kPa', 'KPa']] + ['kPa', 'KPa']:
                units['factor_list'].append( 1e3 )
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['MPa']] + ['MPa']:
                units['factor_list'].append( 1e6 )
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['Bar', 'bar']] + ['Bar', 'bar']:
                units['factor_list'].append( 1e5 )
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['kBar', 'kbar', 'KBar', 'Kbar']] + ['kBar', 'kbar', 'KBar', 'Kbar']:
                units['factor_list'].append( 1e5 * 1e3 )
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['MBar', 'Mbar']] + ['MBar', 'Mbar']:
                units['factor_list'].append( 1e5 * 1e6 )
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

        #viscosity
        elif PU in [ get_physical_unit_inverse(unit) for unit in PU_units['viscosity']] + PU_units['viscosity']:
            #colocando a unidade SI na lista
            units['SI_unit'].append('P')
            #encontrando os fatores de conversão
            if PU in [ get_physical_unit_inverse(unit) for unit in ['cP']] + ['cP']:
                units['factor_list'].append( 1e-2 )
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mP']] + ['mP']:
                units['factor_list'].append( 1e-3 )
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
            if PU in [ get_physical_unit_inverse(unit) for unit in ['ng']] + ['ng']:
                units['factor_list'].append(1e-9)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['ug']] + ['ug']:
                units['factor_list'].append(1e-6)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['mg']] + ['mg']:
                units['factor_list'].append(1e-3)
                units['factor_operation'].append('multiply')
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['g']] + ['g']:
                units['factor_list'].append(1)
                units['factor_operation'].append('multiply')                
            elif PU in [ get_physical_unit_inverse(unit) for unit in ['kg', 'Kg']] + ['kg', 'Kg']:
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
        SI_units_list = []
        
        #varrendo as unidades encontradas (as três listas do dic tem o mesmo length)
        for i in range(len(units['raw_unit'])):
            
            #caso a PU encontrada seja inversa
            if '-' in units['raw_unit'][i]:
                #invertendo a PU
                inverse_PU = get_physical_unit_inverse( units['SI_unit'][i] )
                if inverse_PU not in SI_units_list:            
                    #invertendo o factor de conversão
                    conv_factor_to_multiply = round( conv_factor_to_multiply * ( 1 / units['factor_list'][i] ), 9)
                    SI_units_list.append( inverse_PU )                
                    #print('Conversão de unidade: ', units['raw_unit'][i] , ' > ' , inverse_PU, '( fator: ' , ( 1 / units['factor_list'][i] ) , ' ; multiply )' )
            else:                
                #não precisa inverter a PU
                direct_PU = units['SI_unit'][i]
                if direct_PU not in SI_units_list:
                    #caso a conversão seja por somatório
                    if units['factor_operation'][i] == 'add':
                        #caso a conversão seja por multiplicação
                        conv_factor_to_add = conv_factor_to_add + units['factor_list'][i]
                        SI_units_list.append( direct_PU )                                            
                    elif units['factor_operation'][i] == 'multiply':
                        #caso a conversão seja por multiplicação
                        conv_factor_to_multiply = round( conv_factor_to_multiply * units['factor_list'][i], 9)
                        SI_units_list.append( direct_PU )
                    #print('Conversão de unidade: ', units['raw_unit'][i] , ' > ' , direct_PU, '( fator: ' , units['factor_list'][i], ' ; ', units['factor_operation'][i], ' )' )
        
        #montando as PUs no SI
        SI_units = ''
        for i in range(len(SI_units_list)):
            if i == len(SI_units_list) - 1:
                SI_units += SI_units_list[i]
            else:
                SI_units += SI_units_list[i] + ' '
                
        #print('Converted PUs: ', SI_units)
            
        #time.sleep(5)
        return conv_factor_to_multiply , conv_factor_to_add , SI_units 
    
    #caso nenhuma PU tenha sido identificada ou só parcialmente identificadas
    else:
        print('Erro de extração das PUs: unidade não identificada.')
        return None, None, None


#------------------------------
def get_physical_unit_exponent(unit):
    
    if unit[-1] not in ('23456789'):
        return unit , '1'
        
    else:
        return unit[ : -1 ] , unit[-1]


#------------------------------
def get_physical_unit_inverse(unit):
    
    if unit[-1] not in ('23'):
        return unit + '-1'
        
    else:
        return unit[ : -1 ] + '-' + unit[-1]


#------------------------------
def list_numerical_parameter():

    #ATENÇÃO: ao adicionar uma nova unidade numérica, o primeiro nome deve estar presenta na lista da função "get_physical_units"    
    #ou se for uma unidade composta, deve ser descrita na função "regex_patt_from_parameter" na lista de base_parameters

    base_list = ['concentration_mass_mass',
                 'concentration_mass_vol',
                 'concentration_molar',
                 'elementcontent',
                 'distance',
                 'microbekilling_log10',
                 'percentage',
                 'surface_area',
                 'surface_tension',
                 'temperature',
                 'time',
                 'viscosity_cp',
                 'volume',
                 'weight',
                 'zeta_potential'
                 ]

    mod_list1 = [item +'_inc' for item in base_list]
    mod_list2 = [item +'_dec' for item in base_list]
    
    return base_list + mod_list1 + mod_list2


#------------------------------
def list_textual_parameter(diretorio = None):

    from FUNCTIONS import get_filenames_from_folder
    import regex as re

    parameter_list = []
    filenames = get_filenames_from_folder(diretorio + '/Outputs/ngrams/semantic', file_type = 'csv')
    for filename in filenames:
        parameter_list.append( re.search(r'(?<=n[12x]gram_).+', filename).group() )

    return parameter_list


#------------------------------
def regex_patt_from_parameter(parameter):

    from functions_PARAMETERS import get_physical_units
    from functions_PARAMETERS import get_physical_units_combined
    import regex as re 
    #import time

    print('Encontrando padrão regex para o parâmetro: ', parameter)

    pattern_dic = {}
    #esse termo é para indicar se foi encontrado algum parâmetro
    found_parameter = False

    #checando se o paramêtro será "inc" ou "dec"
    parameter_suffix = None
    if parameter[ -4 : ] == '_inc':
        parameter = parameter[ : -4 ]
        parameter_suffix = '_inc'
    elif parameter[ -4 : ] == '_dec':
        parameter = parameter[ : -4 ]
        parameter_suffix = '_dec'

    #dicionário com as unidades físicas        
    PU_unit_dic = get_physical_units()
    
    #determinando as unidades físicas de interesse
    if parameter.lower() == 'concentration_mass_mass':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'weight'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True, mode = 'all')
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 6
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
        parameter_type = 'combined'
    
    elif parameter.lower() == 'concentration_mass_vol':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True, mode = 'all')
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 6
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
        parameter_type = 'combined'
    
    elif parameter.lower() == 'concentration_molar':

        pattern_dic['first_parameter'] = 'molarity'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True, mode = 'all')

        #determinando o número mínimo e máximo de caracteres numéricos                
        n_min_len , n_max_len = 1 , 6
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'distance':
        
        pattern_dic['first_parameter'] = 'distance'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'single'
        
    elif parameter.lower() == 'microbekilling_log10':
        
        pattern_dic['first_parameter'] = 'log10'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'percentage':
        
        pattern_dic['first_parameter'] = 'percentage'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas        
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3 
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'surface_area':

        pattern_dic['first_parameter'] = 'areametric'
        pattern_dic['second_parameter'] = 'weight'        
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True, mode = 'all')

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'surface_tension':

        pattern_dic['first_parameter'] = 'force'
        pattern_dic['second_parameter'] = 'distance'        
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True, mode = 'all')

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'combined'

    elif parameter[ : ].lower() == 'temperature':
        
        pattern_dic['first_parameter'] = 'temperature'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 3 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'single'
    
    elif parameter.lower() == 'time':
        
        pattern_dic['first_parameter'] = 'time'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 4 
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'viscosity_cp':
        
        pattern_dic['first_parameter'] = 'viscosity'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 4 
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'single'

    elif parameter[ : ].lower() == 'volume':
        
        pattern_dic['first_parameter'] = 'volume'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'single'

    elif parameter[ : ].lower() == 'weight':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 4
        ndec_min_len, ndec_max_len = 1 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'zeta_potential':
        
        pattern_dic['first_parameter'] = 'electric_potential'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 3
        ndec_min_len, ndec_max_len = 1 , 2
        found_parameter = True
        parameter_type = 'single'

    else:
        print(f'Erro! O parâmetro introduzido ({parameter.lower()}) não foi encontrado')
        print('Ver abaixo os parâmetros definidos:')
        for parameter_set in list_numerical_parameter():
            print(parameter_set)
    
    
    #caso o parâmetro tenha sido encontrado
    if found_parameter is True:
        
        print('Padrão regex encontrado para o parâmetro: ', parameter)
        
        #definindo a lista de PUs 
        if parameter_type == 'single':
            list_PUs_to_find = PU_units_to_find
        elif parameter_type == 'combined':
            list_PUs_to_find = PU_units_to_find['joint']

        #lista de unidades físicas a serem encontradas
        pattern_dic['PUs'] = list_PUs_to_find

        #montando os padrões que devem ser encontrados
        PUs_to_find = ''
        for i in range(len(list_PUs_to_find)):            
            if i == len(list_PUs_to_find) - 1:
                PUs_to_find += list_PUs_to_find[i]
            else:
                PUs_to_find += list_PUs_to_find[i] + '|'


        #montando o regex pattern
        if parameter_suffix == '_inc':
            pre_pattern_to_complete = 'increas|rais|ris|rais|enhanc|grow'
            pre_pattern = '({pre_pattern_to_complete})(e|ing|ed)\s*([\-\s\,\;\.\(\[\)\]]|by)?\s*'.format(pre_pattern_to_complete = pre_pattern_to_complete)
        elif parameter_suffix == '_dec':
            pre_pattern_to_complete = 'decreas|reduc|lower|diminish|reduc'
            pre_pattern = '({pre_pattern_to_complete})(e|ing|ed)\s*([\-\s\,\;\.\(\[\)\]]|by)?\s*'.format(pre_pattern_to_complete = pre_pattern_to_complete)
        else:
            pre_pattern = ''

        num_pattern = '\-?\s*[0-9]{n_min_len},{n_max_len}\.?[0-9]{ndec_min_len},{ndec_max_len}?'.format(n_min_len = '{' + str(n_min_len), 
                                                                                                        n_max_len = str(n_max_len) + '}', 
                                                                                                        ndec_min_len = '{' + str(ndec_min_len),
                                                                                                        ndec_max_len = str(ndec_max_len) + '}')
        front_spacer = '([\-\s\,\;\.\(\[\)\]]*(and|or|to|from)?[\-\s\,\;\.\(\[\)\]]+)+'
        back_spacer = '[\-\s\,\;\.\(\[\)\]]'

        initial_pattern = '({front_spacer}({num_pattern})\s*({PUs_to_find})?)?'.format(front_spacer = front_spacer, num_pattern = num_pattern, PUs_to_find = PUs_to_find)
        mid_pattern = '{front_spacer}({num_pattern})\s*({PUs_to_find}){back_spacer}'.format(front_spacer = front_spacer, num_pattern = num_pattern, PUs_to_find = PUs_to_find, back_spacer = back_spacer)
        
        #caso seja um parâmetro do tipo single, não se quer encontrar outros PUs em frente a ele
        if parameter_type == 'single':

            #montando os padrões que não devem ser encontrados
            list_PUs_not_to_find = PU_unit_dic['all'] + [ get_physical_unit_inverse(unit) for unit in PU_unit_dic['all'] ]
            
            PUs_not_to_find = ''
            for i in range(len(list_PUs_not_to_find)):
                if i == len(list_PUs_not_to_find) - 1:
                    PUs_not_to_find += list_PUs_not_to_find[i]
                else:
                    PUs_not_to_find += list_PUs_not_to_find[i] + '|'
                
            last_pattern = '(?!({PUs_not_to_find}){back_spacer})'.format(PUs_not_to_find = PUs_not_to_find, back_spacer = back_spacer)

        elif parameter_type == 'combined':
            last_pattern = '()'


        #gerando a pattern para encontrar
        pattern_dic['PU_to_find_regex'] = r'{pre_pattern}{initial_pattern}{initial_pattern}{initial_pattern}{initial_pattern}{initial_pattern}{mid_pattern}{last_pattern}'.format(pre_pattern = pre_pattern,
                                                                                                                                                                                  initial_pattern = initial_pattern, 
                                                                                                                                                                                  mid_pattern = mid_pattern,
                                                                                                                                                                                  last_pattern = last_pattern)

    #print('PU_to_find_regex: ', pattern_dic['PU_to_find_regex'])    
    
    return pattern_dic