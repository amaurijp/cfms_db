#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#------------------------------
def break_text_in_sentences(text, min_tokens_to_break_sent = 5):
        
    #import time
    import regex as re
    
    sentence_list = []
    
    cumulative_index = 0
    got_first_sent = False
    while True:
        text_fraction = text[ cumulative_index : ]
        match = re.search(r'(?<!(Figure|Fig|Table|Suppl|et\.?\sal|wt%?|i\.e|e\.g|No|Co|Ltd|i.d|\ssp|\sspp))[\.\?\!](?=\s)', text_fraction)
        if match:
            sentece_end_pos = match.end()
            if got_first_sent is False:
                cumulative_index += sentece_end_pos + 1
                got_first_sent = True
            else:
                sent = text_fraction[ : sentece_end_pos ]
                #tamanho mínimo para ser considerado uma sentença
                if len(sent.split()) >= min_tokens_to_break_sent:
                    sentence_list.append( sent )
                    #print(text_fraction[ : sentece_end_pos ])
                    #time.sleep(1)
                cumulative_index += sentece_end_pos + 1
        else:
            break
    
    return sentence_list
        

#------------------------------
def check_text_language(text, language):
    
    import regex as re
    
    check_language = False
    
    if language.lower() == 'english':        
        matches = re.findall('\sthe\s', text)
        if len(matches) >= 40:
            print('Language found: ', language)
            check_language = True
    
    elif language.lower() == 'portuguese':        
        matches = re.findall('\so\s', text)
        if len(matches) >= 40:
            print('Language found: ', language)
            check_language = True
    
    return check_language


#------------------------------
def concat_DF_sent_indexes(sent_index, n_sent_to_concat):
    
    sent_index_range = []
    
    #caso o numero de sentença a serem concatenadas seja par
    while n_sent_to_concat > 2:
        
        if n_sent_to_concat % 2 == 0:            
            delta_index = int(n_sent_to_concat/2)
            #nesse caso, teremos dois sent_index_range a serem testados
            for i in range(delta_index):
                sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) + i + 1 , sent_index + int(n_sent_to_concat/2) + i + 1) ) )
                sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) - i , sent_index + int(n_sent_to_concat/2) - i) ) )
                
        #caso o numero de sentença a serem concatenadas seja impar
        else:        
            delta_index = int( (n_sent_to_concat - 1)/2)
            for i in range(delta_index):      
                sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) + i , sent_index + int(n_sent_to_concat/2) + i + 1) ) )
                sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) - i - 1 , sent_index + int(n_sent_to_concat/2) - i) ) )
            sent_index_range.append( list( range(sent_index , sent_index + n_sent_to_concat) ) )
        
        n_sent_to_concat -= 1
        
    #print(sent_index_range)
    return sent_index_range


#------------------------------
def correct_period_and_spaces(string_text):
    #montando a lista de pattern e substituições    
    pattern_sub_list = [
                        [r'\.', ' . ', 'filtro de "."', False ],
                        [r'\s{2,20}', ' ', 'filtro de "\s" juntos', False ]
                        ]
    string_text = filter_sub(pattern_sub_list, string_text)
    
    return string_text


#------------------------------
def exist_term_in_string(string = 'string', terms = ['term1', 'term2']):

    found_term = False
    for term in terms:
        term_len = len(term)
        #print('Searching: ', string, '; Term: ', term)
        for char_N in range(len(string)):
            if string[ char_N : char_N + term_len ].lower() == term.lower():
                found_term = True
    
    return found_term


#------------------------------
#definindo a função para modificação do texto raw
#essa função serve para modificar parcialmente o match do REGEX; uma parte do texto é preservada
def filter1(pattern, string_text, text_to_insert = '', get_char_begin = 0, get_char_end = 0, 
            filter_name = 'filtro', check = False):
    
    import time
    import regex as re
    
    new_text = ''
    check_first_match = False
    last_end_index = 0

    if (check is True):
        print('PATTERN: ', pattern)
        print('***', repr(filter_name), '***')
    
    #varrendo o texto
    matches = re.finditer(pattern, string_text)
    for match in matches:
        if match:
            #span do match
            start_index, end_index = match.span()
            #pegando a fração do texto que houve o match
            text_fraction = string_text[ start_index : end_index]
            if get_char_end != 0:
                end_text = text_fraction[ -get_char_end : ]
            else:
                end_text = ''
            #caso seja o primeiro match
            if check_first_match is False:
                new_text += string_text[ last_end_index : start_index ] + text_fraction[ : get_char_begin ] + text_to_insert + end_text
                last_end_index = end_index
                check_first_match = True
            else:
                new_text += string_text[ last_end_index : start_index ] + text_fraction[ : get_char_begin ] + text_to_insert + end_text
                last_end_index = end_index
            
            if (check is True):
                
                print('ANTES: ', string_text[ start_index -5 : end_index +5])
                print('DEPOIS: ', new_text[ -20 : ])
                time.sleep(2)
                
    new_text += string_text[ last_end_index : ]            

    return new_text


#------------------------------ 
#filtrando as sentenças
def filter2(sent, text_break_function = 'spacy', min_tokens_per_sent = 5):

    import regex as re
    
    sent_str = str(sent)
    #print('Sentence before: ', sent_str)
    get_sentence = True       
    if sent_str[0] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': #caso o primeiro character não seja maíusculo ou seja um número
        get_sentence = False
    
        '''Essa parte não está sendo usada
        for char_N in range(len(sent_str)):
            if (sent_str[char_N] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') and (len(sent_str[ char_N : ]) >= min_chars_per_sent):
                sent_str = sent_str[ char_N : ]
                get_sentence = True
                break
            else:
                get_sentence = False
                continue'''
                
    else:
        if len(sent_str.split()) >= min_tokens_per_sent:
            get_sentence = True
        else:
            #print('\nSentença apagada: ', sent_str)
            get_sentence = False

    #------------------------------        
    #eliminando sentenças sem verbo ou sem substantivo
    if (get_sentence is True):
        if text_break_function == 'spacy':
            proc_sent = sent
            counter_verb = 0
            counter_noun = 0
            for token in proc_sent:
                if token.pos_.lower() == 'verb':
                    counter_verb +=1
                elif token.pos_.lower() == 'noun':
                    counter_noun +=1
                else:
                    continue
            if counter_verb == 0 or counter_noun == 1:
                #print('\nSentença apagada (f3): ', sent_str)
                get_sentence = False
        
    #------------------------------        
    if (get_sentence is False):
        #print('Sentença apagada: ', sent_str)
        pass
        
    return sent_str, get_sentence


#------------------------------
#ATENÇÃO: QUALQUER MUDANÇA NO TEXTO QUE MUDE O INDEX DOS CHARS DEVE SER FEITA NESSA FUNÇÃO
def filter_chars(string_text, diretorio = None):
        
    import time
    from functions_PARAMETERS import get_physical_units
    from functions_PARAMETERS import get_physical_unit_separate_exponent
    from functions_PARAMETERS import get_physical_unit_replace_special_chars
    from functions_PARAMETERS import get_physical_units_combined_list
    
    print('Aplicando os filtros de texto...')
    
    #primeiro filtro
    #------------------------------
    #caracteres que são aceitos
    letter_min = 'abcdefghijklmnopqrstuvwxyz'
    letter_cap = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789₂₃₄₅₆' #os numeros em subscrito são aceitos por causa das formulas químicas
    greekletters = 'αβßµελζδΦηρπγƟθø∂Δ∆△ϑστφχψωΩΩƩ∑Σ℃'
    special_chars = '\n\t'
    pontuation = ' ' + "'" + '"'
    others = '+-‐–−—―─‑‒=_;:.,!?@#$&*/%(){}[](̴'
    accepted_chars = letter_min + letter_cap + numbers + greekletters + special_chars + pontuation + others
    #------------------------------
    #carcateres que não são aceitos
    not_accepted_chars = 'ⅠⅡ¹²³×âªÅÅÅÃÄÁåãąáàâąäæＢČČÇçččćç¢ÉÈЕęěëęéêèė𝐺𝐻ï𝑖Íìı́£𝑛ñń𝑁ŃÑñňńÕÖööðõóòôôØ𝑃Úúüřşšš𝑆ŠŚŠŠś𝑡x̄𝑥żŻźŁΠ■≈⩽⩾�§Ɨ€∫ⅢⅣⅥ↑…≡' +\
                            '√±÷~∼☆≪＜<>＞≤≥þ½¼‰©™üÞǂ‡⁎∗˚°⁰ºᵒ◦∘íý®€�ı•ł\\⎢⎟⎣⎜⎡⎤⎦⎥⎛⎝⎞⎠）〈〉○□●△▲▼▽◊♦►→∞⇑⇔↔⇆⇋①②⑦†→˭··∙⋅`′´″’‘；˘ˆ^¡„‟ʹ“”|' + '\r' +\
                                '\uf0b0'
    #------------------------------
    new_text = ''
    for char_index in range(len(string_text)):
        #------------------------------
        #testando os caracters aceitos
        if string_text[char_index] in accepted_chars:
            new_text += string_text[char_index]
            continue
        else:
            #characteres não aceitos
            if string_text[char_index] not in not_accepted_chars:
                try:
                    #print('Exceção de character encontrado: ', repr(string_text[char_index]))
                    with open(diretorio + '/Outputs/NotFoundChars.txt', 'a') as file:
                        file.write(repr(string_text[ char_index - 5 : char_index + 5 ]) + ' ' + repr(string_text[char_index]) + '\n')
                        file.close()
                except IndexError:
                    pass
            new_text += ' '
            continue       
    

    #segundo filtro
    #------------------------------
    #eliminando o hífen colado com a quebra de parágrafo
    final_text = filter1(r'[\-\‐\–\−\—\―\─\‑\‒]\n[a-z]', new_text, get_char_begin = 0, get_char_end = 1, filter_name = 'quebra de texto com paragraph', check = False) #checado
    
    counter = 1
    while counter != 0:
        varied_chars = '\+\@\_\-\‐\–\−\—\―\─\‑\‒\,\:\.;\?\!\(\)\[\]\{\}\~\=\%/&'
        #montando a lista de pattern e substituições    
        pattern_sub_list = [
                            [r'\t{1,50}', '', 'filtro de \t', False ],
                            [r'\n{1,50}', ' ', 'filtro de "\n"', False ],
                            [r'\(?http[s]?\:?[\w{varied_chars}]+[\)\]\,\:\.;]*'.format(varied_chars = varied_chars), '', 'filtro http', False ],
                            [r'\(?[\w{varied_chars}]*(www|org|com)\.[\w{varied_chars}]+[\)\]\,\:\.;]*'.format(varied_chars = varied_chars), '', 'filtro www', False ],
                            [r'\(?[\w{varied_chars}]*(doi|DOI)[\.\:\s]+[\w{varied_chars}]+[\)\]\,\:\.;]*'.format(varied_chars = varied_chars), '', 'fitro DOI', False ],
                            [r'(mailto:\s?|E-mail address[es]*:\s?)?[\w{varied_chars}]+@[\w{varied_chars}]+\.[\w{varied_chars}]+[\)\]\,\:\.;]*'.format(varied_chars = varied_chars), '', 'fitro email', False ],
                            [r'[\s\,\.;]et\.? al[\.\s]', '', 'filtro et al.', False ],
                            #[r'[\(\s]Table[s\.]*(\sS?[\(0-9][/0-9a-z\(\)\,]*)+(\sand\sS?[\(0-9][/0-9a-z\(\)]*)?', ' TABLE', 'filtro de Table', False ],
                            #[r'[\(\s]Fig[s\.]*(\sS?[\(0-9][/0-9a-z\(\)\,]*)+(\sand\sS?[\(0-9][/0-9a-z\(\)]*)?', ' FIGURE', 'filtro de Figura', False ],
                            #[r'[\(\s]Figure[s\.]*(\sS?[\(0-9][/0-9a-z\(\)\,]*)+(\sand\sS?[\(0-9][/0-9a-z\(\)]*)?', ' FIGURE', 'filtro de Figura', False ],
                            #[r'[\(\s]Supplementary\s[Ii]nformation\)?', ' SI', 'filtro de SI', False ],
                            #[r'[\(\s]Supporting\s[Ii]nformation\)?', ' SI', 'filtro de SI', False ],
                            [r'\s*[\(\[][^\(\[]*[\,;\s]*[12][09][0-9][0-9][a-z\,;\s]*[\)\]]', '', 'fitro de ref entre parêntesis', False ],
                            [r'\s{2,50}', ' ', 'filtro de "\s" juntos', False ]
                            ]
        final_text = filter_sub(pattern_sub_list, final_text)
        counter -= 1


    #terceiro filtro
    #------------------------------
    #primeira correção de erros no parsing de unidades físicas
    filter1(r'[\s\(\[\,][8o]\s*C[\s\)\]\,\:\.;]', final_text, text_to_insert = 'C', get_char_begin = 1, get_char_end = 1, filter_name = 'erro unidade de temperatura ( 8 C)', check = False)


    #quarto filtro
    #------------------------------
    #padrão para eliminar referências coladas do tipo 'dsds fdfd [1,2,6-8].' e 'fdsfsd [13] fsdfds' (padrão Elsevier)
    final_text = filter1(r'(\[[0-9]+[0-9a-z\,\-\‐\–\−\—\―\─\‑\‒\s;]*\][\s\.\,\:;])', final_text, get_char_begin = 0, get_char_end = 1, filter_name = 'referências entre colchete (Elsevier)', check = False)
    #padrão para eliminar referências coladas do tipo 'dsds fdfd.1,2,6-8' e ' fdsfsd,[1,3-4]' (padrão ACS)
    final_text = filter1(r'([A-Za-z\)][A-Za-z\.\)][\.,;]\[?[0-9]+[a-z]?[0-9a-z\,\-\‐\–\−\—\―\─\‑\‒;]*\]?[A-Z\s])', final_text, get_char_begin = 3, get_char_end = 1, filter_name = 'referências numéricas (ACS)', check = False)


    #quinto filtro
    #------------------------------
    #padronizando unidades físicas
    
    #carregando todas as possíveis unidades físicas    
    PUs = get_physical_units(mode = 'all')

    #rodando duas vezes os filtros
    counter = 2
    while counter != 0:

        #varrendo as unidades simples
        for key in PUs.keys():
            for unit in PUs[key]:                
                #juntando as grandezas com as unidades (ex: km -> k m, m g- -> mg-)
                if len(unit) > 1:                    
                    serated_unit_chars = ''
                    for char in unit:
                        if len(serated_unit_chars) == 0:
                            serated_unit_chars += '{char}\s'.format(char = char)
                        else:
                            serated_unit_chars += '{char}\s?'.format(char = char)
                    #eliminando o "\s?"
                    serated_unit_chars = serated_unit_chars[ : -3 ]
                    
                    #substituindo quando as grandezas estão separadas das unidades (ex: m g -> mg)                
                    final_text = filter1(r'[\s\(\[\,]{serated_unit_chars}[\s\(\)\[\]\.\,\:;\-\‐\–\−\—\―\─\‑\‒]'.format(serated_unit_chars = serated_unit_chars),
                                     final_text, 
                                     text_to_insert = '{unit}'.format(unit = unit),
                                     get_char_begin = 1,
                                     get_char_end = 1,                              
                                     filter_name = '{serated_unit_chars} -> {unit}'.format(serated_unit_chars = serated_unit_chars, unit = unit),
                                     check = False)

        
        #varrendo as combinações de unidades físicas
        PU_units_combined_list = get_physical_units_combined_list()
        for combined_unit in PU_units_combined_list:
                    
            #determinando o expoente da unidade física
            base_unit , exponent = get_physical_unit_separate_exponent(combined_unit)
                      
            #substituindo caracteres especiais nas unidades como "K" e "µ"
            base_unit_replaced = get_physical_unit_replace_special_chars(base_unit)
            
            #juntando a unidade combinada com o expoente (ex: g l 1 -> g l–1, mol cm 3 -> mol cm-3)
            final_text = filter1(r'[\s\(\[\,]{base_unit}[\s\-\‐\–\−\—\―\─\‑\‒]*{exponent}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit , exponent = exponent),
                                 final_text, 
                                 text_to_insert = '{base_unit_replac}–{exponent}'.format(base_unit_replac = base_unit_replaced , exponent = exponent),
                                 get_char_begin = 1, 
                                 get_char_end = 1,                         
                                 filter_name = '{base_unit} {exponent} -> {base_unit_replac}–{exponent}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced, exponent = exponent),
                                 check = False)
        
        #varrendo as unidades simples
        for key in PUs.keys():
            for unit in PUs[key]:
    
                #determinando o expoente da unidade física
                base_unit , exponent = get_physical_unit_separate_exponent(unit)
                          
                #substituindo caracteres especiais nas unidades como "K" e "µ"
                base_unit_replaced = get_physical_unit_replace_special_chars(base_unit)               
                
                #juntando a unidade com o inverso do expoente (ex: m -1 -> m–1 ou m-1 -> m–1)
                final_text = filter1(r'[\s\(\[\,]{base_unit}\s*[\-\‐\–\−\—\―\─\‑\‒]\s*{exponent}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit , exponent = exponent), 
                                     final_text, 
                                     text_to_insert = '{base_unit_replac}–{exponent}'.format(base_unit_replac = base_unit_replaced , exponent = exponent),
                                     get_char_begin = 1, 
                                     get_char_end = 1,                                 
                                     filter_name = '{base_unit} -{exponent} -> {base_unit_replac}–{exponent}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced, exponent = exponent),
                                     check = False)
                
                #caso a unidade tenha potência (ex: mm2, m3, etc)
                if exponent in ('2', '3'):
                    
                    #substituindo as unidades no denominador que tenham potência (ex: /mm 2 -> mm–2)
                    final_text = filter1(r'[/=]\s*{base_unit}\s*{exponent}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit , exponent = exponent), 
                                         final_text, 
                                         text_to_insert = ' {base_unit_replac}–{exponent}'.format(base_unit_replac = base_unit_replaced, exponent = exponent),
                                         get_char_begin = 0, 
                                         get_char_end = 1,                                      
                                         filter_name = '/{base_unit}{exponent} -> {base_unit_replac}–{exponent}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced, exponent = exponent), 
                                         check = False)             
                    
                    #juntando a unidade com o expoente (ex: mm 2 -> mm2)
                    final_text = filter1(r'[\s\(\[\,]{base_unit}\s*{exponent}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit , exponent = exponent),
                                         final_text, 
                                         text_to_insert = '{base_unit_replac}{exponent}'.format(base_unit_replac = base_unit_replaced , exponent = exponent),
                                         get_char_begin = 1, 
                                         get_char_end = 1,                                      
                                         filter_name = '{base_unit} {exponent} -> {base_unit_replac}{exponent}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced, exponent = exponent),
                                         check = False)
    
                    #quando o número está colado com a unidade (ex: 2m 2 -> 2 m2)
                    final_text = filter1(r'[0-9]{base_unit}\s*{exponent}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit , exponent = exponent), 
                                         final_text, 
                                         text_to_insert = ' {base_unit_replac}{exponent}'.format(base_unit_replac = base_unit_replaced , exponent = exponent), 
                                         get_char_begin = 1, 
                                         get_char_end = 1,                                      
                                         filter_name = '[0-9]{base_unit}{exponent} -> [0-9] {base_unit_replac}{exponent}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced, exponent = exponent), 
                                         check = False) 
    
        
                #caso a unidade não tenha potência
                elif exponent == '1':                                
                    
                    #substituindo as unidades no denominador (ex: /mm -> mm–1)
                    final_text = filter1(r'(?<!(H|Br|C|Cl|F|O|N|P|S)\s*)[/=]\s*{base_unit}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit), 
                                         final_text, 
                                         text_to_insert = ' {base_unit_replac}–{exponent}'.format(base_unit_replac = base_unit_replaced, exponent = exponent),
                                         get_char_begin = 0, 
                                         get_char_end = 1,                                    
                                         filter_name = '/{base_unit} -> {base_unit_replac}–{exponent}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced, exponent = exponent),
                                         check = False)
                    
                    #quando o número está colado com a unidade (ex: 2m -> 2 m)
                    final_text = filter1(r'[0-9]{base_unit}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit),
                                         final_text, 
                                         text_to_insert = ' {base_unit_replac}'.format(base_unit_replac = base_unit_replaced),
                                         get_char_begin = 1, 
                                         get_char_end = 1, 
                                         filter_name = '[0-9]{base_unit} -> [0-9] {base_unit_replac}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced),
                                         check = False)                    
                    
                    #juntando a unidade com o inverso do expoente (ex: cm 1 -> cm–1 ou m 1 -> m–1)
                    final_text = filter1(r'[\s\(\[\,]{base_unit}\s[\-\‐\–\−\—\―\─\‑\‒\s]*{exponent}(\s|\(|\)|\[|\]|\,|\:|;|\.(?![0-9]))'.format(base_unit = base_unit , exponent = exponent), 
                                         final_text, 
                                         text_to_insert = '{base_unit_replac}–{exponent}'.format(base_unit_replac = base_unit_replaced , exponent = exponent),
                                         get_char_begin = 1, 
                                         get_char_end = 1,                                 
                                         filter_name = '{base_unit} {exponent} -> {base_unit_replac}–{exponent}'.format(base_unit = base_unit, base_unit_replac = base_unit_replaced, exponent = exponent),
                                         check = False)
        
        pattern_sub_list = [[r'\s{2,20}', ' ', 'filtro de "\s" juntos', False ]]    
        final_text = filter_sub(pattern_sub_list, final_text)
        
        counter -= 1

    #fração/proporção (porcentagem)
    final_text = filter1(r'(?<=[0-9])\:(?=[0-9])', final_text, text_to_insert = '#', get_char_begin = 0, get_char_end = 0, filter_name = 'filtro de proporção (1:10)', check = False)
    
    #porcentagem
    final_text = filter1(r'(?<=[\s\(\[\,])(weight|w|wt|W)\s*[/=]\s*(weight|w|wt|W)(?=[\s\(\)\[\]\.\,\:;/])', final_text, text_to_insert = 'wtperc', get_char_begin = 0, get_char_end = 0, filter_name = 'filtro de % (w / w)', check = False)
    final_text = filter1(r'(?<=[\s\(\[\,])(weight|w|wt|W)\s*[/=]\s*(vol|v|V)(?=[\s\(\)\[\]\.\,\:;/])', final_text, text_to_insert = 'wtvperc', get_char_begin = 0, get_char_end = 0, filter_name = 'filtro de % (w / v)', check = False) 
    final_text = filter1(r'(?<=[\s\(\[\,])(vol|v|V)\s*[/=]\s*(vol|v|V)(?=[\s\(\)\[\]\.\,\:;/])', final_text, text_to_insert = 'volperc', get_char_begin = 0, get_char_end = 0, filter_name = 'filtro de % (v / v)', check = False) 

    for unit_par in [['(weight|w|wt|W)', '%', 'wtperc'], ['(vol|v|V)', '%', 'volperc']]:
        
        final_text = filter1(r'[0-9]{u1}[\.\-\‐\–\−\—\―\─\‑\‒\s]?{u2}(?=[\s\(\)\[\]\.\,\:;/]+)'.format(u1 = unit_par[0], u2 = unit_par[1]), final_text, text_to_insert = ' {}'.format(unit_par[2]), get_char_begin = 1, get_char_end = 0, filter_name = 'filtro num colado com [wt,vol]%', check = False) 
        final_text = filter1(r'[0-9]{u2}[\.\-\‐\–\−\—\―\─\‑\‒\s]?{u1}(?=[\s\(\)\[\]\.\,\:;/]+)'.format(u1 = unit_par[0], u2 = unit_par[1]), final_text, text_to_insert = ' {}'.format(unit_par[2]), get_char_begin = 1, get_char_end = 0, filter_name = 'filtro num colado com %[wt,vol]', check = False) 
    
        final_text = filter1(r'(?<=[\s\(\[\,]){u1}[\.\-\‐\–\−\—\―\─\‑\‒\s]?{u2}(?=[\s\(\)\[\]\.\,\:;/]+)'.format(u1 = unit_par[0], u2 = unit_par[1]), final_text, text_to_insert = '{}'.format(unit_par[2]), get_char_begin = 0, get_char_end = 0, filter_name = 'filtro de [wt,vol]%', check = False) 
        final_text = filter1(r'(?<=[\s\(\[\,]){u2}[\.\-\‐\–\−\—\―\─\‑\‒\s]?{u1}(?=[\s\(\)\[\]\.\,\:;/]+)'.format(u1 = unit_par[0], u2 = unit_par[1]), final_text, text_to_insert = '{}'.format(unit_par[2]), get_char_begin = 0, get_char_end = 0, filter_name = 'filtro de %[wt,vol]', check = False)
    
    #potencial elétrico
    final_text = filter1(r'[\s\(\[mµun](Volts|Volt|volts|volt|V)\s*[\-\‐\–\−\—\―\─\‑\‒]?1[\s\(\)\[\]\.\,\:;]', final_text, text_to_insert = 'V–1', get_char_begin = 1, get_char_end = 1, filter_name = 'filtro V colado com -1', check = False) 

    #molaridade
    for unit_par in [['',''],['(µ|u)', 'u'], ['m','m']]:
        final_text = filter1(r'(?<=[0-9]\s*){}M(?=[\s\(\)\[\]\.\,\:;/]+)'.format(unit_par[0]), final_text, text_to_insert = ' {}mol L–1'.format(unit_par[1]), get_char_begin = 0, get_char_end = 0, filter_name = 'filtro de M -> mol L–1', check = False)


    #sexto filtro
    #------------------------------
    #eliminado o erro numérico (os caracteres para erro do tipo "±" viram espaço)
    counter = 3
    while counter != 0:
        for n in [2, 3, 4, 5, 6]:
            #com número negativos
            final_text = filter1(r'(?<![\,\.a-zA-Z0-9])[\-\‐\–\−\—\―\─\‑\‒][0-9]{num1}(\.[0-9]+)*(?![e0-9/;\:\.\-\‐\–\−\—\―\─\‑\‒\(\[]).[\-\‐\–\−\—\―\─\‑\‒][0-9]{num2}(\.[0-9]+)*(?=[\s\:\.;])'.format(
                                                                                                                              num1='{'+str(n)+'}', 
                                                                                                                              num2='{1,'+str(n-1)+'}'), 
                                                                                                                              final_text, 
                                                                                                                              text_to_insert = '', 
                                                                                                                              get_char_begin = n + 1, 
                                                                                                                              get_char_end = 0, 
                                                                                                                              filter_name = f'filtro de -200 ± 10 -> 200 (n = {n}) ', 
                                                                                                                              check = False)
            #com números positivos
            final_text = filter1(r'(?<![\,\.a-zA-Z0-9])[0-9]{num1}(\.[0-9]+)*(?![e0-9/;\:\.\-\‐\–\−\—\―\─\‑\‒\(\[]).[0-9]{num2}(\.[0-9]+)*(?=[\s\:\.;])'.format(
                                                                                                                              num1='{'+str(n)+'}', 
                                                                                                                              num2='{1,'+str(n-1)+'}'), 
                                                                                                                              final_text, 
                                                                                                                              text_to_insert = '', 
                                                                                                                              get_char_begin = n, 
                                                                                                                              get_char_end = 0, 
                                                                                                                              filter_name = f'filtro de 200 ± 10 -> 200 (n = {n}) ', 
                                                                                                                              check = False)
        counter -= 1


    #setimo filtro
    #------------------------------    
    #padronizando os números grandes
    counter = 3
    while counter != 0:    
        for n in [1, 2, 3, 4, 5, 6]:
            final_text = filter1(r'(?<![\,\.\-\‐\–\−\—\―\─\‑\‒a-zA-Z0-9])[0-9]{num1}(?![e0-9/;\:\.\-\‐\–\−\—\―\─\‑\‒\(\[]).[0-9]{num2}(?=[0-9])'.format(num1='{'+str(n)+'}', 
                                                                                                                          num2='{2}'), 
                                                                                                                          final_text, 
                                                                                                                          text_to_insert = '', 
                                                                                                                          get_char_begin = n, 
                                                                                                                          get_char_end = 2, 
                                                                                                                          filter_name = f'filtro de vírgula 3[\s,]000[\s,]000 -> 3000000 (n = {n})',
                                                                                                                          check = False)
        counter -= 1


    #oitavo filtro
    #------------------------------    
    #padronizando os decimais
    counter = 3
    while counter != 0:
        for n in [1, 2, 3, 4]:
            final_text = filter1(r'(?<![\,\.a-zA-Z0-9\-\‐\–\−\—\―\─\‑\‒])0(?![0-9/;\:\.\-\‐\–\−\—\―\─\‑\‒\(\[]).[0-9]{num}(?=\s)'.format(num='{'+str(n)+'}'),
                                                                                    final_text, 
                                                                                    text_to_insert = '.', 
                                                                                    get_char_begin = 1, 
                                                                                    get_char_end = n,
                                                                                    filter_name = f'filtro de vírgula 3[\s,]0 -> 3.0 (n = {n})', 
                                                                                    check = False)
        counter -= 1


    #nono filtro
    #------------------------------
    #n mols em moleculas químicas
    
    mol_number_list = ['₂', '₃', '₄', '₅', '₆']
    val_list_to_replace = ['2', '3', '4', '5', '6']
    for i in range(len(mol_number_list)):
        pattern_sub_list = [[r'{}'.format(mol_number_list[i]), '{}'.format(val_list_to_replace[i]), 'filtro de {0} -> {1}'.format(mol_number_list[i], val_list_to_replace[i]), False ]]
        final_text = filter_sub(pattern_sub_list, final_text)


    #decimo filtro
    #------------------------------
    #cations e anions
    
    for cation in ('Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Ti', 'Zr', 'V', 'Nb', 'Cr', 'Mo', 'W', 
                   'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg', 'Al', 'Ga', 'In', 'Sn', 'Pb', 
                   'Bi', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Er', 'Tm', 'Yb', 'Lu', 'CH4', 'NH4'):
        for valence in ('', '2', '3', '4', '5', '6'):
            final_text = filter1(r'[\s\(]{cat}\s*(\+{val}|{val}\+)[\s\)]'.format(cat = cation , val = valence ), 
                                                                           final_text, 
                                                                           text_to_insert = f'{cation}+{valence}',
                                                                           get_char_begin = 1,
                                                                           get_char_end = 1, 
                                                                           filter_name = f' {cation} {valence}+ -> {cation}+{valence}', 
                                                                           check = False) 

    for anion in ('Br', 'Cl', 'CO3', 'F', 'HCO3', 'HPO4', 'H2PO4', 'HSO4', 'I', 'NO3', 'OH', 'PO4', 'SO4'):
        for valence in ('', '2', '3'):
            final_text = filter1(r'[\s\(\[\,]{ani}\s*([\s\-\‐\–\−\—\―\─\‑\‒]+{val}|{val}[\s\-\‐\–\−\—\―\─\‑\‒]+)[\s\(\)\[\]\.\,\:;]'.format(ani = anion , val = valence ), 
                                                                                                                            final_text, 
                                                                                                                            text_to_insert = f'{anion}@{valence}',
                                                                                                                            get_char_begin = 1, 
                                                                                                                            get_char_end = 1, 
                                                                                                                            filter_name = f' {anion} {valence}― -> {anion}@{valence}', 
                                                                                                                            check = False) 


    #decimo primeiro filtro
    #------------------------------
    #relação entre elementos químicos
    for elem1 in ('H', 'Br', 'C', 'Cl', 'F', 'O', 'N', 'P', 'S'):
        for elem2 in ('H', 'Br', 'C', 'Cl', 'F', 'O', 'N', 'P', 'S'):
            final_text = filter1(r'(?<=[\s\(\)\[\]\.\,\:;]){el1}\s*[/=]\s*{el2}(?=[\s\(\)\[\]\.\,\:;])'.format(el1=elem1, 
                                                                                                               el2=elem2), 
                                                                                                        final_text, 
                                                                                                        text_to_insert = '{el1}#{el2}'.format(el1=elem1, el2=elem2), 
                                                                                                        get_char_begin = 0, 
                                                                                                        get_char_end = 0, 
                                                                                                        filter_name = 'filtro {el1}/{el2} -> {el1}#{el2}'.format(el1=elem1,
                                                                                                                                                                 el2=elem2), 
                                                                                                        check = False)
    
    #compostos químicos
    for molecule in ['Li', 'Na', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Ti', 'Zr', 'Nb', 'Cr', 'Mo',
                     'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg', 'Al', 'Ga', 'In', 'Sn', 'Pb', 
                     'Bi', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Er', 'Tm', 'Yb', 'Lu',                       
                     'Al2O3', 'CaCl2', 'CaCO3', 'CaOH2', 'Ca(OH)2', 'CH3CH2OH', 'CH3COOH', 'CH4', 'CO', 'CO2', 'FeOH3', 'Fe(OH)3', 'FeO', 'FeO2', 'Fe2O3', 
                     'Fe3O4', 'H2O', 'HBr', 'HCl', 'HF', 'HI', 'H2CO3', 'H3PO4', 'H2SO4', 'KMnO4', 'KOH', 'LiOH', 'NaOH', 'NH3', 'NH4', 'NH4OH', 'SiO2']:
        serated_unit_chars = ''
        for char in molecule:
            if char in ('(',')'):
                serated_unit_chars += '\{char}\s?'.format(char = char)
            else:
                serated_unit_chars += '{char}\s?'.format(char = char)
        #eliminando o "\s?"
        serated_unit_chars = serated_unit_chars[ : -3 ]

        #substituindo quando os nomes das moléculas estão separados (ex: C 2 C O 3 -> C2CO3)
        final_text = filter1(r'[\s\(\[\,]{serated_unit_chars}[\s\(\)\[\]\.\,\:;]'.format(serated_unit_chars = serated_unit_chars),
                         final_text, 
                         text_to_insert = '{mol}'.format(mol = molecule),
                         get_char_begin = 1,
                         get_char_end = 1,                              
                         filter_name = '{serated_unit_chars} -> {mol}'.format(serated_unit_chars = serated_unit_chars, mol = molecule),
                         check = False)

    
    #decimo segundo filtro
    #------------------------------    
    #filtros variados de substituição
    pattern_sub_list = [
                        #variados
                        [r'℃', 'C', 'filtro de "℃"', False ],
                        [r'(Ω|Ω)', 'OHM', 'filtro de "OHM"', False ],     
                        [r'\(̴', '(', 'filtro de "(̴"', False ],                        
                        #[r'\(', ' ', 'filtro de "("', False ],
                        #[r'\)', ' ', 'filtro de ")"', False ],
                        #[r'\[', ' ', 'filtro de "["', False ],
                        #[r'\]', ' ', 'filtro de "]"', False ],
                        #[r'/', ' ', 'filtro de "/"', False ],
                        [r'\:', ' ', 'filtro de ":"', False ],
                        [r';', ' ', 'filtro de ";"', False ],
                        #[r'\,', ' ', 'filtro de ","', False ],
                        #[r'\!', ' ', 'filtro de "!"', False ],
                        #[r'\?', ' ', 'filtro de "?"', False ],
                        [r'\s{2,20}', ' ', 'filtro de "\s" juntos', False ]
                        ]
    
    final_text = filter_sub(pattern_sub_list, final_text)


    #decimo terceiro filtro
    #------------------------------            
    #montando o padrão para eliminar intervalos núméricos
    
    #padronizando numeros negativos
    final_text = filter1(r'[\-\‐\–\−\—\―\─\‑\‒][0-9]', final_text, text_to_insert = '–', get_char_begin = 0, get_char_end = 1, filter_name = 'filtro de valores negativos', check = False)
    #substituindo sequências numéricas do tipo 4 -3.4 -3 -0.4 -434
    final_text = filter1(r'(?<![a-zA-Z])[0-9][e\s\-\‐\–\−\—\―\─\‑\‒]{1,2}[\-\‐\–\−\—\―\─\‑\‒][0-9]', final_text, text_to_insert = '&–', get_char_begin = 1, get_char_end = 1, filter_name = 'filtro de intervalo numérico 1', check = False)
    #substituindo sequências numéricas do tipo 4 3.4 3 0.4 434
    final_text = filter1(r'(?<![a-zA-Z])[0-9][e\s\-\‐\–\−\—\―\─\‑\‒]{1,2}[0-9]', final_text, text_to_insert = '&', get_char_begin = 1, get_char_end = 1, filter_name = 'filtro de intervalo numérico 2', check = False)
    #eliminado as sequências após o merging
    #final_text = filter1(r'([\s\(\[\,]|[A-Za-z])([\-\‐\–\−\—\―\─\‑\‒]?[0-9.]+&){2,20}[\-\‐\–\−\—\―\─\‑\‒]?[0-9.]+([\s\)\]\.\,\:;]|[A-Za-z])', final_text, text_to_insert = '', get_char_begin = 1, get_char_end = 1, filter_name = 'filtro de remoção de números', check = False) 


    #decimo quarto filtro
    #------------------------------
    #filtro para eliminar caracteres separados
    pattern_sub_list = [
                        [r'(\s+[a-zA-Z{varied_chars}]{num1}){num2}\s+'.format(varied_chars = varied_chars, num1='{1,3}', num2='{8,200}'), ' ', 'filtro de caracteres separados', False ]
                        ]
    final_text = filter_sub(pattern_sub_list, final_text)
    
    #time.sleep(100)
    
    return final_text, len(final_text)


#------------------------------
#definingo uma função para substituição de match do REGEX
def filter_sub(patter_sub_list, string_text, use_raw_string = False):
    import regex as re
    from FUNCTIONS import check_regex_subs
    #patter_sub_list é uma lista de items "i", os quais possuem o seguinte formato:        
    #i[0] é pattern do regex
    #i[1] é o string que será usado para a substituição
    #i[2] é o nome do filtro
    #i[3] é o booleno para indicar se a representação "raw string" será usada
    #i[4] é o booleno para indicar se a função check será usada
    
    #print(repr(string_text[ : 100]))
    if (use_raw_string is True): #caso se use a representação raw da string
        final_text = repr(string_text)
    else:
        final_text = string_text #caso se use a representação literal string 
    #fazendo as modificações para cada filtro
    for pattern_sub in patter_sub_list:              
        if (pattern_sub[3] is True):
            print('***', repr(pattern_sub[2]), '***')
            check_regex_subs(pattern_sub[0], pattern_sub[1], final_text, char_range = 40)
        final_text = re.sub(pattern_sub[0], pattern_sub[1], final_text)
    #print(repr(final_text[ : 100]))    
    return final_text


#------------------------------
def filter_tokens(token_list):
    
    tokens_to_remove = ['(', ')', '[', ']', '{', '}', '<', '>', '.', ',', ':', ';', '+', '=', '#', '~', ' ', '/', '\\',
                        '|', '-', '‐', '–', '−', '—', '_', '?' ,'!', '·', '*', '&', '’']
    
    ftokens = []
    #removendo os token listados
    for token in token_list:
        get_token = True
        temp_token = str(token)
        #primeiro filtro
        #------------------------------
        if temp_token in tokens_to_remove:
            get_token = False
            pass
        if (get_token is True):
            ftokens.append(temp_token)
    
    return ftokens


'''
#------------------------------
def get_tokens_from_text(text_str, replaced_text_str, collection_token_list = ['a', 'b', 'c'], filter_stopwords = False):
    
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')
    import spacy
    nlp = spacy.load('en_core_web_sm')
    from collections import OrderedDict

    tokens_to_remove = ['(', ')', '[', ']', '{', '}', '<', '>', '.', ',', ':', ';', '_', '+', '=', '#', '^', '~', ' ', '/', '\\',
                        '|', '-', '‐', '–', '−', '—', '$', '?' ,'!', '·', '*', '&', '’']

    tokens_dic = OrderedDict()
    full_text_tokens = list( nlp(text_str) )
    for idx, token in enumerate(full_text_tokens):
        tokens_dic[idx] = str(token)
    
    repl_tokens_dic = OrderedDict()
    repl_text_tokens = list( nlp(replaced_text_str) )
    for idx, token in enumerate(repl_text_tokens):
        repl_tokens_dic[idx] = str(token)
        
    index_list = []
    for i in range(len(repl_text_tokens)):    
        get_token = True
        if str(repl_text_tokens[i]) not in tokens_to_remove:
            pass
        else:
            get_token = False
        
        if str(repl_text_tokens[i]) in collection_token_list:
            pass
        else:
            get_token = False

        if (filter_stopwords is True):
            if str(repl_text_tokens[i]) not in stopwords_list:
                pass
            else:
                get_token = False

        if (get_token is True):            
            index_list.append(i)
    
    final_tokens = []
    for i in index_list:
        final_tokens.append(tokens_dic[i])

    repl_final_tokens = []
    for i in index_list:
        repl_final_tokens.append(repl_tokens_dic[i])
        
    return final_tokens, repl_final_tokens
'''


#------------------------------
def get_term_list_from_TXT(filepath):

    term_list = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            term_list.append(line[ : -1])
        file.close()
    
    return term_list


#------------------------------
def make_column_text(string_text):
    
    new_text = ''
    for char_index in range(len(string_text)):
        if char_index > 0 and char_index % 100 == 0:
            new_text += '\n'
        new_text += string_text[char_index]
    
    return new_text


#------------------------------
def process_sentences(sent_list, text_break_function, min_tokens_per_sent = 5):
           
    sentence_list = []
    for sent in sent_list:
        #--------------------------------------------------------------            
        #usando o filtro (filtro 2)
        final_sent, get_sent = filter2(sent, min_tokens_per_sent = min_tokens_per_sent, text_break_function = text_break_function)
        if (get_sent is True):
            sentence_list.append(final_sent)             
    
    return sentence_list


#------------------------------
def save_full_text_to_json(text, filename, folder, raw_string = False, diretorio=None):
    import json
    import os
    Dic = {}
    Dic['File_name'] = filename
    if (raw_string is True):
        Dic['Full_text'] = repr(text)
    else:
        Dic['Full_text'] = text
    #caso não haja o diretorio ~/Outputs/folder
    if not os.path.exists(diretorio + '/Outputs/' + folder):
        os.makedirs(diretorio + '/Outputs/' + folder)    
    with open(diretorio + '/Outputs/' + folder + '/' + filename + '.json', 'w') as write_file:
        json_str = json.dumps(Dic, sort_keys=True, indent=2)
        write_file.write(json_str)
        write_file.close()
    print('Salvando o raw full text extraido em ~/Outputs/' + folder + '/' + filename + '.json')
    

#------------------------------
def save_text_to_TXT(text, pdf_file_name, diretorio=None):

    import os

    colunized_text = make_column_text(text)
    
    #print('Salvando o texto extraido...')
    #caso não haja o diretorio ~/TXT_extracted
    if not os.path.exists(diretorio + '/Outputs/TXT_extracted'):
        os.makedirs(diretorio + '/Outputs/TXT_extracted')
    with open(diretorio + '/Outputs/TXT_extracted/' + pdf_file_name + '_extracted.txt', 'w') as pdf_file_write:
        pdf_file_write.write(colunized_text)
    print('Salvando o texto extraido em ~/Outputs/TXT_extracted/' + pdf_file_name + '_extracted.txt')
    del(colunized_text)