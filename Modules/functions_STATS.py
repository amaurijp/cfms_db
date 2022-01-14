#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#------------------------------
def calculate_crossentropy(prob_vector, label_prob_vector):

    import math
    
    #print('prob_func.sum() = ', prob_func.sum())
    #print('base_prob_func.sum() = ', base_prob_func.sum())
    
    cross_entropy = 0
    
    for i in range(len(label_prob_vector)):            
        p_label = label_prob_vector[i]
        p_score = prob_vector[i]
        if p_label == 0 or p_score == 0:
            continue        
        val = - ( p_label * math.log(p_score))
        cross_entropy += val
    
    return cross_entropy


#------------------------------
def calculate_prob_dist_RSS(prob_vector, label_prob_vector):
    
    return ( ( prob_vector.cumsum() - label_prob_vector.cumsum() )**2 ).sum()