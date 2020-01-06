# -*- coding: utf-8 -*-

import re
import numpy as np
import string

def normalize(title):
  # remove punctuation
  replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  title = title.translate(replace_punctuation)
  # remove numbers
  title = re.sub('\d', ' ', title)
  # lowercase all characters
  title = title.lower()
  # remove extra spaces between string
  title = ' '.join(title.split())
  return title


def clean_data(str):
    '''
    This function takes a string, clean it with regular expression
    and return the cleaned string
    '''
    matched_list = re.findall(r'[a-zA-Z\.]+', str)
    return ' '.join(matched_list)


def from_str_to_list(text):
    '''
    This function takes string object (with list formatting) then parsing it 
    and converting it to list 
    '''
    str_cleaned = re.sub('[\'"\[\]]', '', text)
    return [token.strip() for token in str_cleaned.split(',')]



    
