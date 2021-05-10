import json
import csv
import re
import nltk
from nltk import word_tokenize as nltk_word_tokenize
import string
import pandas as pd
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
import itertools



def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)




def clean_sentence(sentence):
    fa_punctuations = ['،','«','»',':','؛','ْ','ٌ','ٍ','ُ','ِ','َ','ّ','ٓ','ٰ','-','*']
    clean_sentences = sentence
    re_pattern1 = "/(\s)*[0-9]+"
    re_pattern2 = "\\u200c|\\u200d|\\u200e|\\u200b|\\u2067|\\u2069"

    x = re.search(re_pattern1, clean_sentences)
    if (x):
      clean_sentences = re.sub(re_pattern1, "", clean_sentences)
        
    x = re.search(re_pattern2, clean_sentences)
    if (x):
      clean_sentences = re.sub(re_pattern2, "", clean_sentences)   
        
    punc_regex = re.compile('|'.join(map(re.escape, list(string.punctuation) + list(fa_punctuations))))

    clean_sentences = punc_regex.sub("", clean_sentences)
    clean_sentences = deEmojify(clean_sentences)

    return clean_sentences





f = open('News_channel_messages.json',)

data = json.load(f)

f.close()

list_of_sentences = []

for item in data:
    if "message" in item.keys():
        list_of_sentences.append(item['message'])

clean_sentences = []

for temp in list_of_sentences:
    for line in temp.splitlines():
        clean_sentences.append(clean_sentence(line))

print("total Number of cleaned news sentences:")
print(len(clean_sentences))

address = "/home/mohammad/Dars/NLP/NLP_project/data/clean_news_senteces.txt"

final_sentences = open(address, 'w')

for item in clean_sentences:
    final_sentences.write(item)
    final_sentences.write('\n')

final_sentences.close()

list_of_words = []

for sentence in list_of_sentences:
    splited_words = sentence.split()
    for word in splited_words:
        list_of_words.append(word)

print("total Number of news words:")
print(len(list_of_words))

address = "/home/mohammad/Dars/NLP/NLP_project/stop_words.txt"

stop_words = list(open(address))

for item in list_of_words:
    if item in stop_words:
        list_of_words.remove(item)

cleaned_news_words = []
extra_words = ['.', ',', ':', ';', '\@', '/', '@', '#', '!', '-', '_', '+', '؟', '?', '<', '>', '*', '،', 'و' ,'به', 'را', 'در', 'از',
                 'بر', 'اگر', 'با', 'تا', 'نیز', 'که', '.', ')', '(', 'ها', 'ای', 'رو', 'هم', 'بین', 'برای', 'نه', 'بلکه',
                  'بااینکه', 'چنانکه', 'اگرچه', 'آن', 'این', 'آنها', 'طرف', 'او', 'پس', 'ولی', 'زیرا', 'چون‌که', 'چندان‌که', 'همین‌که',
                   'خواه', 'نیز', 'باری', 'اینکه', 'ایشان', 'شما', 'چون', 'ولی', 'چه', 'زیرا', 'هم', 'اگرچه', 'بااین‌حال', 'باوجوداین',
                    'چنانچه', 'این', 'ما', 'او', 'طرف', 'من', 'تو', 'ایشان']

for word in list_of_words:
    if word in extra_words:
        continue
    if word is not None and word[0] == '@':
        continue 
    if word is not None and word[0] in extra_words:
        word.replace(word[0], '')
    if word[len(word)-1] in extra_words:
        word.replace(word[len(word)-1], '')
    if word is not None and word.isdecimal():
        continue
    if word is not None:
        word = ''.join([i for i in word if not i.isdigit()])
    word = deEmojify(word)
    cleaned_news_words.append(word)

print("number of news words after cleaning:")
print(len(cleaned_news_words))


address = "/home/mohammad/Dars/NLP/NLP_project/data/inital_news_data.txt"

inital_data = open(address, 'w')

for item in list_of_sentences:
    inital_data.write(item)
    inital_data.write('\n')

inital_data.close()

address = "/home/mohammad/Dars/NLP/NLP_project/data/news_words.txt"

words = open(address, 'w')

for item in list_of_words:
    words.write(item)
    words.write('\n')

words.close()


address = "/home/mohammad/Dars/NLP/NLP_project/data/cleaned_news_data.txt"

cleaned_data = open(address, 'w')

for item in cleaned_news_words:
    cleaned_data.write(item)
    cleaned_data.write('\n')

cleaned_data.close()

distinct_news_words = []

for word in cleaned_news_words:
    if word not in distinct_news_words:
        distinct_news_words.append(word)

print("number of distinct news words:")
print(len(distinct_news_words))

print("number of undistinct news words:")
print(len(cleaned_news_words)-len(distinct_news_words))









f = open('Sport_channel_messages.json',)

data = json.load(f)

f.close()

list_of_sentences = []

for item in data:
    if "message" in item.keys():
        list_of_sentences.append(item['message'])

clean_sentences = []

for temp in list_of_sentences:
    for line in temp.splitlines():
        clean_sentences.append(clean_sentence(line))

print("total Number of cleaned sport sentences:")
print(len(clean_sentences))

address = "/home/mohammad/Dars/NLP/NLP_project/data/clean_sport_senteces.txt"

final_sentences = open(address, 'w')

for item in clean_sentences:
    final_sentences.write(item)
    final_sentences.write('\n')

final_sentences.close()

list_of_words = []

for sentence in list_of_sentences:
    splited_words = sentence.split()
    for word in splited_words:
        list_of_words.append(word)

print("total number of sport words:")
print(len(list_of_words))

address = "/home/mohammad/Dars/NLP/NLP_project/stop_words.txt"

stop_words = list(open(address))

for item in list_of_words:
    if item in stop_words:
        list_of_words.remove(item)

cleaned_sport_words = []

for word in list_of_words:
    if word in extra_words:
        continue
    if word is not None and word[0] == '@':
        continue 
    if word is not None and word[0] in extra_words:
        word.replace(word[0], '')
    if word[len(word)-1] in extra_words:
        word.replace(word[len(word)-1], '')
    if word is not None and word.isdecimal():
        continue
    if word is not None:
        word = ''.join([i for i in word if not i.isdigit()])
    word = deEmojify(word)
    cleaned_sport_words.append(word)

print("number of sport words after cleaning:")
print(len(cleaned_sport_words))


address = "/home/mohammad/Dars/NLP/NLP_project/data/inital_sport_data.txt"

inital_data = open(address, 'w')

for item in list_of_sentences:
    inital_data.write(item)
    inital_data.write('\n')

inital_data.close()

address = "/home/mohammad/Dars/NLP/NLP_project/data/sport_words.txt"

words = open(address, 'w')

for item in list_of_words:
    words.write(item)
    words.write('\n')

words.close()


address = "/home/mohammad/Dars/NLP/NLP_project/data/cleaned_sport_data.txt"

cleaned_data = open(address, 'w')

for item in cleaned_sport_words:
    cleaned_data.write(item)
    cleaned_data.write('\n')

cleaned_data.close()


distinct_sport_words = []

for word in cleaned_sport_words:
    if word not in distinct_sport_words:
        distinct_sport_words.append(word)

print("number of distinct sport words:")
print(len(distinct_sport_words))

print("number of undistinct sport words:")
print(len(cleaned_sport_words)-len(distinct_sport_words))









##############################################################

#analysis

list1_as_set = set(distinct_sport_words)
intersection = list1_as_set.intersection(distinct_news_words)

distnict_common_words = list(intersection)

print("number of distinct common words:")
print(len(distnict_common_words))

distinct_uncommon_words = []

for word in distinct_news_words:
    if word not in distinct_uncommon_words:
        distinct_uncommon_words.append(word)

for word in distinct_sport_words:
    if word not in distinct_uncommon_words:
        distinct_uncommon_words.append(word)


print("number of distinct uncommon words:")
print(len(distinct_uncommon_words))




most_repeated_news_words = defaultdict(int)

for word in cleaned_news_words:
    if word not in distnict_common_words:
        most_repeated_news_words[word] += 1

most_repeated_sport_words = defaultdict(int)

for word in cleaned_sport_words:
    if word not in distnict_common_words:
        most_repeated_sport_words[word] += 1


sport_res = max(most_repeated_sport_words, key=temp.get)

news_res = max(most_repeated_news_words, key=temp.get)

print(most_repeated_news_words)

print(most_repeated_sport_words)
