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
from parsivar import Normalizer
from parsivar import Tokenizer
from collections import Counter
import matplotlib.pyplot as plt



################################################

#inital_values


extra_elements = ['.', ',', '@', '#', '/', ')', '(', '!', '?', '؟', '_', '-', '،', ':', ';', '+', '-', '%', '*', '\@', '&', '<', '>', '{', '}', '،']

extra_words = ['.', ',', ':', ';', '\@', '/', '@', '#', '!', '-', '_', '+', '؟', '?', '<', '>', '*', '،', 'و' ,'به', 'را', 'در', 'از',
                 'بر', 'اگر', 'با', 'تا', 'نیز', 'که', '.', ')', '(', 'ها', 'ای', 'رو', 'هم', 'بین', 'برای', 'نه', 'بلکه',
                  'بااینکه', 'چنانکه', 'اگرچه', 'آن', 'این', 'آنها', 'طرف', 'او', 'پس', 'ولی', 'زیرا', 'چون‌که', 'چندان‌که', 'همین‌که',
                   'خواه', 'نیز', 'باری', 'اینکه', 'ایشان', 'شما', 'چون', 'ولی', 'چه', 'زیرا', 'هم', 'اگرچه', 'بااین‌حال', 'باوجوداین',
                    'چنانچه', 'این', 'ما', 'او', 'طرف', 'من', 'تو', 'ایشان', '%', 'یه', ':', '#', '؛', 'آ', '(ع)', '(', 'با', '،', '!',
                    '؟', '،', '»', '«', 'ع', '"']


english_chars = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'E', 'e', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'K', 'k',
 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'p', 'q', 'Q', 's', 'S', 'r', 'R', 't', 'T', 'y', 'Y', 'U', 'u', 'x', 'X', 'z',
  'Z', 'w', 'W', 'v', 'V']

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
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

my_normalizer = Normalizer()

my_tokenizer = Tokenizer()


#########################################

#News_Data





f = open('News_channel_messages.json',)

data = json.load(f)

f.close()

list_of_sentences = []

for item in data:
    if "message" in item.keys():
        list_of_sentences.append(item['message'])

clean_sentences = []

for temp in list_of_sentences:
    item = my_tokenizer.tokenize_sentences(my_normalizer.normalize(temp))
    for line in item:
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

for item in clean_sentences:
    list_words = my_tokenizer.tokenize_words(my_normalizer.normalize(item))
    for temp in list_words:
        list_of_words.append(temp)


print("total Number of news words:")
print(len(list_of_words))

address = "/home/mohammad/Dars/NLP/NLP_project/stop_words.txt"

stop_words = list(open(address))

for item in list_of_words:
    if item in stop_words:
        list_of_words.remove(item)

cleaned_news_words = []

for word in list_of_words:
    if word in extra_words:
        continue
    if word is not None and word[0] == '@':
        continue 
    if word is not None and word[0] in extra_words:
        word.replace(word[0], '')
    if word is not None and word[0] in english_chars:
        continue
    if word[len(word)-1] in extra_words:
        word.replace(word[len(word)-1], '')
    if word is not None and word.isdecimal():
        continue
    if word is not None:
        word = ''.join([i for i in word if not i.isdigit()])
    for chars in word:
        if chars in extra_elements:
            word.replace(chars,'')
    word = deEmojify(word)
    cleaned_news_words.append(word)

for word in cleaned_news_words:
    if word in extra_words:
        cleaned_news_words.remove(word)

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




print('\n')




####################################################

#Soprt_Data




f = open('Sport_channel_messages.json',)

data = json.load(f)

f.close()

list_of_sentences = []

for item in data:
    if "message" in item.keys():
        list_of_sentences.append(item['message'])

clean_sentences = []

for temp in list_of_sentences:
    item = my_tokenizer.tokenize_sentences(my_normalizer.normalize(temp))
    for line in item:
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

for item in clean_sentences:
    list_words = my_tokenizer.tokenize_words(my_normalizer.normalize(item))
    for temp in list_words:
        list_of_words.append(temp)


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
    if word is not None and word[0] in english_chars:
        continue
    if word[len(word)-1] in extra_words:
        word.replace(word[len(word)-1], '')
    if word is not None and word.isdecimal():
        continue
    if word is not None:
        word = ''.join([i for i in word if not i.isdigit()])
    word = deEmojify(word)
    cleaned_sport_words.append(word)

for word in cleaned_sport_words:
    if word in extra_words:
        cleaned_sport_words.remove(word)

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




Counter1 = Counter(cleaned_news_words)

most_news_occur = Counter1.most_common(10)

most_common_news_values = []

address = "/home/mohammad/Dars/NLP/NLP_project/data/most_repeated_news_data.txt"

inital_data = open(address, 'w')

for (key, value) in most_news_occur:
    inital_data.write(key)
    inital_data.write('\n')
    inital_data.write(str(value))
    inital_data.write('\n')
inital_data.close()





Counter2 = Counter(cleaned_sport_words)

most_sport_occur = Counter2.most_common(10)

most_common_sport_values = []

address = "/home/mohammad/Dars/NLP/NLP_project/data/most_repeated_sport_data.txt"

inital_data = open(address, 'w')

for (key, value) in most_sport_occur:
    inital_data.write(key)
    inital_data.write('\n')
    inital_data.write(str(value))
    inital_data.write('\n')
inital_data.close()




list1_as_set = set(cleaned_news_words)
intersection = list1_as_set.intersection(cleaned_sport_words)

common_words = list(intersection)


RelativeNormalizedFrequency = {}

for word in common_words:
    count1 = 0
    count2 = 0 
    for item1 in cleaned_news_words:
        if word == item1:
            count1 += 1
    for item2 in cleaned_sport_words:
        if word == item2:
            count2 += 1
    RelativeNormalizedFrequency[word] = (count1/len(cleaned_news_words))/(count2/len(cleaned_sport_words))


RelativeNormalizedFrequencylist = sorted(RelativeNormalizedFrequency.items(), key=lambda x: x[1], reverse=True)

address = "/home/mohammad/Dars/NLP/NLP_project/data/RelativeNormalizedFrequency_data.txt"

inital_data = open(address, 'w')

for item in RelativeNormalizedFrequencylist:
    inital_data.write(item[0])
    inital_data.write('\n')
    inital_data.write(str(item[1]))
    inital_data.write('\n')
inital_data.close()


TF_news = {}
for (key, value) in most_news_occur:
    TF_news[key] = value/len(most_news_occur)

TF_sport = {}
for (key, value) in most_sport_occur:
    TF_sport[key] = value/len(most_sport_occur)


IDF_news = {}
for (key, value) in most_news_occur:
    if key in cleaned_news_words and key in cleaned_sport_words:
        IDF_news[key] = 1/2
    elif key in cleaned_sport_words and key not in cleaned_sport_words:
        IDF_news[key] = 1
    elif key not in cleaned_sport_words and key in cleaned_sport_words:
        IDF_news[key] = 1
    elif key not in cleaned_sport_words and key not in cleaned_sport_words:
        IDF_news[key] = 0


IDF_sport = {}
for (key, value) in most_sport_occur:
    if key in cleaned_news_words and key in cleaned_sport_words:
        IDF_sport[key] = 1/2
    elif key in cleaned_news_words and key not in cleaned_sport_words:
        IDF_sport[key] = 1
    elif key not in cleaned_news_words and key in cleaned_sport_words:
        IDF_sport[key] = 1
    elif key not in cleaned_news_words and key not in cleaned_sport_words:
        IDF_sport[key] = 0

TF_IDF_news = {}
for (key, value) in most_news_occur:
    TF_IDF_news[key] = TF_news[key]*IDF_news[key]


TF_IDF_sport = {}
for (key, value) in most_sport_occur:
    TF_IDF_sport[key] = TF_sport[key]*IDF_sport[key]

TF_IDF_news_list = sorted(TF_IDF_news.items(), key=lambda x: x[1], reverse=True)

TF_IDF_sport_list = sorted(TF_IDF_sport.items(), key=lambda x: x[1], reverse=True)

address = "/home/mohammad/Dars/NLP/NLP_project/data/TFIDF_news_data.txt"

inital_data = open(address, 'w')

for item in TF_IDF_news_list:
    inital_data.write(item[0])
    inital_data.write('\n')
    inital_data.write(str(item[1]))
    inital_data.write('\n')
inital_data.close()

address = "/home/mohammad/Dars/NLP/NLP_project/data/TFIDF_sport_data.txt"

inital_data = open(address, 'w')

for item in TF_IDF_sport_list:
    inital_data.write(item[0])
    inital_data.write('\n')
    inital_data.write(str(item[1]))
    inital_data.write('\n')
inital_data.close()

Counter3 = Counter(cleaned_news_words)

most_occur = Counter3.most_common(len(cleaned_news_words))

most_common_news_keys = []

temp = 1
for (key, value) in most_occur:
    most_common_news_values.append(value)
    most_common_news_keys.append(temp)
    temp += 1

plt.xlabel('word index')
plt.ylabel('Frequency')
plt.bar(most_common_news_keys, most_common_news_values)
plt.show()





# Counter4 = Counter(cleaned_sport_words)

# most_occur = Counter4.most_common(len(cleaned_sport_words))

# most_common_sport_keys = []

# temp = 1
# for (key, value) in most_occur:
#     most_common_sport_values.append(value)
#     most_common_sport_keys.append(temp)
#     temp += 1

# plt.xlabel('word index')
# plt.ylabel('Frequency')
# plt.bar(most_common_sport_keys, most_common_sport_values)
# plt.show()
