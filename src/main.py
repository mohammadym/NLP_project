import json
import csv
import re

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


f = open('News_channel_messages.json',)

data = json.load(f)

f.close()

list_of_sentences = []

for item in data:
    if "message" in item.keys():
        list_of_sentences.append(item['message'])

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

cleaned_words = []
extra_words = ['.', ',', ':', ';', '\@', '/', '@', '#', '!', '-', '_', '+', '؟', '?', '<', '>', '*', '،', 'و' ,'به', 'را', 'در', 'از',
                 'بر', 'اگر', 'با', 'تا', 'نیز', 'که', '.', ')', '(', 'ها', 'ای', 'رو', 'هم', 'بین', 'برای']

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
    cleaned_words.append(word)

print("Number of news words after cleaning:")
print(len(cleaned_words))


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

for item in cleaned_words:
    cleaned_data.write(item)
    cleaned_data.write('\n')

cleaned_data.close()

distinct_news_words = []

for word in cleaned_words:
    if word not in distinct_news_words:
        distinct_news_words.append(word)

print("number of distinct news words:")
print(len(distinct_news_words))

print("number of undistinct news words:")
print(len(cleaned_words)-len(distinct_news_words))









f = open('Sport_channel_messages.json',)

data = json.load(f)

f.close()

list_of_sentences = []

for item in data:
    if "message" in item.keys():
        list_of_sentences.append(item['message'])

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

cleaned_words = []
extra_words = ['.', ',', ':', ';', '\@', '/', '@', '#', '!', '-', '_', '+', '؟', '?', '<', '>', '*', '،', 'و' ,'به', 'را', 'در', 'از',
                 'بر', 'اگر', 'با', 'تا', 'نیز', 'که', '.', ')', '(', 'ها', 'ای', 'رو', 'هم', 'بین', 'برای']

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
    cleaned_words.append(word)

print("Number of sport words after cleaning:")
print(len(cleaned_words))


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

for item in cleaned_words:
    cleaned_data.write(item)
    cleaned_data.write('\n')

cleaned_data.close()


distinct_sport_words = []

for word in cleaned_words:
    if word not in distinct_sport_words:
        distinct_sport_words.append(word)

print("number of distinct sport words:")
print(len(distinct_sport_words))

print("number of undistinct sport words:")
print(len(cleaned_words)-len(distinct_sport_words))