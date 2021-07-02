import os

labels = ['sport', 'news']

sentences = ['انسان باید',
             'چهار',
             'گروه',
             'کمک',
             'همچنین کاربر',
             'پسز',
             'ملاقات',
             'ژانر',
             ]

for label in labels:
    for sent in sentences:
        os.system("python predict.py --char {0} --input {1}".format(label, sent))
