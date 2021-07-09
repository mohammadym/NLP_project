import os
from src.logger.logger import log

labels = ['sport', 'news']

sentences = ['انسان باید',
             'چهار',
             'گروه',
             'کمک',
             'همچنین کاربر',
             'پسز',
             'ملاقات',
             'رقبا',
             'امتیاز',
             'جانشین',
             'صدر',
             'جدول',
             'لباس',
             'هیجان',
             'تنش',
             ]

for label in labels:
    log(f"Generating sentences for {label} label", "language_model")
    for sent in sentences:
        os.system("python predict.py --char {0} --input {1}".format(label, sent))
