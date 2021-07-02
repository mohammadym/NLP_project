import os
labels = ['sport', 'news']

for label in labels:
    print(label)
    os.system("python train.py --char {0}".format(label))
