# NLP_project

This is the dataset of final project of natrual language processing course.
In this project we want to train the network based on two types of data.
to classess of data is about sport and news messages in telegram client. we get api of telegram by telethon library in python and save this data in json files.
then save message field in a txt file and then make some analysis on it.
In project we get a message to system and system determines this message is belongs to which class.



to get data of telegram:
1-Go to src file to access to codes.
2-Run 'python3 SportChannelMessages.py' and 'python3 NewsChannelMessages.py' command to create client and get your phone id and connect your profile and then get data from chosen channels and save json data in Sport_channel_messages.json and News_channel_messages.json
3-in the end run 'python3 main.py' to create .txt files in data directory
