import os
import datetime


def log(message, task):
    url = None
    if task == "word2vec":
        url = "../../logs/word2vec.log"
    if task == "language_model":
        url = "../../logs/language_model.log"
    if task == "fine_tuning":
        url = "../../logs/fine_tuning.log"
    if url is None:
        return
    if not os.path.exists(url):
        os.makedirs('logs', exist_ok=True)
        with open(url, 'w') as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")
    else:
        with open(url, 'a') as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")
