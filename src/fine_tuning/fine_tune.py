import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from berm_lm import *
from dataset import *


def fine_tune(class_name, json_file, tokenizer, epochs, batch_size, save_url=None, mlm_prob=0.25, use_gpu=True):
    dataset = MaskedLMDataset(json_file, tokenizer)
    data_collector = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collector)
    model = BertLM(class_name)
    # using CPU
    if use_gpu:
        trainer = pl.Trainer(max_epochs=epochs, checkpoint_callback=False, logger=False, gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=epochs, checkpoint_callback=False, logger=False)
    trainer.fit(model, train_loader)
    if save_url is not None:
        torch.save(model.state_dict(), save_url)
