#!/bin/bash
python -m src.word2vec.run $1
python -m src.tokenization.vocab $2
python -m src.parsing.run $3
python -m src.language_model.main $4
python -m src.language_model.generate $5
python -m src.fine_tuning.generate_text $6