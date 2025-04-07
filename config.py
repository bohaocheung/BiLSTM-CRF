import numpy as np

def get_embedding_dim(word_to_ix):
    return 2 * int(np.floor(np.power(len(word_to_ix), 0.25)))

config = {
    "is_training": True,
    "start_tag": "<START>",
    "stop_tag": "<STOP>",
    "train_file": "data_ner/train.txt",
    "dev_file": "data_ner/dev.txt",
    "test_file": "data_ner/test.txt",
    "ckpt_foler": "ckpt",
    "best_ckpt_file": "ckpts/best.ckpt",
    "epoch": 20
}