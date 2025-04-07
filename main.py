import torch
import torch.optim as optim
import time

from config import config, get_embedding_dim
from model import BiLSTM_CRF
from dataloader import load_data, get_all_words_and_tags
from evaluator import get_precision_recall_f1
from trainer import train, test

def main(is_training):
    # Make up some training data
    training_data = load_data(config["train_file"])
    test_data = load_data(config["test_file"])

    # get word_to_idx, tag_to_idx
    word_to_ix, tag_to_ix = get_all_words_and_tags()

    # calculate embedding_dim & hidden_dim
    config["embedding_dim"] = get_embedding_dim(word_to_ix)
    config["hidden_dim"] = config["embedding_dim"]

    # create model & optimizer
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, config["embedding_dim"], config["hidden_dim"]).to("cuda:0")
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # training
    if is_training:
        train(model, optimizer, training_data, word_to_ix, tag_to_ix)
    # testing
    predicts = test(model, test_data, word_to_ix)
    # evaluating
    precision_scores, recall_scores, f1_scores = get_precision_recall_f1(predicts, test_data, tag_to_ix)
    # for show
    print("precision   recall   f1_score")
    for tag in tag_to_ix:
        print(f"{precision_scores[tag]}   {recall_scores[tag]}   {f1_scores[tag]}  {tag}")

if __name__ == "__main__":
    main(config["is_training"])