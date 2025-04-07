import torch
import time

from config import config

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=torch.device("cuda:0"))

def train(model, optimizer, training_data, word_to_ix, tag_to_ix):
# Make sure prepare_sequence from earlier in the LSTM section is loaded
    min_loss = None
    for epoch in range(
            config["epoch"]):  # again, normally you would NOT do 300 epochs, it is toy data
        start_time= time.time()
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long, device=torch.device("cuda:0"))

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            
            optimizer.step()
        end_time = time.time()
        print(f"epoch: {epoch}; loss: {loss}; {end_time - start_time}")
        torch.save(model.state_dict(), f"{config["ckpt_folder"]}/result_{epoch}.ckpt")
        if min_loss == None or loss <= min_loss:
            min_loss = loss
            torch.save(model.state_dict(), config["best_ckpt_file"])

def test(model, test_data, word_to_ix):
    predicts = []
    with torch.no_grad():
        model.load_state_dict(torch.load(config["best_ckpt_file"]))
        for sentence, _ in test_data:
            sentence_in = prepare_sequence(sentence, word_to_ix)
            predict = model(sentence_in)
            predicts.append(predict[1])
    return predicts