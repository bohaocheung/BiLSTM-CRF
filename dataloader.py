from config import config

def get_words_and_tags_in_one_file(file_path, word_to_ix, tag_to_ix):
    with open(file_path, "r") as f:
        for i,line in enumerate(f):
            if len(line.strip()) > 0:
                l = line.strip().split(" ")
                if l[0] not in word_to_ix:
                    word_to_ix[l[0]] = len(word_to_ix)
                if l[1] not in tag_to_ix:
                    tag_to_ix[l[1]] = len(tag_to_ix)

def get_all_words_and_tags():
    word_to_ix = {}
    tag_to_ix = {}
    get_words_and_tags_in_one_file(config["train_file"], word_to_ix, tag_to_ix)
    get_words_and_tags_in_one_file(config["dev_file"], word_to_ix, tag_to_ix)
    get_words_and_tags_in_one_file(config["test_file"], word_to_ix, tag_to_ix)
    tag_to_ix[config["start_tag"]] = len(tag_to_ix)
    tag_to_ix[config["stop_tag"]] = len(tag_to_ix)
    return word_to_ix, tag_to_ix

def load_data(file_path):
    data = []
    words = []
    tags = []
    with open(file_path, "r") as f:    
        for line in f:
            if len(line.strip()) > 0:
                l = line.strip().split(" ")
                words.append(l[0])
                tags.append(l[1])
            else:
                if len(words) > 0:
                    data.append((words, tags))
                words = []
                tags = []
    return data