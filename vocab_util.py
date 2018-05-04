from pathlib import Path
import numpy as np
import nltk
import re
from parse_json import tokenize
from tqdm import tqdm
import constants
import json

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    file = Path(vocabulary_path)

    if not file.exists():
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="r") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 10000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenize(
                        nltk.word_tokenize(line))  # tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = constants.START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with open(vocabulary_path, "w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


# Returns:
# vocab: dictionary of form <token>, <token_id>
# rev_vocab: list of all words in vocabulary without the new line character
def initialize_vocabulary(vocabulary_path):
    file = Path(vocabulary_path)

    if file.exists():
        rev_vocab = []

        with open(vocabulary_path, "r") as f:
            rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip('\n') for line in rev_vocab]

        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


# Create a subset of the glove file having only the tokens in our vocabulary
def process_glove(glove_file_path, vocab_list, save_path, size=4e5, random_init=True, glove_dim=300):
    file = Path(save_path)

    if not file.exists():

        if random_init:
            glove = np.random.randn(len(vocab_list), glove_dim)
        else:
            glove = np.zeros((len(vocab_list), glove_dim))

        found = 0

        #Fix the padding to zero
        glove[constants.PAD_ID, :] = np.zeros((1, glove_dim))

        with open(glove_file_path, 'r') as fh:

            for line in tqdm(fh, total=size):

                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))

                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_file_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, constants.UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    target_file = Path(target_path)
    data_file = Path(data_path)

    if not target_file.exists():

        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)

        with open(data_path, "r") as data_file:

            with open(target_path, "w") as tokens_file:

                counter = 0

                for line in data_file:

                    counter += 1

                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)

                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def word2charix(word, char2ix, norm_word_length=16):
    """
    Converting a word to a list of indices representing its character
    We truncate/pad the word to be of size 'norm_word_length'
    We assume each word directly fed as a string
    """
    # splitting into list of chars
    word = [char for char in word]

    # padding / truncating each word to word_length
    if len(word) > norm_word_length:
        word = word[:norm_word_length]
    elif len(word) < norm_word_length:
        word = word + (norm_word_length - len(word)) * ['<pad>']

    # converting characters to int in word list
    tmp = []
    for i in range(len(word)):
        
        if word[i] in char2ix:
            char = word[i]
        else:
            char = '<unk>'
        tmp.append(int(char2ix[char]))
    word = tmp

    return word


def create_vocab2charix_dict(vocab_file, vocab2charix_file, char2ix):
    vocab2charix = {}

    with open(vocab_file) as f:
        for line in f:
            line = line.strip()
            if line in ['<pad>', '<sos>', '<unk>']:
                continue

            vocab2charix[line] = word2charix(line, char2ix)

    with open(vocab2charix_file, 'w') as f:
        json.dump(vocab2charix, f)