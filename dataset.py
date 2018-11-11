import sqlite3 as lite
import spacy
import string
from tqdm import tqdm
import pickle, os, re
from typing import List, Tuple, Any, Iterable
from torchtext import data as D
from torchtext import vocab as V
from itertools import cycle
import torch

nlp = spacy.load('en_core_web_sm')

def clean_str(content : str) -> str:
    content = re.sub(r"\'s ", " ", content)
    content = re.sub(r"\'m ", " ", content)
    content = re.sub(r"\'ve ", " ", content)
    content = re.sub(r"n\'t ", " not ", content)
    content = re.sub(r"\'re ", " ", content)
    content = re.sub(r"\'d ", " ", content)
    content = re.sub(r"\'ll ", " ", content)
    content = re.sub("-", " ", content)
    content = re.sub(r"@", " ", content)
    content = re.sub('\'', '', content)
    content = content.translate(content, dict((ord(char), u' ') for char in string.punctuation))
    content = content.replace("..", "").strip()
    return content

def tokenize_pitchfork_data(data_path : str = './data/pitchfork.sqlite',
                            tokenized_data_path : str = './data/pitchfork.pkl') -> List[Tuple[List[str], str, float]]:
    
    if os.path.exists(tokenized_data_path):
        print("Loading tokenized data...")
        with open(tokenized_data_path, "rb") as handle:
            return pickle.load(handle)
        
    connect = lite.connect(data_path)
    with connect:
        cur = connect.cursor()
    cur.execute("SELECT c.reviewid, c.content, g.genre, r.score  \
                FROM content c \
                  JOIN genres g \
                    ON g.reviewid = c.reviewid \
                  JOIN reviews r \
                    ON r.reviewid = c.reviewid")
    result = cur.fetchall()
    tokenized_review = []
    for review in tqdm(result, desc="Tokenize data"):
        id_, content, genre, score  = review
        genre = str(genre)
        doc = nlp(content)
        for sentence in doc.sents:
#            sentence = nlp(clean_str(str(sentence)))
            if len(sentence) < 2:
                continue
            tokenized_review.append(([str(word) for word in sentence], genre, score))
           
    print("Saving tokenized data...")
    with open(tokenized_data_path, "wb") as handle:
        pickle.dump(tokenized_review, handle)
        
    return tokenized_review

class pitchforkDataset(D.Dataset):
    def __init__(self, data : Iterable[Tuple[List[str], str, float]], 
                 text_field : D.Field, label_field : D.Field, score_field : D.Field,
                 **kwargs):
        fields = [('text', text_field), ('label', label_field), ('score', score_field)]
        examples = []
        for item in data:
            examples.append(D.Example.fromlist(item, fields))
        super(pitchforkDataset, self).__init__(examples, fields, **kwargs)

def make_dataset(data : Iterable[Tuple[List[str], str, float]],
                 opt) -> Tuple[D.Dataset, D.Dataset, V.Vocab]:
    text_field = D.Field(init_token="<sos>", eos_token="<eos>", batch_first=True)
    text_field.build_vocab(map(lambda x: x[0], data), max_size=opt.max_words, vectors=opt.embedding)
    label_field = D.Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None)
    label_field.build_vocab(map(lambda x : x[1], data))
    score_field = D.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None, dtype=torch.float32)
    if opt.dataset == 'pitchfork':
        dataset = pitchforkDataset(data, text_field, label_field, score_field)
    elif opt.dataset == 'imdb':
        raise NotImplementedError
    else:
        raise AttributeError
    train_dataset, val_dataset = dataset.split(split_ratio=0.9)
    return (train_dataset, val_dataset, text_field.vocab)

def make_iterator(data: Tuple[D.Dataset, D.Dataset], opt):
    return D.BucketIterator.splits(data, batch_sizes=[opt.batch_size] * len(data),
                                   sort_key = lambda x :len(x.text),
                                   repeat = False)

def make_loader(data_iter, opt):
    for data in cycle(data_iter):
        if len(data) != opt.batch_size:
            continue
        if opt.dataset == 'pitchfork':
            label_vec = torch.zeros(opt.batch_size, opt.label_cats)
            label_vec = label_vec.scatter(1, data.label.unsqueeze(1), 1)
            label = torch.cat((label_vec, data.score.unsqueeze(1)), dim=1)
            yield data.text, label
        elif opt.dataset == 'imdb':
            raise NotImplementedError
        raise AttributeError