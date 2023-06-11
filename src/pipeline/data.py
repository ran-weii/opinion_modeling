import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from gensim.models import FastText

from collections import Counter, OrderedDict
from torchtext.vocab import vocab as build_vocab
from torchtext.data.utils import get_tokenizer

class TwitterData(Dataset):
    def __init__(self, df, pipeline, max_vocab, device=torch.device("cpu")):
        super().__init__()
        self.df = df
        self.pipeline = pipeline
        self.max_vocab = max_vocab
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["lemmas"]
        
        sentiment = self.df.iloc[idx][
            ["negative", "neutral", "positive"]
        ].values.astype(np.float32)

        processed_text = torch.Tensor(self.pipeline(text)).long()
        sentiment = torch.from_numpy(sentiment)
        sentiment = 0.5 * (sentiment.dot(torch.Tensor([-1, 0, 1])) + 1)

        bow = torch.zeros(self.max_vocab)
        bow[processed_text] = 1
        bow = bow

        return (
            bow.to(self.device), 
            processed_text.to(self.device), 
            sentiment.to(self.device)
        )


def pad_collate(batch):
    """ Pad sequence with zeros 

    returns:
        bow: bag of words vectors
        x_pad: padded token indices
        sentiment: sentiment values
        adj: fully connected adjacency matrices
    """
    # unpack data type
    bow = torch.stack([b[0] for b in batch])
    x = [b[1] for b in batch]
    sentiment = torch.stack([b[2] for b in batch])
    
    x_pad = pad_sequence(x, batch_first=True, padding_value=0).long()
    
    # make mask
    mask = [torch.ones(len(b[1])) for b in batch]
    mask = pad_sequence(mask, batch_first=True, padding_value=0).to(bow.device)
    adj = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    return bow, x_pad, sentiment, adj

def build_torchtext_vocab(docs, min_df=1, max_vocab=10000):
    """ 
    args:
        docs: list or df of tweet strings
        min_df: min doc frequency, default=1
        max_vocab_size: max vocab to include, default=10000

    returns:
        text_pipeline: tokenizing pipeline for dataset object
        vocab: torchtext vocab object
        ordered_dict: vocab dict and freq
    """
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()
    for line in docs:
        counter.update(tokenizer(line))
    counter = counter.most_common(max_vocab)

    sorted_by_freq_tuples = sorted(counter, key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab = build_vocab(ordered_dict, min_freq=min_df)

    # add <unk> token and default index
    if "<unk>" not in vocab: vocab.insert_token("<unk>", 0)
    vocab.set_default_index(-1)
    vocab.set_default_index(vocab["<unk>"])
    
    # preprocess pipeline
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

    # make vocab dict
    df_vocab = pd.DataFrame(
        [[k, v, (ordered_dict[k] if k != "<unk>" else 0)] 
        for k, v in vocab.get_stoi().items()], 
        columns=["word", "index", "freq"]
    ).sort_values(by="word", ascending=True).reset_index(drop=True)
    return text_pipeline, vocab, df_vocab

def load_fasttext_embedding(vocab, fasttext_model):
    embedding_dim = fasttext_model.vector_size
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, i in vocab.get_stoi().items():
        embedding_vector = fasttext_model.wv[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector.copy()
    embedding_matrix = torch.from_numpy(embedding_matrix).to(torch.float32)
    return embedding_matrix

def train_test_split(
    df, embedding_dict=None, sample_ratio=0.99, train_ratio=0.8, batch_size=128, 
    min_df=3, max_vocab=10000, seed=0, device=torch.device("cpu")
    ):
    """
    args:
        sample_ratio: subsample dataset, default=0.99
        train_ratio: percent of train data, default=0.8
        batch_size: default=128
        max_df: max doc frequency, default=0.95
        min_df: min doc count, default=3,
        max_vocab: most frequent words, default=10000
    """
    # shuffle data
    df = df.sample(frac=1, random_state=seed)
    
    # subsample dataset
    num_samples = np.ceil(len(df) * sample_ratio).astype(int)
    df = df.iloc[:num_samples]
    
    # train-test split
    num_train = np.ceil(len(df) * train_ratio).astype(int)
    df_train = df.iloc[:num_train]
    df_test = df.iloc[num_train:]

    print(f"\ntrain size: {df_train.shape}, test_size: {df_test.shape}")
    
    pipeline, vocab, df_vocab = build_torchtext_vocab(
        df_train["lemmas"], min_df, max_vocab
    )
    
    # load embedding
    if embedding_dict is None:
        embedding_matrix = torch.empty(0)
        missing_words = 0
    else:
        assert isinstance(embedding_dict, FastText)
        print("load fasttext emebedding")
        embedding_matrix = load_fasttext_embedding(vocab, embedding_dict)
        missing_words = sum(embedding_matrix.abs().sum(1) == 0)

    # make loaders
    train_loader = DataLoader(
        TwitterData(df_train, pipeline, len(vocab), device=device),
        batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=pad_collate
    )
    test_loader = DataLoader(
        TwitterData(df_test, pipeline, len(vocab), device=device),
        batch_size=batch_size, shuffle=False, drop_last=True,
        collate_fn=pad_collate
    )

    print("dictionary size: %d" % len(df_vocab["index"].unique()))
    print("pretrained embedding matrix size: {}, missing words: {}".format(
        embedding_matrix.shape, missing_words
    ))
    return train_loader, test_loader, df_vocab, embedding_matrix