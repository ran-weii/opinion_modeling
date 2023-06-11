import torch

def get_top_words(topic_term, df_vocab, topk=20):
    """ top k words in each topic """
    df_vocab = df_vocab.sort_values(by="index")
    num_topics = len(topic_term)

    top_words = []
    for i in range(num_topics):
        idx = topic_term[i].argsort()[:-topk - 1:-1]
        top_words.append([df_vocab["word"].iloc[i] for i in idx])

    return top_words

def get_topic_coherence(loader, topic_term, df_vocab, topk=20):
    """ mutual info based on topk words 
    
    args:
        loader: torch dataloader
        topic_term: topic-term torch tensor
        df_vocab:
        topk: topk words to use, default=20
    """

    # find the most frequent word index in each topic
    num_topics = len(topic_term)
    topic_term_index = torch.stack(
        [topic_term[i].argsort(descending=True)[:topk] for i in range(num_topics)]
    ).long()
    
    # get full doc-term matrix
    doc_term = torch.cat([d[0].cpu() for d in loader], dim=0) # use bow
    num_docs = doc_term.shape[0]
    
    # word occurrance prob
    p_i = doc_term.sum(0) / num_docs
    
    # calculate mutual info
    TC = torch.zeros(num_topics)
    for i in range(topk):
        for j in range(i+1, topk):
            p_ij = torch.sum(
                (doc_term[:, topic_term_index[:, i]] + \
                doc_term[:, topic_term_index[:, j]]) > 0, 
            dim=0) / num_docs

            num = torch.log(p_ij + 1e-10) - \
                torch.log(p_i[topic_term_index[:, i]] + 1e-10) - \
                torch.log(p_i[topic_term_index[:, j]] + 1e-10)
            dnm = -torch.log(p_ij + 1e-10)
            TC += num / dnm
    
    return TC.mean().data.item()

def get_topic_diversity(topic_term):
    """ number of unique words in topics """
    num_topics = len(topic_term)

    word_list = []
    count = 0
    for i in range(num_topics):
        for word in topic_term[i]:
            word_list.append(word)
            count += 1
    word_list = list(set(word_list))
    TD = len(word_list) / count
    return TD

def get_sentiment_diversity(topic_sentiment):
    d = torch.pow(
        topic_sentiment.unsqueeze(-1) - topic_sentiment.unsqueeze(-2), 2
    )
    return d.mean().data.numpy()