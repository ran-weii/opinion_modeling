from datetime import datetime
import hashlib
import preprocessor as p
import spacy

def remove_newline(text):
    return text.replace("\n", " \n")

def match_to_list(match):
    """ Convert preprocessor match to list of strings """
    out = []
    if match is not None:
        for i in range(len(match)):
            out.append(match[i].match)
    return out

def process_json_item(item):
    """ Process raw json item """
    text = item[0]
    receiver = item[1]
    author = item[2]
    date = datetime.strptime(item[3], "%Y-%m-%d %H:%M:%S")
    depth = item[4].replace("depth = ", "")
    parent_id = item[5]

    text = remove_newline(text)
    parsed_text = p.parse(text)

    emojis = ", ".join(match_to_list(parsed_text.emojis))
    hashtags = ", ".join(match_to_list(parsed_text.hashtags))
    mentions = ", ".join(match_to_list(parsed_text.mentions))
    numbers = ", ".join(match_to_list(parsed_text.numbers))
    reserved = ", ".join(match_to_list(parsed_text.reserved))
    smileys = ", ".join(match_to_list(parsed_text.smileys))
    urls = ", ".join(match_to_list(parsed_text.urls))

    tokens = p.tokenize(text)
    out = {
        "parent_id": parent_id,
        "text": text,
        "tokens": tokens,
        "author": author,
        "receiver": receiver,
        "date": date,
        "depth": depth,
        "emojis": emojis,
        "hashtags": hashtags,
        "mentions": mentions,
        "numbers": numbers,
        "reserved": reserved,
        "smileys": smileys,
        "urls": urls
    }
    return out

def remove_tags(df):
    """ remove tweet-processor lib tags """
    # python does not allow $XXX$ symbols
    df = df.str.replace("$", "")
    df = df.str.replace("URL", "")
    df = df.str.replace("HASHTAG", "")
    df = df.str.replace("MENTION", "")
    df = df.str.replace("RESERVED", "")
    df = df.str.replace("EMOJI", "")
    df = df.str.replace("SMILEY", "")
    df = df.str.replace("NUMBER", "")
    return df

def hash_md5(text):
    return hashlib.md5(text.encode()).digest()

def lemmatize(
    doc, stop_words=None, pos=["PROPN", "NOUN", "VERB"], 
    keywords=[], min_len=3
    ):
    """ 
    args:
        doc: spacy doc object
        pos (list): allowed part of speech, default=["PROPN", "NOUN", "VERB"]
        keywords (list): included keywords anyway, default=[]
        min_len (int): min number of letters in a word, default=3
    """
    tokens = [
        token.lemma_.lower() for token in doc \
        if ((token.pos_ in pos and len(token.lemma_) >= min_len) \
            or token.lemma_.lower() in keywords) \
            and (token.lemma_.lower() not in stop_words)
    ]
    return " ".join(tokens)

def preprocess(
    df, stop_words=None, pos=["PROPN", "NOUN", "VERB"], 
    keywords=[], min_len=3, min_words=5
    ):
    text = remove_tags(df["tokens"].astype(str))
    text = text.str.replace("[^\w\s]", " ") # strip punctuations

    nlp = spacy.load(
        "en_core_web_sm", 
        disable=["parser", "ner", "ent"], 
    )
    docs = list(nlp.pipe(text.tolist()))
    lemmas = [
        lemmatize(doc, stop_words, pos, keywords, min_len) for doc in docs
    ]

    df = df.assign(lemmas=lemmas)
    df = df.assign(num_lemmas=df["lemmas"].str.split().apply(len))

    # drop short text
    df = df.loc[df["num_lemmas"] >= min_words]

    # filter md5
    df = df.assign(md5=df["lemmas"].apply(hash_md5))
    df = df.drop_duplicates(subset="md5", keep="first")
    df = df.reset_index(drop=True)
    return df