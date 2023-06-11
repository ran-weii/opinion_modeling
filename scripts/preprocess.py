import argparse
import os
import time
import pandas as pd

from spacy.lang.en.stop_words import STOP_WORDS
from src.pipeline.preprocessing import preprocess

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--save", type=bool_, default=True)
    parser.add_argument("--seed", type=int, default=0)
    arglist = vars(parser.parse_args())
    return arglist

""" TODO: refractor this """
def main(arglist):
    file_path = os.path.join(
        arglist["data_path"],
        "av_classification",
        "av_tweets.csv"
    )
    df = pd.read_csv(file_path, lineterminator="\n")
    print(f"data size: {df.shape}")

    # preprocess
    process_arglist = {
        "debug": False, 
        "pos": ["PROPN", "NOUN", "VERB", "ADJ", "ADV", "AUX", "INTJ", ], # pos to keep
        "keywords": ["autonomous", "av"],
        "min_len": 3, # min number of letters per word
        "min_words": 5
    }

    start_time = time.time()
    df_processed = df.iloc[:10000] if process_arglist["debug"] else df
    df_processed = preprocess(
        df_processed, 
        STOP_WORDS,
        process_arglist["pos"], 
        process_arglist["keywords"], 
        process_arglist["min_len"],
        process_arglist["min_words"]
    )

    print(f"processed data size: {df_processed.shape}, time: {time.time() - start_time:.2f}")
    
    save_path = os.path.join(
        arglist["data_path"],
        "av_classification",
        "av_tweets_processed.csv"
    )
    # df_processed.to_csv(save_path, index=False)

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)