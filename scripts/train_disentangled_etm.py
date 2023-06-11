import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
from gensim.models import FastText

from src.models.etm_disentangled import MLPEncoder, Decoder, DETM
from src.pipeline.data import train_test_split
from src.pipeline.training import train
from src.pipeline.logger import Logger, load_cp
from src.pipeline.visualization import plot_topic_term, plot_topic_sentiment

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="detm")
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--cp_path", type=str, default="none")
    parser.add_argument("--fasttext_path", type=str, 
        default="sample_ratio_0.8_embedding_size_300_window_5_min_count_3_lr_0.05_seed"
    )
    # data args
    parser.add_argument("--sample_ratio", type=float, default=0.99)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--min_df", type=float, default=3)
    parser.add_argument("--max_vocab", type=int, default=10000)
    # model args
    parser.add_argument("--num_topics", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--use_embedding", type=bool_, default=False)
    parser.add_argument("--pred_sentiment", type=bool_, default=True)
    # train args
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cp_every", type=int, default=1)
    parser.add_argument("--debug", type=bool_, default=False)
    parser.add_argument("--save", type=bool_, default=True)
    parser.add_argument("--seed", type=int, default=0)
    arglist = vars(parser.parse_args())
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    pyro.set_rng_seed(arglist["seed"])
    pyro.clear_param_store()
    torch.cuda.empty_cache()
    print(f"training {arglist['model']} with arglist: {arglist}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    file_path = os.path.join(
        arglist["data_path"],
        "av_classification",
        "av_tweets_processed.csv"
    )
    df = pd.read_csv(file_path, lineterminator="\n")
    print(f"data size: {df.shape}")
    
    fasttext_model = None
    if arglist["fasttext_path"] != "none":
        fasttext_path = os.path.join(
            arglist["exp_path"], "fasttext", arglist["fasttext_path"]
        )
        fasttext_model = FastText.load(os.path.join(fasttext_path, "model.m"))
        print("loaded fasttext model")
    
    train_loader, test_loader, df_vocab, embedding_matrix = train_test_split(
        df, 
        embedding_dict=fasttext_model,
        sample_ratio=arglist["sample_ratio"],
        train_ratio=arglist["train_ratio"],
        batch_size=arglist["batch_size"],
        min_df=arglist["min_df"],
        max_vocab=arglist["max_vocab"],
        seed=arglist["seed"],
        device=device,
    )

    # init model
    vocab_size = df_vocab.shape[0]
    encoder = MLPEncoder(
        vocab_size,
        arglist["embedding_dim"],
        arglist["num_topics"],
        arglist["hidden_dim"],
        arglist["num_layers"],
        arglist["activation"],
        arglist["use_embedding"],
    )
    decoder = Decoder(
        vocab_size,
        arglist["num_topics"],
        arglist["embedding_dim"]
    )
    model = DETM(encoder, decoder)
    model.encoder.load_embedding(embedding_matrix)
    model.to(device)
    
    optimizer = pyro.optim.Adam({"lr": arglist["lr"]})
    
    print(model)

    cp_history = None
    if arglist["cp_path"] != "none":
        cp_history = load_cp(arglist["cp_path"], model, optimizer, device)

    # init callback
    logger = None
    if arglist["save"]:
        logger = Logger(arglist, cp_history)

    model, history = train(
        model, 
        optimizer, 
        train_loader, 
        test_loader, 
        arglist["epochs"], 
        df_vocab, 
        callback=logger,
        debug=arglist["debug"],
    )
    
    # save vocab
    if arglist["save"]:
        logger.save_history(pd.DataFrame(history))
        logger.save_checkpoint(model)
        
        # save topic term fig
        query = [-1, 0, 1]
        topk = 20
        for q in query:
            topic_term = model.beta(q).data.numpy()
            fig_topic = plot_topic_term(topic_term, df_vocab, topk, show=False)
            fig_topic.savefig(os.path.join(logger.save_path, f"topic_term_{q}.png"), dpi=100)

        # save topic sentiment fig
        y_mu, y_std = model.y(query)
        fig_sentiment = plot_topic_sentiment(y_mu, y_std, query, show=False)
        fig_sentiment.savefig(os.path.join(logger.save_path, "topic_sentiment.png"), dpi=100)

        plt.clf(); plt.close("all")
        print(f"done, results saved: {arglist['save']}")

    return model, df_vocab, pd.DataFrame(history)

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)