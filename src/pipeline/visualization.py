import numpy as np
import matplotlib.pyplot as plt

def plot_history(df_history, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(df_history["epoch"], df_history["train_loss"], "-o", label="Train")
    ax.plot(df_history["epoch"], df_history["test_loss"], "-o", label="Test")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid()
    ax.legend()

    if show:
        plt.show()
    return fig

def plot_topic_term(topic_term, df_vocab, topk=20, show=False):
    df_vocab = df_vocab.sort_values(by="index")
    num_topics = len(topic_term)
    n_col = min(num_topics, 5)
    n_row = np.ceil(num_topics / 5).astype(int)

    fig, ax = plt.subplots(n_row, n_col, figsize=(15, 5*n_row), sharex=True)
    ax = np.ravel(ax)
    for i in range(num_topics):
        idx = topic_term[i].argsort()[::-1][:topk]
        top_words = [df_vocab["word"].iloc[i] for i in idx]
        weights = topic_term[i, idx]

        ax[i].barh(top_words, weights, height=0.7)
        ax[i].set_title(f'Topic {i +1}', fontdict={'fontsize': 12})
        ax[i].invert_yaxis()
        ax[i].tick_params(axis='both', which='major', labelsize=12)
        for j in 'top right left'.split():
            ax[i].spines[j].set_visible(False)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)

    if show:
        plt.show()
    return fig

def plot_topic_sentiment(mu, sd, query, show=False):
    """ plot topic sentiment for each query value """
    num_topics = mu.shape[1]

    fig, ax = plt.subplots(1, len(query), figsize=(5 * len(query), 5))
    if len(query) == 1:
        ax = [ax]
    for i, s in enumerate(query):
        ax[i].plot(np.arange(num_topics) + 1, mu[i], "o")
        ax[i].errorbar(
            np.arange(num_topics) + 1, 
            mu[i], 
            yerr=sd[i]
        )
        ax[i].set_ylim([-1, 1])
        ax[i].set_xlabel("Topic")
        ax[i].set_ylabel(f"Sentiment {s}")
        ax[i].grid()

    plt.tight_layout()
    
    if show:
        plt.show()
    return fig