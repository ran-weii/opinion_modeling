import time
import pandas as pd
from tqdm import tqdm
from pyro.infer import SVI, Trace_ELBO

from src.pipeline.metrics import get_top_words, get_topic_diversity

def run_epoch(model, loader, train=True, debug=False):
    """ pass data through model 

    args:
        model: 
        loader: train or test dataloader
        train: train mode, default=True
        debug: default=False
    """
    stats = []
    for i, data_batch in enumerate(loader):
        if train:
            loss = model.step(data_batch)
        else:
            loss = model.evaluate_loss(data_batch)
        
        stats.append({"loss": loss / len(data_batch[0])})

        if debug:
            break

    stats = pd.DataFrame(stats).mean(0).to_dict()
    stats.update({"train": "train" if train else "test"})
    return stats

def train(
    model, optimizer, train_loader, test_loader, epochs, 
    df_vocab, callback=None, debug=False
    ):
    svi = SVI(
        model.model, 
        model.guide, 
        optimizer, 
        loss=Trace_ELBO()
    )

    start_time = time.time()
    bar = tqdm(range(epochs))
    history = []
    for e in bar: 
        train_stats = run_epoch(svi, train_loader, train=True, debug=debug)
        test_stats = run_epoch(svi, test_loader, train=False, debug=debug)
        
        # collect stats
        tnow = time.time() - start_time
        topic_term = model.beta().data.numpy()
        td = get_topic_diversity(get_top_words(topic_term, df_vocab))
        stats = {
            "epoch": e + 1,
            "time": tnow,
            "train_loss": train_stats["loss"],
            "test_loss": test_stats["loss"],
            "td": td,
        }
        history.append(stats)
        bar.set_description("train_loss: {:.3f}, test_loss: {:.3f}, time: {:.2f}".format(
            train_stats["loss"],
            test_stats["loss"],
            tnow,
        ))

        if callback is not None:
            callback(model, pd.DataFrame(history))
        
        if debug and e > 0:
            break
    return model, history