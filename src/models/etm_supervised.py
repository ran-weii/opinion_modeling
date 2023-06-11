import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from src.models.backbone import MLP, GNN

class MLPEncoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        num_topics, 
        hidden_dim,
        num_layers,
        activation,
        use_embedding=True
        ):
        super().__init__()
        input_dim = embedding_dim if use_embedding else vocab_size
        self.use_embedding = use_embedding

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.backbone = MLP(
            input_dim,
            hidden_dim,
            num_layers,
            activation
        )
        self.mu = nn.Linear(hidden_dim, num_topics)
        self.lv = nn.Linear(hidden_dim, num_topics)
    
    def load_embedding(self, embedding, train_embedding=True):
        self.embedding.weight.data = embedding
        self.embedding.weight.requires_grad = train_embedding
    
    def forward(self, x):
        # avg pool doc embedding
        if self.use_embedding:
            x = x.matmul(self.embedding.weight) / x.sum(-1, keepdims=True)

        x = self.backbone(x)
        mu = self.mu(x)
        sd = torch.exp(0.5 * self.lv(x))
        return mu, sd


class GNNEncoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        num_topics, 
        hidden_dim,
        num_gnn_layers,
        num_heads,
        num_mlp_layers,
        activation,
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.backbone = GNN(
            embedding_dim,
            hidden_dim,
            num_gnn_layers,
            num_heads,
            num_mlp_layers,
            activation
        )
        self.mu = nn.Linear(hidden_dim, num_topics)
        self.lv = nn.Linear(hidden_dim, num_topics)
    
    def load_embedding(self, embedding, train_embedding=True):
        self.embedding.weight.data = embedding
        self.embedding.weight.requires_grad = train_embedding

    def forward(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        mu = self.mu(x)
        sd = torch.exp(0.5 * self.lv(x))
        return mu, sd
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, embedding_dim):
        super().__init__()
        self.num_topics = num_topics

        # topic params
        self.topic_embedding = nn.Linear(embedding_dim, num_topics, bias=False)
        self.log_freq = nn.Parameter(torch.zeros(1, vocab_size), requires_grad=True)
        
        # sentiment params
        self.y_alpha = nn.Linear(num_topics, 1, bias=True)
        self.y_beta = nn.Linear(num_topics, 1, bias=True)

    def forward(self, z, embedding):
        topic_term = torch.softmax(self.beta(embedding), dim=-1)
        p_words = z.matmul(topic_term)

        # sentiment beta params
        alpha = self.y_alpha(z).exp().squeeze(-1)
        beta = self.y_beta(z).exp().squeeze(-1)
        return p_words, alpha, beta

    def beta(self, word_embedding):
        """ Compute topic term matrix [num_topics, num_vocab] """
        beta = self.topic_embedding(word_embedding).t() + self.log_freq
        return beta


class SETM(nn.Module):
    """ Supervised embedded topic model """
    def __init__(self, encoder, decoder, pred_sentiment=True):
        super().__init__()
        self.num_topics = decoder.num_topics
        self.pred_sentiment = pred_sentiment

        self.encoder = encoder
        self.decoder = decoder

    def model(self, data):
        """ generative process """ 
        bow, docs, sentiment, adj = data 
        embedding = self.encoder.embedding.weight

        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # standard logit normal prior 
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", 
                dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = torch.softmax(logtheta, dim=-1)

            # generate words and sentiment
            word_count, y_alpha, y_beta = self.decoder(theta, embedding)
            total_count = int(docs.sum(-1).max()) # multinomial requirement
            pyro.sample(
                "obs_word",
                dist.Multinomial(total_count, word_count),
                obs=bow
            )

            if self.pred_sentiment:
                pyro.sample(
                    "obs_sentiment",
                    dist.Beta(y_alpha, y_beta),
                    obs=sentiment
                )

    def guide(self, data):
        """ inference process """
        bow, docs, sentiment, adj = data 

        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            logtheta_loc, logtheta_scale = self.encoder(bow)
            logtheta = pyro.sample(
                "logtheta", 
                dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
            )

    def beta(self):
        """ topic term matrix: [num_topics, num_vocab] """
        with torch.no_grad():
            embedding = self.encoder.embedding.weight.data
            return self.decoder.beta(embedding).cpu().detach()

    def y(self):
        with torch.no_grad():
            y_alpha = self.decoder.y_alpha.weight.data.exp()
            y_beta = self.decoder.y_beta.weight.data.exp()

            beta_dist = dist.Beta(y_alpha, y_beta)
            mu = beta_dist.mean.squeeze(-1)
            std = beta_dist.variance.sqrt().squeeze(-1)

            # transform stats
            mu = 2 * mu - 1
            std = 2 * std
        return mu.cpu(), std.cpu()