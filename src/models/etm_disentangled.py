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
        self.zmu = nn.Linear(hidden_dim, num_topics)
        self.zlv = nn.Linear(hidden_dim, num_topics)
        self.smu = nn.Linear(hidden_dim, 1)
        self.slv = nn.Linear(hidden_dim, 1)
    
    def load_embedding(self, embedding, train_embedding=True):
        self.embedding.weight.data = embedding
        self.embedding.weight.requires_grad = train_embedding
    
    def forward(self, x):
        # avg pool doc embedding
        if self.use_embedding:
            x = x.matmul(self.embedding.weight) / x.sum(-1, keepdims=True)

        x = self.backbone(x)

        # infer topic
        z_mu = self.zmu(x)
        z_sd = torch.exp(0.5 * self.zlv(x))
        
        # infer sentiment
        s_mu = self.smu(x)
        s_sd = torch.exp(0.5 * self.slv(x))
        return z_mu, z_sd, s_mu, s_sd


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
        self.zmu = nn.Linear(hidden_dim, num_topics)
        self.zlv = nn.Linear(hidden_dim, num_topics)
        self.smu = nn.Linear(hidden_dim, 1)
        self.slv = nn.Linear(hidden_dim, 1)
    
    def load_embedding(self, embedding, train_embedding=True):
        self.embedding.weight.data = embedding
        self.embedding.weight.requires_grad = train_embedding
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        
        # infer topic
        z_mu = self.zmu(x)
        z_sd = torch.exp(0.5 * self.zlv(x))
        
        # infer sentiment
        s_mu = self.smu(x)
        s_sd = torch.exp(0.5 * self.slv(x))
        return z_mu, z_sd, s_mu, s_sd


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, embedding_dim):
        super().__init__()
        self.num_topics = num_topics

        # topic params
        self.topic_embedding = nn.Linear(embedding_dim, num_topics, bias=False)
        self.log_freq = nn.Parameter(torch.zeros(1, vocab_size), requires_grad=True) 
        self.sentiment_embedding = nn.Linear(embedding_dim, num_topics, bias=False)   
        
        # sentiment params
        self.y_alpha = nn.Linear(num_topics, 1, bias=True)
        self.y_beta = nn.Linear(num_topics, 1, bias=True)
        self.y_alpha_s = nn.Parameter(torch.randn(1, num_topics))
        self.y_beta_s = nn.Parameter(torch.randn(1, num_topics))
        
        nn.init.xavier_normal_(self.y_alpha_s)
        nn.init.xavier_normal_(self.y_beta_s)

    def forward(self, z, s, embedding):
        topic_term = torch.softmax(self.beta(embedding, s), dim=-1)
        p_words = z.unsqueeze(-2).matmul(topic_term).squeeze(-2)

        # sentiment params
        alpha, beta = self.y_dist(z, s)
        return p_words, alpha, beta

    def beta(self, embedding, s=0.):
        """ Compute topic term matrix [num_topics, num_vocab] """
        topic_term = self.topic_embedding(embedding).t() + self.log_freq        
        sentiment_term = self.sentiment_embedding(embedding).t()
        topic_term = (
            topic_term.unsqueeze(0) + \
            s.unsqueeze(-1) * sentiment_term.unsqueeze(0)
        )
        return topic_term

    def y_dist(self, z, s):
        alpha = torch.exp(
            self.y_alpha(z) + s * z.matmul(self.y_alpha_s.t()**2)
        ).squeeze(-1)
        beta = torch.exp(
            self.y_beta(z) - s * z.matmul(self.y_beta_s.t()**2)
        ).squeeze(-1)
        return alpha, beta


class DETM(nn.Module):
    """ Disentangled embedded topic model """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.num_topics = decoder.num_topics

        self.encoder = encoder
        self.decoder = decoder

    def model(self, data):
        """ generative process """ 
        bow, docs, sentiment, adj = data 
        embedding = self.encoder.embedding.weight

        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # topic prior
            z_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            z_scale = docs.new_ones((docs.shape[0], self.num_topics))
            z = pyro.sample(
                "z", 
                dist.Normal(z_loc, z_scale).to_event(1)
            )
            z = torch.softmax(z, dim=-1)

            # sentiment prior
            s_loc = docs.new_zeros((docs.shape[0], 1))
            s_scale = docs.new_ones((docs.shape[0], 1))
            s = pyro.sample(
                "s", 
                dist.Normal(s_loc, s_scale).to_event(1)
            )

            # generate words and sentiment
            word_count, y_alpha, y_beta = self.decoder(z, s, embedding)
            
            total_count = int(docs.sum(-1).max()) # multinomial requirement
            pyro.sample(
                "obs_word",
                dist.Multinomial(total_count, word_count),
                obs=bow
            )

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
            z_mu, z_sd, s_mu, s_sd = self.encoder(bow)

            z = pyro.sample(
                "z", 
                dist.Normal(z_mu, z_sd).to_event(1)
            )
            s = pyro.sample(
                "s", 
                dist.Normal(s_mu, s_sd).to_event(1)
            )

    def beta(self, s=0):
        """ construct sentiment conditioned topic term matrix
        
        args:
            s (float): sentiment factor value [-inf, inf]
        
        returns:
            topic_term: topic term matrix [num_topics, num_vocab] 
        """
        device = self.encoder.embedding.weight.device
        s = s * torch.ones(1, 1).to()

        with torch.no_grad():
            embedding = self.encoder.embedding.weight.data
            topic_term = self.decoder.beta(embedding, s)
        return topic_term.squeeze(0).cpu().detach()

    def y(self, query):
        """ get sentiment distributions for each query 

        args:
            query (list): sentiment query values
        
        returns:
            mu: sentiment means [num_query, num_topics]
            sd: sentiment stds [num_query, num_topics]
        """
        device = self.encoder.embedding.weight.device
        mu = torch.zeros(len(query), self.num_topics).to(device)
        sd = torch.zeros(len(query), self.num_topics).to(device)
        for i, s in enumerate(query):
            with torch.no_grad():
                z = torch.eye(self.num_topics).to(device)
                s = s * torch.ones(self.num_topics, 1).to(device)
                alpha, beta = self.decoder.y_dist(z, s)

                beta_dist = dist.Beta(alpha, beta)
                mu_ = beta_dist.mean.squeeze(-1)
                sd_ = beta_dist.variance.sqrt().squeeze(-1)

                # transform stats
                mu_ = 2 * mu_ - 1
                sd_ = 2 * sd_
            
            mu[i] = mu_
            sd[i] = sd_

        return mu.cpu().detach(), sd.cpu().detach()