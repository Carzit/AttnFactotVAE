import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from utils import multiLinear

class Exp(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return torch.exp(x)


class AttnFeatureExtractor(nn.Module):
    def __init__(self, 
                 fundamental_feature_size, 
                 quantity_price_feature_size,
                 hidden_size, 
                 num_gru_layer=1,
                 gru_dropout=0) -> None:
        super().__init__()
        self.norm_layer1 = nn.LayerNorm(quantity_price_feature_size)
        self.proj_layer = nn.Sequential(nn.Linear(quantity_price_feature_size, hidden_size), 
                                        nn.LeakyReLU())
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          batch_first=False,
                          num_layers=num_gru_layer,
                          dropout=gru_dropout)
        
        self.norm_layer2 = nn.LayerNorm(fundamental_feature_size)
        self.q_layer = nn.Linear(fundamental_feature_size, hidden_size)
        self.k_layer = nn.Linear(fundamental_feature_size, hidden_size)
        self.v_layer = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, fundamental_features, quantity_price_features):
        fundamental_features = self.norm_layer2(fundamental_features)
        q = self.q_layer(fundamental_features)
        k = self.k_layer(fundamental_features)

        quantity_price_features = self.norm_layer1(quantity_price_features)
        quantity_price_features = self.proj_layer(quantity_price_features)
        output, hidden_state = self.gru(quantity_price_features)
        v = self.v_layer(output[-1]) # -> (batch_size, hidden_size)
        qk = torch.matmul(q, k.T)
        attn = torch.matmul(qk, v)
        residual = attn + v
        return residual # -> (batch_size, hidden_size)

class FeatureExtractor(nn.Module):
    def __init__(self, 
                 quantity_price_feature_size,
                 hidden_size, 
                 num_gru_layer=1,
                 gru_dropout=0) -> None:
        super().__init__()
        self.norm_layer = nn.LayerNorm(quantity_price_feature_size)
        self.proj_layer = nn.Sequential(nn.Linear(quantity_price_feature_size, hidden_size), 
                                        nn.LeakyReLU())
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          batch_first=False,
                          num_layers=num_gru_layer,
                          dropout=gru_dropout)
        
    def forward(self, quantity_price_features):
        quantity_price_features = self.norm_layer(quantity_price_features)
        quantity_price_features = self.proj_layer(quantity_price_features)
        output, hidden_state = self.gru(quantity_price_features)
        return output[-1]  # -> (batch_size, hidden_size)


class AttnRet(nn.Module):
    def __init__(self, 
                 fundamental_feature_size, 
                 quantity_price_feature_size,
                 num_gru_layers, 
                 gru_hidden_size,
                 gru_drop_out = 0.1,
                 num_fc_layers = 2) -> None:
        super().__init__()
        self.feature_extractor = AttnFeatureExtractor(fundamental_feature_size=fundamental_feature_size,
                                                      quantity_price_feature_size=quantity_price_feature_size,
                                                      hidden_size=gru_hidden_size,
                                                      num_gru_layer=num_gru_layers,
                                                      gru_dropout=gru_drop_out)
        self.fc_layers = multiLinear(input_size=gru_hidden_size,
                                    output_size=1,
                                    num_layers=num_fc_layers)
        
    def forward(self, fundamental_features, quantity_price_features):
        residual = self.feature_extractor(fundamental_features, quantity_price_features)
        out = self.fc_layers(residual)
        out = out.squeeze()
        return out
    

class PortfolioLayer(nn.Module):
    """
    Because the number of individual stocks in cross-section is large and varies with time, 
    instead of using stock returns y directly, we construct a set of portfolios inspired by (Gu, Kelly, and Xiu 2021), 
    these portfolios are dynamically re-weighted on the basis of stock latent features, i.e., 
        y_p = y * φ_p(e) = y * a_p , 
    where a_p ∈ R^M denotes the weight of M portfolios.

        a_p_(i,j) = softmax(w_p * e_i + b_p)
        y_p_(j) = ∑ y(i) * a_p_(i,j)
    where a_p_(i,j) denotes the weight of i-th stock in j-th portfolio and meets ∑a_p_(i,j) = 1, y_p ∈ R^M is the vector of portfolio returns. 
    The main advantages of constructing portfolios lie in: 1) reducing the input dimension and avoiding too many parameters. 2) robust to the missing stocks in cross-section and thus suitable for the market
    """

    def __init__(self, num_portfolios, input_size) -> None:
        super(PortfolioLayer, self).__init__()
        self.w_p = nn.Parameter(torch.randn(num_portfolios, input_size))
        self.b_p = nn.Parameter(torch.randn(num_portfolios, 1))

    def forward(self, y:torch.Tensor, e:torch.Tensor): # y: [num_stocks] e: [num_stocks, input_size]
        a_p = F.softmax(torch.matmul(self.w_p, e.T) + self.b_p, dim=-1) #-> [num_portfolios, num_stocks]
        y_p = torch.matmul(a_p, y) #-> [num_portfolios]
        return y_p

class FactorEncoder(nn.Module):
    """
    Factor encoder extracts posterior factors `z_post` from the future stock returns `y` and the latent features `e`
        [μ_post, σ_post] = φ_enc(y, e)
        z_post ~ N (μ_post, σ_post^2)
    where `z_post` is a random vector following the independent Gaussian distribution, 
    which can be described by the mean `μ_post` ∈ R^K and the standard deviation σ_post ∈ R^K, K is the number of factors.

    Because the number of individual stocks in cross-section is large and varies with time, 
    instead of using stock returns y directly, we construct a set of portfolios inspired by (Gu, Kelly, and Xiu 2021), 
    these portfolios are dynamically re-weighted on the basis of stock latent features, i.e., 
        y_p = y · φ_p(e) = y · a_p , 
    where a_p ∈ R^M denotes the weight of M portfolios.

        a_p_(i,j) = softmax(w_p * e_i + b_p)
        y_p_(j) = ∑ y(i) * a_p_(i,j)
    where a_p_(i,j) denotes the weight of i-th stock in j-th portfolio and meets ∑a_p_(i,j) = 1, y_p ∈ R^M is the vector of portfolio returns. 
    The main advantages of constructing portfolios lie in: 1) reducing the input dimension and avoiding too many parameters. 2) robust to the missing stocks in cross-section and thus suitable for the market

    And then the mean and the std of posterior factors are output by a mapping layer [μ_post, σ_post] = φ_map(y_p)
        μ_post = w * y_p + b
        σ_post = Softplus(w * y_p + b)
    where Softplus(x) = log(1 + exp(x))

    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 latent_size,
                 std_activ:Literal["exp", "softplus"] = "exp") -> None:
        super(FactorEncoder, self).__init__()
        self.portfoliolayer = PortfolioLayer(input_size=input_size, num_portfolios=hidden_size)
        self.map_mu_z_layer = nn.Linear(hidden_size, latent_size)
        if std_activ == "exp":
            self.map_sigma_z_layer = nn.Sequential(nn.Linear(hidden_size, latent_size),
                                                   Exp())
        elif std_activ == "softplus":
            self.map_sigma_z_layer = nn.Sequential(nn.Linear(hidden_size, latent_size),
                                                   nn.Softplus())

    def forward(self, y:torch.Tensor, e:torch.Tensor):# y: [num_stocks] e: [num_stocks, num_features]
        y_p = self.portfoliolayer(y, e) #-> [hidden_size(num_portfolios)]
        mu_z = self.map_mu_z_layer(y_p) #-> [latent_size(num_factors)]
        sigma_z = self.map_sigma_z_layer(y_p) #-> [latent_size(num_factors)]
        return mu_z, sigma_z

class AlphaLayer(nn.Module):
    """
    Alpha layer outputs idiosyncratic returns α from the latent features e. 
    We assume that α is a Gaussian random vector described by 
        α ~ N (μ_α, σ_α^2)
    where the mean μ_α ∈ R^N and the std σ_α ∈ R^N are output by a distribution network π_α, i.e., [μ_α, σ_α] = π_α(e). 
    Specifically,
        h_α = LeakyReLU(w_α * e + b_α)
        μ_α = w_μ_α * h_α + b_μ_α
        σ_α = Softplus(w_σ_α * h_α + b_σ_α)
    where h_α ∈ R^H is the hidden state.
    """
    def __init__(self, input_size, hidden_size, std_activ:Literal["exp", "softplus"] = "exp") -> None:
        super(AlphaLayer, self).__init__()
        self.alpha_h_layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                           nn.LeakyReLU())
        self.alpha_mu_layer = nn.Linear(hidden_size, 1)
        if std_activ == "exp":
            self.alpha_sigma_layer = nn.Sequential(nn.Linear(hidden_size, 1),
                                                   Exp())
        elif std_activ == "softplus":
            self.alpha_sigma_layer = nn.Sequential(nn.Linear(hidden_size, 1),
                                                   nn.Softplus())
    
    def forward(self, e):#e: [num_stocks, num_features]
        h_alpha = self.alpha_h_layer(e) #->[num_stocks, num_portfolios(hidden_size)]
        mu_alpha = self.alpha_mu_layer(h_alpha).squeeze() #->[num_stocks]
        sigma_alpha = self.alpha_sigma_layer(h_alpha).squeeze() #->[num_stocks]
        return mu_alpha, sigma_alpha
    
class BetaLayer(nn.Module):
    """
    Beta layer calculates factor exposure β ∈ R^{N*K} from the latent features e by linear mapping. Formally,
        β = φ_β(e) = w_β * e + b_β
    """
    def __init__(self, input_size, latent_size) -> None:
        super(BetaLayer, self).__init__()
        self.beta_layer = nn.Linear(input_size, latent_size)
    
    def forward(self, e):#e: [num_stocks, num_features(input_size)]
        beta = self.beta_layer(e) 
        return beta #->[num_stocks, num_fators(latent_size)]

class FactorDecoder(nn.Module):
    """
    Factor decoder uses factors z and the latent feature e to calculate stock returns `y_hat`
        y_hat = φ_dec(z, e) = α + β * z
    Essentially, the decoder network φdec consists of alpha layer and beta layer.

    Note that α and z are both follow independent Gaussian distribution, and thus the output of decoder y_hat ~ N(μ_y , σ_y^2), where
        μ_y = μ_α + ∑ β_k * μ_z_k
        σ_y = \sqrt{ σ_α^2 + ∑ β_k ^ 2 σ_z_k^2 }
    where μ_z , σ_z ∈ R^K are the mean and the std of factors respectively.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 latent_size,
                 std_activ:Literal["exp", "softplus"] = "exp") -> None: 
        super(FactorDecoder, self).__init__()
        self.alpha_layer = AlphaLayer(input_size, hidden_size, std_activ=std_activ)
        self.beta_layer = BetaLayer(input_size, latent_size)
    
    def forward(self, e, mu_z, sigma_z):# e: [num_stocks, num_features], mu_z: [latent_size(num_factors)], sigma_z: [latent_size(num_factors)]
        mu_alpha, sigma_alpha = self.alpha_layer(e) #->[num_stocks]
        beta = self.beta_layer(e) #->[num_stocks, latent_size(num_fators)]

        mu_y = mu_alpha +  torch.matmul(beta, mu_z) #->[num_stocks]
        sigma_y = torch.sqrt(sigma_alpha**2 + torch.matmul(beta**2, sigma_z**2)) #->[num_stocks]
        
        y = self.reparameterization(mu_y, sigma_y) #->[num_stocks]
        return y

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std).to(next(self.parameters()).device)      
        y = mean + std * epsilon
        return y

class SingleHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleHeadAttention, self).__init__()
        self.q = nn.Parameter(torch.randn(hidden_size))
        self.w_key = nn.Linear(input_size, hidden_size, bias=False)
        self.w_value = nn.Linear(input_size, hidden_size, bias=False)
        self.epsilon = 1e-6

    def forward(self, e):# e: [num_stocks, input_size(features)]
        k = self.w_key(e)  # -> [num_stocks, hidden_size]
        v = self.w_value(e)  # -> [num_stocks, hidden_size]
        
        q_norm = self.q / self.q.norm(dim=-1, keepdim=True)  # -> [hidden_size]
        k_norm = k / k.norm(dim=-1, keepdim=True)  # -> [num_stocks, hidden_size]
        
        attn_scores = F.relu(torch.matmul(q_norm, k_norm.transpose(-1,-2))) # -> [num_stocks]
        attn_weights = attn_scores / (attn_scores.sum(dim=-1, keepdim=True) + self.epsilon)   # -> [num_stocks]
        
        h_att = torch.matmul(attn_weights, v)  # -> [hidden_size]
        return h_att

class MultiHeadAttention(nn.Module):
    """
    Considering that a factor usually represents a certain type of risk premium in the market (such as the size factor focuses on the risk premium of small-cap stocks), we design a muti-head global attention mechanism to integrate the diverse global representations of the market in parallel, and extract factors from them to represent diverse risk premium of market. Formally, a single-head attention performs as

        k_i = w_key * e_i, v_i = w_value * e_i
        a_i = relu(q × k_i.T) / norm(q) / norm(k_i)
        h_att = φ_att(e) = ∑ a_i * v_i

    where query token q ∈ R^H is a learnable parameter, and h_att ∈ R^H is the global representation of market. 
    The muti-head attention concatenates K independent heads together 

        h_muti = Concat([φ_att_1(e), . . . , φ_att_K(e)]) 
    
    where h_muti ∈ R^(K * H) is the muti-global representation.
    """
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([SingleHeadAttention(input_size, hidden_size) for _ in range(num_heads)])
        
    def forward(self, e):
        head_outputs = [head(e) for head in self.heads]
        h_muti = torch.stack(head_outputs, dim=-2) # -> [num_heads, hidden_size]
        return h_muti #->(K, H)
    
class DistributionNetwork(nn.Module):
    """
    And then we use a distribution network πprior to predict the mean µ_prior and the std σ_prior of prior factors z_prior.

        [µ_prior, σ_prior] = π_prior(h_muti)

    """
    def __init__(self, hidden_size, std_activ:Literal["exp", "softplus"] = "exp"):
        super(DistributionNetwork, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, 1)
        
        if std_activ == "exp":
            self.sigma_layer = nn.Sequential(nn.Linear(hidden_size, 1),
                                             Exp())
        elif std_activ == "softplus":
            self.sigma_layer = nn.Sequential(nn.Linear(hidden_size, 1),
                                             nn.Softplus())
    
    def forward(self, h_multi):#h_multi: [num_factors(=num_heads), hidden_size]
        mu_prior = self.mu_layer(h_multi).squeeze() #->[num_factors]
        sigma_prior = self.sigma_layer(h_multi).squeeze() #->[num_factors]
        return mu_prior, sigma_prior

class FactorPredictor(nn.Module):
    """
    Factor predictor extracts prior factors z_prior from the stock latent features e:
        [μ_prior, σ_prior] = φ_pred(e)
        z_prior ∼ N (μ_prior, σ_prior^2)
    where z_prior is a Gaussian random vector, described by the mean μ_prior ∈ R^K and the std σprior ∈ R^K, K is the number of factors. 
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 latent_size, 
                 std_activ:Literal["exp", "softplus"] = "exp") -> None:
        super(FactorPredictor, self).__init__()
        self.multihead_attention = MultiHeadAttention(input_size, hidden_size, latent_size)
        self.distribution_network = DistributionNetwork(hidden_size, std_activ=std_activ)

    def forward(self, e):
        h_multi = self.multihead_attention(e)
        mu_prior, sigma_prior = self.distribution_network(h_multi)
        return mu_prior, sigma_prior

class FactorVAE(nn.Module):
    """
    Pytorch Implementation of FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns (https://ojs.aaai.org/index.php/AAAI/article/view/20369)

    Our model follows the encoder-decoder architecture of VAE, to learn an optimal factor model, which can reconstruct the cross-sectional stock returns by several factors well. As shown in Figure 3, with access to future stock returns, the encoder plays a role as an oracle, which can extract optimal factors from future data, called posterior factors, and then the decoder reconstructs future stock returns by the posterior factors. Specially, the factors in our model are regarded as the latent variables in VAE, with the capacity of modeling noisy data. 
    Concretely, this architecture contains three components: feature extractor, factor encoder and factor decoder.
    """
    def __init__(self, 
                 quantity_price_feature_size,
                 num_gru_layers, 
                 gru_hidden_size,
                 hidden_size,
                 latent_size,
                 gru_drop_out = 0.1, 
                 std_activ:Literal["exp", "softplus"] = "exp") -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(quantity_price_feature_size=quantity_price_feature_size,
                                                  hidden_size=gru_hidden_size,
                                                  num_gru_layer=num_gru_layers,
                                                  gru_dropout=gru_drop_out)
        self.encoder = FactorEncoder(input_size=gru_hidden_size, 
                                     hidden_size=hidden_size,
                                     latent_size=latent_size,
                                     std_activ=std_activ)
        self.predictor = FactorPredictor(input_size=gru_hidden_size,
                                         hidden_size=hidden_size,
                                         latent_size=latent_size,
                                         std_activ=std_activ)
        self.decoder = FactorDecoder(input_size=gru_hidden_size,
                                     hidden_size=hidden_size,
                                     latent_size=latent_size,
                                     std_activ=std_activ)
    def forward(self, quantity_price_feature, y):
        e = self.feature_extractor(quantity_price_feature)
        mu_posterior, sigma_posterior = self.encoder(y, e)
        mu_prior, sigma_prior = self.predictor(e)
        y_hat = self.decoder(e, mu_posterior, sigma_posterior)
        return y_hat, mu_posterior, sigma_posterior, mu_prior, sigma_prior
    
    def predict(self, quantity_price_feature):
        e = self.feature_extractor(quantity_price_feature)
        mu_prior, sigma_prior = self.predictor(e)
        y_pred = self.decoder(e, mu_prior, sigma_prior)
        return y_pred, mu_prior, sigma_prior    


class AttnFactorVAE(nn.Module):
    """
    Pytorch Implementation of AttnFactorVAE, which is a combination of RiskAttention and FactorVAE.
    """
    def __init__(self, 
                 fundamental_feature_size, 
                 quantity_price_feature_size,
                 num_gru_layers, 
                 gru_hidden_size,
                 hidden_size,
                 latent_size,
                 gru_drop_out = 0.1, 
                 std_activ:Literal["exp", "softplus"] = "exp") -> None:
        super().__init__()
        self.feature_extractor = AttnFeatureExtractor(fundamental_feature_size=fundamental_feature_size,
                                                      quantity_price_feature_size=quantity_price_feature_size,
                                                      hidden_size=gru_hidden_size,
                                                      num_gru_layer=num_gru_layers,
                                                      gru_dropout=gru_drop_out)
        self.encoder = FactorEncoder(input_size=gru_hidden_size, 
                                     hidden_size=hidden_size,
                                     latent_size=latent_size,
                                     std_activ=std_activ)
        self.predictor = FactorPredictor(input_size=gru_hidden_size,
                                         hidden_size=hidden_size,
                                         latent_size=latent_size,
                                         std_activ=std_activ)
        self.decoder = FactorDecoder(input_size=gru_hidden_size,
                                     hidden_size=hidden_size,
                                     latent_size=latent_size,
                                     std_activ=std_activ)
    def forward(self, fundamental_feature, quantity_price_feature, y):
        e = self.feature_extractor(fundamental_feature, quantity_price_feature)
        mu_posterior, sigma_posterior = self.encoder(y, e)
        mu_prior, sigma_prior = self.predictor(e)
        y_hat = self.decoder(e, mu_posterior, sigma_posterior)
        return y_hat, mu_posterior, sigma_posterior, mu_prior, sigma_prior
    
    def predict(self, fundamental_feature, quantity_price_feature):
        e = self.feature_extractor(fundamental_feature,  quantity_price_feature)
        mu_prior, sigma_prior = self.predictor(e)
        y_pred = self.decoder(e, mu_prior, sigma_prior)
        return y_pred, mu_prior, sigma_prior 

