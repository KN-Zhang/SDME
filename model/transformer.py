import torch
import torch.nn as nn
from model.linear_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return message ##x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([encoder_layer])
        self.full_attention = LinearAttention()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, agents, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        b = feat0.size(0)
        if feat1 is not None:
            for layer in self.layers:
                updated_agents0 = layer(agents.repeat(b, 1, 1), feat0, None, mask0)  ## 更新后的query
                updated_agents1 = layer(agents.repeat(b, 1, 1), feat1, None, mask1)
            A0 = self.full_attention(feat0[:, :, None, :], updated_agents0[:, :, None, :], updated_agents0[:, :, None, :], mask0, None)
            feat0 = feat0 + A0.squeeze(2)

            A1 = self.full_attention(feat1[:, :, None, :], updated_agents1[:, :, None, :], updated_agents1[:, :, None, :], mask1, None)
            feat1 = feat1 + A1.squeeze(2)

            return feat0, feat1
        else:
            for layer in self.layers:
                updated_agents0 = layer(agents.repeat(b, 1, 1), feat0, None, mask0)  ## 更新后的query
            # A0 = self.full_attention(feat0[:, :, None, :], updated_agents0[:, :, None, :], updated_agents0[:, :, None, :], mask0, None)
            A0 = self.full_attention(feat0[:, :, None, :], agents.repeat(b, 1, 1)[:, :, None, :], updated_agents0[:, :, None, :], mask0, None)
            
            feat0 = feat0 + A0.squeeze(2)

            return feat0, updated_agents0        
