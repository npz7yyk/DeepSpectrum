import torch.nn as nn
import torch


class AttentalSum(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)
        self.act = nn.Tanh()
        self.soft = nn.Softmax(dim=0)

    def forward(self, x, src_mask=None):
        # x: S B D, src_mask: B S
        weight = self.w(x)
        weight = self.act(weight).clone()

        if src_mask is not None:
            weight[src_mask.transpose(0, 1)] = -torch.inf
        weight = self.soft(weight)

        weighted_embed = torch.sum(x * weight, dim=0)
        return weighted_embed


class PrositFrag(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.peptide_embed_dim = kwargs.pop('peptide_embed_dim', 32)
        self.percursor_dim = kwargs.pop('peptide_embed_dim', 6)
        self.hidden_size = kwargs.pop('bi_dim', 256)
        self.max_sequence = kwargs.pop('max_lenght', 30)

        self.embedding = nn.Embedding(self.peptide_dim, self.peptide_embed_dim)
        self.bi = nn.GRU(input_size=self.peptide_embed_dim,
                         hidden_size=self.hidden_size,
                         bidirectional=True)
        self.drop3 = nn.Dropout(p=0.3)
        self.gru = nn.GRU(input_size=self.hidden_size * 2,
                          hidden_size=self.hidden_size * 2)
        self.agg = AttentalSum(self.hidden_size * 2)
        self.leaky = nn.LeakyReLU()

        self.side_encoder = nn.Linear(
            self.percursor_dim + 1, self.hidden_size * 2)

        self.gru_decoder = nn.GRU(input_size=self.hidden_size * 2,
                                  hidden_size=self.hidden_size * 2)
        self.in_frag = nn.Linear(self.max_sequence - 1, self.max_sequence - 1)
        self.final_decoder = nn.Linear(self.hidden_size * 2, 6)

    def comment(self):
        return "PrositFrag"

    def forward(self, x):
        self.bi.flatten_parameters()
        self.gru.flatten_parameters()
        self.gru_decoder.flatten_parameters()

        peptides = x['sequence_integer']
        nce = x['collision_energy_aligned_normed'].float()
        charge = x['precursor_charge_onehot'].float()
        B = peptides.shape[0]
        x = self.embedding(peptides)
        x = x.transpose(0, 1)
        x, _ = self.bi(x)
        x = self.drop3(x)
        x, _ = self.gru(x)
        x = self.drop3(x)
        x = self.agg(x)

        side_input = torch.cat([charge, nce], dim=1)
        side_info = self.side_encoder(side_input)
        side_info = self.drop3(side_info)

        x = x * side_info
        x = x.expand(self.max_sequence - 1, x.shape[0], x.shape[1])
        x, _ = self.gru_decoder(x)
        x = self.drop3(x)
        x_d = self.in_frag(x.transpose(0, 2))

        x = x * x_d.transpose(0, 2)
        x = self.final_decoder(x)
        x = self.leaky(x)
        x = x.transpose(0, 1).reshape(B, -1)
        return x
