import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple

from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights

class DistMult(nn.Module):
    def __init__(self, args, num_ent_uid, target_list, device, all_locs=None, num_habitat=None, all_timestamps=None):
        super(DistMult, self).__init__()
        self.args = args
        self.num_ent_uid = num_ent_uid

        self.num_relations = 4

        self.ent_embedding = torch.nn.Embedding(self.num_ent_uid, args.embedding_dim, sparse=False)
        self.rel_embedding = torch.nn.Embedding(self.num_relations, args.embedding_dim, sparse=False)

        self.location_embedding = MLP(args.location_input_dim, args.embedding_dim, args.mlp_location_numlayer)

        self.time_embedding = MLP(args.time_input_dim, args.embedding_dim, args.mlp_time_numlayer)
        
        if self.args.img_embed_model == 'resnet50':
            self.image_embedding = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.image_embedding.fc = nn.Linear(2048, args.embedding_dim)
        else:
            self.image_embedding = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.image_embedding.fc = nn.Linear(512, args.embedding_dim)

        self.target_list = target_list
        
        if all_locs is not None:
            self.all_locs = all_locs.to(device)
        if all_timestamps is not None:
            self.all_timestamps = all_timestamps.to(device)

        self.args = args
        self.device = device

        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        nn.init.xavier_uniform_(self.image_embedding.fc.weight.data)

    def forward_ce(self, h, r, triple_type=None):
        emb_h = self.batch_embedding_concat_h(h) # [batch, hid]
        
        emb_r = self.rel_embedding(r.squeeze(-1)) # [batch, hid]

        emb_hr = emb_h * emb_r  # [batch, hid]

        if triple_type == ('image', 'id'):
            score = torch.mm(emb_hr, self.ent_embedding.weight[self.target_list.squeeze(-1)].T) # [batch, n_ent]
        elif triple_type == ('id', 'id'):
            score = torch.mm(emb_hr, self.ent_embedding.weight.T) # [batch, n_ent]
        elif triple_type == ('image', 'location'):
            loc_emb = self.location_embedding(self.all_locs) # computed for each batch
            score = torch.mm(emb_hr, loc_emb.T)
        elif triple_type == ('image', 'time'):
            time_emb = self.time_embedding(self.all_timestamps)
            score = torch.mm(emb_hr, time_emb.T)
        else:
            raise NotImplementedError

        return score

    def batch_embedding_concat_h(self, e1):
        e1_embedded = None
        
        if len(e1.size())==1 or e1.size(1) == 1:  # uid
            # print('ent_embedding = {}'.format(self.ent_embedding.weight.size()))
            e1_embedded = self.ent_embedding(e1.squeeze(-1))
        elif e1.size(1) == 15:  # time
            e1_embedded = self.time_embedding(e1)
        elif e1.size(1) == 2:  # GPS
            e1_embedded = self.location_embedding(e1)
        elif e1.size(1) == 3:  # Image
            e1_embedded = self.image_embedding(e1)

        return e1_embedded


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=3,
                 p_dropout=0.0,
                 bias=True):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.p_dropout = p_dropout
        step_size = (input_dim - output_dim) // num_layers
        hidden_dims = [output_dim + (i * step_size)
                       for i in reversed(range(num_layers))]

        mlp = list()
        layer_indim = input_dim
        for hidden_dim in hidden_dims:
            mlp.extend([nn.Linear(layer_indim, hidden_dim, bias),
                        nn.Dropout(p=self.p_dropout, inplace=True),
                        nn.PReLU()])

            layer_indim = hidden_dim

        self.mlp = nn.Sequential(*mlp)

        # initialize weights
        self.init()

    def forward(self, x):
        return self.mlp(x)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)
