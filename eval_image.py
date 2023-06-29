import os
import time
import argparse
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
import json
from collections import defaultdict
import math

from model import DistMult

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
from PIL import Image
from torchvision import transforms

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

def evaluate(model, id2entity, target_list, args):
    model.eval()
    torch.set_grad_enabled(False)

    overall_id_to_name = json.load(open('data/iwildcam_v2.0/overall_id_to_name.json'))

    img = Image.open(args.img_path).convert('RGB')
            
    transform_steps = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN, _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD)])
    h = transform_steps(img)
    r = torch.tensor([3])

    h = move_to(h, args.device).unsqueeze(0)
    r = move_to(r, args.device).unsqueeze(0)

    outputs = model.forward_ce(h, r, triple_type=('image', 'id'))

    y_pred = detach_and_clone(outputs.cpu())
    y_pred = y_pred.argmax(-1)

    pred_label = target_list[y_pred].item()
    species_label = overall_id_to_name[str(id2entity[pred_label])]
    print('species label = {}'.format(species_label))
    
    return

def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def generate_target_list(data, entity2id):
    sub = data.loc[(data["datatype_h"] == "image") & (data["datatype_t"] == "id"), ['t']]
    sub = list(sub['t'])
    categories = []
    for item in tqdm(sub):
        if entity2id[str(int(float(item)))] not in categories:
            categories.append(entity2id[str(int(float(item)))])
    # print('categories = {}'.format(categories))
    # print("No. of target categories = {}".format(len(categories)))
    return torch.tensor(categories, dtype=torch.long).unsqueeze(-1)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/iwildcam_v2.0/')
    parser.add_argument('--img-path', type=str, required=True, help='path to species image to be classified')
    parser.add_argument('--seed', type=int, default=813765)
    parser.add_argument('--ckpt-path', type=str, default=None, help='path to ckpt for restarting expt')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--location_input_dim', type=int, default=2)
    parser.add_argument('--time_input_dim', type=int, default=1)
    parser.add_argument('--mlp_location_numlayer', type=int, default=3)
    parser.add_argument('--mlp_time_numlayer', type=int, default=3)

    parser.add_argument('--img-embed-model', choices=['resnet18', 'resnet50'], default='resnet50')
    parser.add_argument('--use-data-subset', action='store_true')
    parser.add_argument('--subset-size', type=int, default=10)

    args = parser.parse_args()

    print('args = {}'.format(args))
    args.device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)

    entity_id_file = os.path.join(args.data_dir, 'entity2id_subtree.json')

    if not os.path.exists(entity_id_file):
        entity2id = {} # each of triple types have their own entity2id
        
        for i in tqdm(range(datacsv.shape[0])):
            if datacsv.iloc[i,1] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,0]))))

            if datacsv.iloc[i,-2] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,-3]))))
        json.dump(entity2id, open(entity_id_file, 'w'))
    else:
        entity2id = json.load(open(entity_id_file, 'r'))
    
    id2entity = {v:k for k,v in entity2id.items()}

    num_ent_id = len(entity2id)

    # print('len(entity2id) = {}'.format(len(entity2id)))
    
    target_list = generate_target_list(datacsv, entity2id)

    model = DistMult(args, num_ent_id, target_list, args.device)

    model.to(args.device)

    # restore from ckpt
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'], strict=False)
        print('ckpt loaded...')

    evaluate(model, id2entity, target_list, args)
