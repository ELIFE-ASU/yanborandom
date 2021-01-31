import Dataset

import importlib
import csv

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import matplotlib.pyplot as plt


all_things = Dataset.get_recipe(Dataset.path)
name2idx, idx2name = Dataset.get_all_name(all_things)
# map the name list to an idx list
idx_list = list(map(lambda names:list(map(lambda name:name2idx[name], names)), all_things))

rdata=Dataset.recipe_data(idx_list)

import SkipGram as SG

SG = importlib.reload(SG)

import wandb

class Reporter:
    def __init__(self, dt):
        self.dt = dt
        self.loss_count = {}
        self.k = 0.0
    
    def report(self):
        if self.k > 0:
            for k in self.loss_count.keys():
                self.loss_count[k] /= self.k
            wandb.log(self.loss_count)
    
    def step(self, loss_dict):
        self.k += 1
        for k, v in loss_dict.items():
            if not (k in self.loss_count):
                self.loss_count[k] = 0.0
            self.loss_count[k] += v
        if self.k >= self.dt:
            self.report()
            self.k = 0
            for k in self.loss_count.keys():
                self.loss_count[k] = 0.0

import os
import json

def train(net, training_data, optimizer, epco, dt):
    wandb.init(config=args, project="Skip-gram Dropout (Recipe)")
    #wandb.config["more"] = f"rule {110}"

    reporter = Reporter(dt)
    net.train()
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    for i in range(epco):
        t = 0
        for u,v,n in training_data:
            t += 1
            optimizer.zero_grad()
            if torch.cuda.is_available():
                u = u.cuda()
                v = v.cuda()
                n = n.cuda()
            
            loss = net(u_pos=u, v_pos=v, v_neg=n, batch_size=len(u))
            
            loss.backward()
            optimizer.step()
            reporter.step({'loss':loss.cpu().item(), 'epco':i+t/len(training_data)}) # report to wandb
        #scheduler.step()
    net.eval()
    net.cpu()
    torch.save(net, os.path.join(wandb.run.dir, 'model.pth'))
    return net

print('training ...')
args = {'epco':1000, 'lr':0.2, 'dim':10, 'batch_size':128}
model = SG.skipgram(vocab_size=len(name2idx), embedding_dim=args['dim'], dropout=True, p=0.5)

if torch.cuda.is_available():
    model.cuda()
opt = optim.SGD(model.parameters(), lr=args['lr'], weight_decay=0)

training_data = data.DataLoader(rdata, batch_size=args['batch_size'], shuffle=True, collate_fn=SG.cf)
train(model, training_data, opt, args['epco'], 250)

print('saving ...')
with open(os.path.join(wandb.run.dir, 'idx2name.json'), 'w') as f:
    json.dump(idx2name, f)

with open(os.path.join(wandb.run.dir, 'name2idx.json'), 'w') as f:
    json.dump(name2idx, f)