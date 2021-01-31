import csv
import numpy as np
import torch
import torch.utils.data as data
import random

path = './data/allr_recipes.txt'

def get_recipe(file_path):
    raw = []
    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            raw.append(row[1:])
    return raw


def get_all_name(recipes):
    name_list = set()
    for line in recipes:
        for name in line:
            name_list.add(name)
    name_list = sorted(name_list)
    return dict(zip(name_list, range(len(name_list)))), dict(zip(range(len(name_list)), name_list))


class recipe_data(data.Dataset):

    def __init__(self, raw, neg_size=2):
        super(recipe_data, self).__init__()
        self.max_idx = max(map(max, raw))
        self.raw = self.remove1(raw)
        self.length = len(self.raw)
        self.neg_size = neg_size
        
    def remove1(self, raw):
        rec_list = []
        for line in raw:
            if len(line) > 1:
                rec_list.append(line)
        return rec_list
    
    def context(self, idxs, center):
        left = idxs[:center]
        right = idxs[center+1:]
        return left + right
    
    def neg_sample(self, neg_size, n):
        negative = []
        for i in range(n):
            neg = []
            idxs = [self.raw[random.randint(0, len(self.raw)-1)] for i in range(neg_size)]
            for line in idxs:
                neg.append(line[random.randint(0, len(line)-1)])
            negative.append(neg)
        return negative
    
    def __getitem__(self, index):
        '''
        return:
            center word
            context words
            negative sampled words
        '''
        # center word (u)
        center_id = random.randint(0, len(self.raw[index])-1)
        center = self.raw[index][center_id]
        center = [center] * (len(self.raw[index]) - 1)
        # context words (v)
        context = self.context(self.raw[index], center_id)
        # negative samples
        neg = self.neg_sample(self.neg_size, len(center))
        
        return torch.LongTensor(center), torch.LongTensor(context), torch.LongTensor(neg)
    
    def __len__(self):
        return self.length