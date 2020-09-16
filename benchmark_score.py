import argparse
import json
import numpy as np
import re
from tabulate import tabulate
from graphviz import Digraph
from collections import defaultdict

"""Manually collected data from github and google scholars

   Ultimately, could move this to hubconf api so model provides its info
"""
model_meta = {
    'BERT-pytorch': {'stars': 3463, 'citations': 8868,},
    'Background-Matting': {'stars': 2921, 'citations': 2,},
    'Super-SloMo': {'stars': 2262, 'citations': 1,},
    'attention-is-all-you-nee...': {'stars': 3940, 'citations': 11371,},
    'demucs': {'stars': 1540, 'citations': 2,},
    'fastNLP': {'stars': 1430, 'citations': 41,},
    'moco': {'stars': 944, 'citations': 211,},
    'pytorch-CycleGAN-and-pix...': {'stars': 11898, 'citations': 5617,},
    'pytorch-mobilenet-v3': {'stars': 530, 'citations': 287,},
    'pytorch-struct': {'stars': 700, 'citations': 2,},
    'tacotron2': {'stars': 1754, 'citations': 605,},
}

model_cats = {
    'CV': {
        'tasks':{
            'Segmentation': {
                'maskrcnn_benchmark',
            },
            'Classification': {
                'pytorch-mobilenet-v3',
            },
            'Detection': {
                'yolov3',
            },
            'Generation': {
                'pytorch-CycleGAN-and-pix...',
            },
            'Other CV': {
                'Background-Matting',
                'Super-SloMo',
            },
        },
    },
    'NLP': {
        'tasks':  {
            'Translation': {
                'attention-is-all-you-nee...',
            },
            'Language Modeling': {
                'BERT-Pytorch',
            },
            'Other NLP': {
                'fastNLP',
            },

        },
    },
    'Other Categories': {
        'tasks': {
            'Other Tasks': {
                'demucs',
                'Learning To Paint',
                'pytorch-struct',
                'moco',
            },
        },
    },
}

class ScoreNode(object):
    def __init__(self, name, weight=None):
        self.name = name
        if weight is not None:
            self.__weight = weight
        self.specified_weight = weight is not None
        self.__specified_children = []
        self.__unspecified_children = []

    def add_child(self, child):
        """Add a new child weight.
        If weight is provided, it must be less than 1 - sum(w_other) where
        w_other are all the other weights that were provided to sibling nodes.

        If weight is not provided, it will be an even fraction of the remainder
        of 1 - sum(w_specified).
        """
        specified_weights = sum([c.weight for c in self.__specified_children])
        if child.specified_weight:
            specified_weights += child.weight
            assert specified_weights <= 1.0, "Attempted to add child with specified weight greater than remaining unspecified weight"
            self.__specified_children.append(child)
        else:
            remainder = 1.0 - specified_weights
            assert remainder > 0, "Manual weight specification leaves no weight remainder for unspecified children"
            self.__unspecified_children.append(child)
            weight = remainder / len(self.__unspecified_children)
            for c in self.__unspecified_children:
                c.weight = weight
        
        return child

    @property
    def children(self):
        return self.__unspecified_children + self.__specified_children 

    @property
    def weight(self):
        return self.__weight
 
    @weight.setter
    def weight(self, weight):
        assert not self.specified_weight
        self.__weight = weight

        
def draw_graph(filename, root):
    g = Digraph('TorchBench Score Weights', filename=filename)
    visit = [root]
    while visit:
        curr = visit.pop()
        for child in curr.children: 
            g.edge(curr.name, child.name, label=f'{child.weight:.2f}')
            visit.append(child)

    g.view()

def split_name(name):
    # e.g. test_train[BERT-pytorch-cpu-jit]
    return re.search('(.+)\[(.+)-(.+)-(.+)\]', name).groups()


def print_as_table(data, title, column_keys):
    table = []
    for model in data:
        row = [model]
        for cfg in column_keys:
            row.append(data[model].get(cfg, None))
        table.append(row)
    print(tabulate(table, headers=[title] + column_keys))


def compute_model_weights(stars_weight=0.5, citations_weight=0.5):
    stars_total = 0
    citations_total = 0
    for model in model_meta:
        meta = model_meta[model]
        stars_total += meta['stars']
        citations_total += meta['citations']

    for model in model_meta:
        meta = model_meta[model]
        meta['stars_weight'] = meta['stars'] / float(stars_total)
        meta['citations_weight'] = meta['citations'] / float(citations_total)
        meta['final_weight'] = meta['stars_weight'] * stars_weight +  \
                               meta['citations_weight'] * citations_weight

    print_as_table(model_meta, 'Weight Factors', ['stars_weight', 'citations_weight', 'final_weight'])

    return meta

def extract_data(data):
    benchmarks = data['benchmarks']
    train = defaultdict(dict)
    eval = defaultdict(dict)
    configs = set()
    for b in benchmarks:
        test, model, device, compiler = split_name(b['name'])
        configs.add((device, compiler)) 
        if test == 'test_train':
            train[model][(device, compiler)] = b['stats']['ops']
        elif test == 'test_eval':
            eval[model][(device, compiler)] = b['stats']['ops']

    columns = list(configs)
    print_as_table(train, 'test_train op/sec', columns)
    print()
    print_as_table(eval, 'test_eval op/sec', columns)
    print()

    
    return train, eval


def compute_score(data):
    train, eval = extract_data(data)

    model_weights = compute_model_weights()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json")
    parser.add_argument("--graph", help="filename to write graph visualization")
    args = parser.parse_args()
    with open(args.json) as f:
        data = json.load(f)
        if args.graph is not None:
            root = ScoreNode('TorchBench v0.1')
            synth = root.add_child(ScoreNode('Synthetic Benchmarks', 0.2))
            synth.add_child(ScoreNode('FastRNNs'))
            synth.add_child(ScoreNode('Op uBench'))
            models = root.add_child(ScoreNode('Open Source Models', 0.8))
            for category in model_cats:
                cat = models.add_child(ScoreNode(category))
                for task in model_cats[category]['tasks']:
                    t = cat.add_child(ScoreNode(task))
                    for model in model_cats[category]['tasks'][task]:
                        m = t.add_child(ScoreNode(model))
            
            draw_graph(args.graph, root)
        print(compute_score(data))