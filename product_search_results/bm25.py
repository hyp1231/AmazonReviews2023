import argparse
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from six import iteritems
from six.moves import xrange
import math
from recbole.evaluator.metrics import NDCG
from tqdm import tqdm
import torch
import numpy as np
import json
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import hf_hub_download


class BM25_Model(object):
    def __init__(self, args, corpus):
        self.eps = args.e
        self.k1 = args.k1
        self.b = args.b

        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()
        self.average_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def initialize(self):
        print('Initialize BM25 Model')
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
        print('Finish initialization!!')

    def get_score(self, document, index):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.eps * self.average_idf
            score += (idf * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * self.corpus_size / self.avgdl)))
        return score

    def get_scores(self, document):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index)
            scores.append(score)
        return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='McAuley-Lab/Amazon-C4', help='McAuley-Lab/Amazon-C4 or esci')
    parser.add_argument('--cache_path', type=str, default='./cache/')
    parser.add_argument('-k', type=int, default=100, help='top k')
    parser.add_argument('-k1', type=float, default=1.5)
    parser.add_argument('-b', type=float, default=0.75)
    parser.add_argument('-e', type=float, default=0.25)
    return parser.parse_args()


def sent_tokenize(sentence):
    sentence = word_tokenize(sentence)
    sentence = [w for w in sentence if w not in stopwords]
    return sentence


if __name__ == '__main__':
    args = parse_args()
    stopwords = set(stopwords.words('english'))

    """
    Load item text
    """
    all_text = []
    item2id = {}
    if args.dataset == 'McAuley-Lab/Amazon-C4':
        filepath = hf_hub_download(
            repo_id=args.dataset,
            filename='sampled_item_metadata_1M.jsonl',
            repo_type='dataset'
        )
    elif args.dataset == 'esci':
        filepath = os.path.join(args.cache_path, 'esci/sampled_item_metadata_esci.jsonl')
    else:
        raise NotImplementedError('Dataset not supported')
    with open(filepath, 'r') as file:
        for idx, line in tqdm(enumerate(file), desc='Loading item text: '):
            data = json.loads(line.strip())
            sentence = sent_tokenize(data['metadata'])
            all_text.append(sentence)
            item2id[data['item_id']] = idx

    """
    Load asin2category
    """
    filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename='asin2category.json',
        repo_type='dataset'
    )
    with open(filepath, 'r') as file:
        asin2category = json.loads(file.read())

    """
    Load queries
    """
    cat2results = defaultdict(list)

    all_queries = []
    if args.dataset == 'McAuley-Lab/Amazon-C4':
        query_dataset = load_dataset(args.dataset)['test']
    elif args.dataset == 'esci':
        query_dataset = load_dataset('csv', data_files=os.path.join(args.cache_path, 'esci/test.csv'))['train']
    else:
        raise NotImplementedError('Dataset not supported')
    for query, item_id in zip(query_dataset['query'], query_dataset['item_id']):
        sentence = sent_tokenize(query)
        all_queries.append([sentence, item_id])

    """
    BM25 Model
    """
    model = BM25_Model(args, all_text)
    # topk
    metric = NDCG({
        'metric_decimal_place': 6,
        'topk': args.k
    })
    results = []
    for q, target in tqdm(all_queries):
        scores = torch.FloatTensor(model.get_scores(q)).unsqueeze(0)
        topk_scores, topk_indices = torch.topk(scores, args.k, dim=-1)

        target_cat = asin2category[target]

        batch_target = torch.full((1, args.k), item2id[target], dtype=torch.long)
        pos_index = (batch_target == topk_indices).cpu()
        pos_len = torch.ones((1), dtype=torch.long).cpu().numpy()
        ndcg = metric.metric_info(pos_index, pos_len)
        ndcg = ndcg[:,-1]
        results.append(ndcg)
        cat2results[target_cat].append(ndcg)

    results = np.concatenate(results)
    print(args)
    print(f'Overall NDCG@{args.k}: {results.mean()}')

    for d in cat2results:
        print(f'  {d}: {np.mean(cat2results[d])}')
