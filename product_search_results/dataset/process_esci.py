import os
import re
import html
import json
from collections import defaultdict
from tqdm import tqdm
import random
import argparse
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neg', type=int, default=50, help='the number of negative samples')
    parser.add_argument('--output_path', type=str, default='./cache/')
    parser.add_argument('--n_workers', type=int, default=16)
    return parser.parse_args()


def get_asin2category():
    filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename='asin2category.json',
        repo_type='dataset'
    )
    with open(filepath, 'r') as file:
        asin2category = json.loads(file.read())
    return asin2category


def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    return cleaned_text


def clean_metadata(example):
    meta_text = ''
    features_needed = ['title', 'description']
    for feature in features_needed:
        if feature in example and example[feature] is not None:
            meta_text += clean_text(example[feature]) + ' '
    example['cleaned_metadata'] = meta_text.replace('\t', ' ')
    return example


if __name__ == '__main__':
    args = parse_args()

    # Collect potential negative samples
    asin2category = get_asin2category()
    category2item = defaultdict(list)
    for asin, cat in tqdm(asin2category.items()):
        category2item[cat].append(asin)

    # Create output directory
    os.makedirs(
        os.path.join(args.output_path, 'esci/'),
        exist_ok=True
    )

    # Filter ESCI dataset
    query_data = {
        'qid': [],
        'query': [],
        'item_id': []
    }
    candidate_item = set()
    category2items = defaultdict(set)

    raw_dataset = load_dataset("tasksource/esci")

    qid = 0
    for line in tqdm(raw_dataset['test']):
        item_id = line['product_id']

        if item_id not in asin2category:
            continue
        cat = asin2category[item_id]
        if 'Unknown' in cat:
            continue
        if line['product_locale'] != 'us':
            continue
        if line['esci_label'] != 'Exact':
            continue
        if line['small_version'] != 1:
            continue

        candidate_item.add(item_id)
        category2items[cat].add(item_id)
        neg_items = random.sample(category2item[cat], args.n_neg)
        for neg_item in neg_items:
            candidate_item.add(neg_item)
            category2items[cat].add(neg_item)
        query = line['query'].strip().replace('\t', ' ')

        query_data['qid'].append(qid)
        query_data['query'].append(query)
        query_data['item_id'].append(item_id)

        qid += 1

    query_data_hf = Dataset.from_dict(query_data)
    query_file_path = os.path.join(args.output_path, 'esci/test.csv')
    query_data_hf.to_csv(query_file_path)

    # Write item metadata
    metadata_file_path = os.path.join(args.output_path, 'esci/sampled_item_metadata_esci.jsonl')
    with open(metadata_file_path, 'w') as file:
        print(f'Writing metadata to {metadata_file_path}')
        for category in category2items:
            print(f'Processing category: {category}')
            meta_dataset = load_dataset(
                'McAuley-Lab/Amazon-Reviews-2023',
                f'raw_meta_{category.replace(" ", "_")}',
                split='full',
                trust_remote_code=True
            )
            meta_dataset = meta_dataset.map(
                clean_metadata,
                num_proc=args.n_workers
            )
            for item_id, metadata in zip(meta_dataset['parent_asin'], meta_dataset['cleaned_metadata']):
                if item_id in category2items[category]:
                    data = {
                        'item': item_id,
                        'category': category,
                        'metadata': metadata
                    }
                    file.write(json.dumps(data) + '\n')
