import argparse
import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download


def sentence2emb(args, order_texts, feat_name, tokenizer, model, prompt=None):
    embeddings = []
    start, batch_size = 0, 8
    if args.mode in ['simcse']:
        batch_size = 32
    print(f'{feat_name}: ', len(order_texts))
    for start in tqdm(range(0, len(order_texts), batch_size)):
        sentences = order_texts[start: start + batch_size]
        if args.mode == 'simcse':
            outputs = model.encode(
                sentences,
                device=args.device,
                batch_size = batch_size,
                max_length = 512
            )
            embeddings.extend(outputs.cpu())
        else:
            encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
            outputs = model(**encoded_sentences)
            if args.emb_type == 'CLS':
                cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
                embeddings.append(cls_output)
            elif args.emb_type == 'Mean':
                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output[:,1:,:].sum(dim=1) / \
                    encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                embeddings.append(mean_output)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.cache_path, args.dataset_name,
                        args.dataset_name + f'.{feat_name}' + args.emb_type)
    embeddings.tofile(file)


def generate_item_emb(args, tokenizer, model):
    item_pool = []
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
        for line in file:
            item_pool.append(json.loads(line.strip())['metadata'])
    sentence2emb(args, item_pool, args.feat_name, tokenizer, model)


def generate_query_emb(args, tokenizer, model):
    if args.dataset == 'McAuley-Lab/Amazon-C4':
        dataset = load_dataset(args.dataset)['test']
    elif args.dataset == 'esci':
        dataset = load_dataset('csv', data_files=os.path.join(args.cache_path, 'esci/test.csv'))['train']
    else:
        raise NotImplementedError('Dataset not supported')
    sentence2emb(args, dataset['query'], f'q_{args.feat_name}', tokenizer, model)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def load_plm(model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='McAuley-Lab/Amazon-C4', choices=['McAuley-Lab/Amazon-C4', 'esci'])
    parser.add_argument('--cache_path', type=str, default='./cache/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='hyp1231/blair-roberta-base')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--feat_name', type=str, default='blair-base', help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.dataset_name = args.dataset.split('/')[-1]

    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    if 'simcse' in args.plm_name.lower():
        args.mode = 'simcse'
        # Cloning https://github.com/princeton-nlp/SimCSE in the same directory
        from SimCSE.simcse.tool import SimCSE
        plm_tokenizer, plm_model = None, SimCSE(args.plm_name, pooler='cls_before_pooler')
    else:
        args.mode = 'general'
        plm_tokenizer, plm_model = load_plm(args.plm_name)
    if args.mode == 'simcse':
        plm_model.model = plm_model.model.to(device)
    else:
        plm_model = plm_model.to(device)
    print(plm_model)

    # create output dir
    os.makedirs(
        os.path.join(args.cache_path, args.dataset_name),
        exist_ok=True
    )

    generate_item_emb(args, plm_tokenizer, plm_model)
    generate_query_emb(args, plm_tokenizer, plm_model)
