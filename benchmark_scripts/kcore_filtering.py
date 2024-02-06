import argparse
import collections
import json
import os
from tqdm import tqdm


def load_ratings(file):
    inters = []
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                dp = json.loads(line.strip())
                item, user, rating, time = dp['parent_asin'], dp['user_id'], dp['rating'], dp['sortTimestamp']
                inters.append((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return inters


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), []
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        his_items = set()
        for inter in user_inters:
            user, item, rating, timestamp = inter
            if item in his_items:
                continue
            his_items.add(item)
            new_inters.append(inter)
    return new_inters


def filter_inters(inters, user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []
    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def preprocess_rating(args):
    print('Process rating data: ')
    print(' Dataset: ', args.file_path)

    # load ratings
    rating_inters = load_ratings(args.file_path)

    # Sort and remove repeated reviews
    rating_inters = make_inters_in_order(rating_inters)

    # K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    kcore_rating_inters = filter_inters(rating_inters,
                                        user_k_core_threshold=args.k,
                                        item_k_core_threshold=args.k)

    # return: list of (user_ID, item_ID, rating, timestamp)
    return kcore_rating_inters, rating_inters


def write_rating_only(output_path, prefix, inters, k):
    output_filename = os.path.join(output_path, f'{k}core', 'rating_only', f'{prefix}.csv')
    with open(output_filename, 'w') as file:
        for user, item, rating, ts in tqdm(inters, desc='Write rating only file: '):
            file.write(f'{user},{item},{rating},{ts}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5, help='k-core filtering')
    parser.add_argument('--input_path', type=str, default='AmazonRaw/review_categories/')
    parser.add_argument('--output_path', type=str, default='release_amazon/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    all_files = os.listdir(args.input_path)
    for single_file in all_files:
        assert single_file.endswith('.json')
        prefix = single_file[:-len('.json')]

        args.file_path = os.path.join(args.input_path, single_file)
        kcore_rating_inters, rating_inters = preprocess_rating(args)

        write_rating_only(args.output_path, prefix, rating_inters, k=0)
        write_rating_only(args.output_path, prefix, kcore_rating_inters, k=args.k)
