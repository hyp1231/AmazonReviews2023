import argparse
import os
import collections
from tqdm import tqdm


valid_timestamp = 1628643414042
test_timestamp = 1658002729837


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='release_amazon/5core/rating_only')
    parser.add_argument('--output_path', type=str, default='release_amazon/5core/timestamp')
    parser.add_argument('--seq_path', type=str, default='release_amazon/5core/timestamp_w_his')
    parser.add_argument('--zero', action='store_true', help='if true, will process for 0-core, else for 5-core (by default)')
    return parser.parse_args()


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), collections.defaultdict(list)
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
            new_inters[user].append(inter)
    return new_inters


if __name__ == '__main__':
    args = parse_args()

    if args.zero:
        args.input_path = args.input_path.replace('5core', '0core')
        args.output_path = args.output_path.replace('5core', '0core')
        args.seq_path = args.seq_path.replace('5core', '0core')
        print(args)

    all_files = os.listdir(args.input_path)
    for single_file in all_files:
        assert single_file.endswith('.csv')
        prefix = single_file[:-len('.csv')]
        args.file_path = os.path.join(args.input_path, single_file)
        print(args.file_path)

        inters = []
        with open(args.file_path, 'r') as file:
            for line in tqdm(file, 'Loading'):
                user_id, item_id, rating, timestamp = line.strip().split(',')
                timestamp = int(timestamp)
                inters.append((user_id, item_id, rating, timestamp))

        ordered_inters = make_inters_in_order(inters=inters)

        # For direct recommendation
        train_file = open(os.path.join(args.output_path, f'{prefix}.train.csv'), 'w')
        valid_file = open(os.path.join(args.output_path, f'{prefix}.valid.csv'), 'w')
        test_file = open(os.path.join(args.output_path, f'{prefix}.test.csv'), 'w')

        for user in tqdm(ordered_inters, desc='Write direct files'):
            cur_inter = ordered_inters[user]
            for i in range(len(cur_inter)):
                ts = cur_inter[i][-1]
                out_file = None
                if ts >= test_timestamp:
                    out_file = test_file
                elif ts >= valid_timestamp:
                    out_file = valid_file
                else:
                    out_file = train_file
                out_file.write(f'{cur_inter[i][0]},{cur_inter[i][1]},{cur_inter[i][2]},{cur_inter[i][3]}\n')

        for file in [train_file, valid_file, test_file]:
            file.close()

        # For sequential recommendation
        train_file = open(os.path.join(args.seq_path, f'{prefix}.train.csv'), 'w')
        valid_file = open(os.path.join(args.seq_path, f'{prefix}.valid.csv'), 'w')
        test_file = open(os.path.join(args.seq_path, f'{prefix}.test.csv'), 'w')

        for user in tqdm(ordered_inters, desc='Write seq files'):
            cur_inter = ordered_inters[user]
            for i in range(len(cur_inter)):
                ts = cur_inter[i][-1]
                cur_his = ' '.join([_[1] for _ in cur_inter[:i]])
                out_file = None
                if ts >= test_timestamp:
                    out_file = test_file
                elif ts >= valid_timestamp:
                    out_file = valid_file
                else:
                    out_file = train_file
                out_file.write(f'{cur_inter[i][0]},{cur_inter[i][1]},{cur_inter[i][2]},{cur_inter[i][3]},{cur_his}\n')

        for file in [train_file, valid_file, test_file]:
            file.close()
