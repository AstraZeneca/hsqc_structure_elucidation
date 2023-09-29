
import os, csv
import argparse

from data import nmrshiftdb2_get_data
from train import train

data_path = './data'
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target', help ='13C or 1H', choices=['13C', '1H'], default='13C', type = str)
arg_parser.add_argument('--message_passing_mode', help ='proposed or baseline', choices=['proposed', 'baseline'], default='proposed', type = str)
arg_parser.add_argument('--readout_mode', help ='proposed or baseline', choices = ['proposed_set2set','proposed_mlp', 'baseline'], default='', type = str)
arg_parser.add_argument('--graph_representation', help ='sparsified or fully_connected', choices=['sparsified', 'fully_connected'], default='sparsified', type = str)
arg_parser.add_argument('--memo', help ='settings', default='', type = str)
arg_parser.add_argument('--fold_seed', default=0, type = int)
arg_parser.add_argument('--node_embedding_dim', type=int)
arg_parser.add_argument('--node_hidden_dim', type=int)
arg_parser.add_argument('--readout_n_hidden_dim', type=int)


args = arg_parser.parse_args()

print('-- CONFIGURATIONS')
print(f'--- current mode: target: {args.target} message_passing_mode: {args.message_passing_mode}, readout_mode: {args.readout_mode}, graph_representation: {args.graph_representation}, fold_seed: {args.fold_seed}')

data_filename = os.path.join(data_path, f'nmrshiftdb2_graph_{args.graph_representation}_{args.target}.npz')

if not os.path.isfile(data_filename):
    print('no data found, preprocessing...')
    nmrshiftdb2_get_data.preprocess(args)
    
model, test_mae, test_rmse, time_per_mol = train(args)

# save result to csv
if not os.path.exists('./result'):
        os.mkdir('./result')

if not os.path.isfile(f'result/result.csv'):
    with open(f'result/result.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['message_passing_mode', 'readout_mode', 'graph_representation', 'target', 'fold_seed', 'test_mae', 'test_rmse', 'time_per_mol'])

with open(f'result/result.csv', 'a', newline='') as f:
    w = csv.writer(f)
    w.writerow([args.message_passing_mode, args.readout_mode, args.graph_representation, args.target, args.fold_seed, test_mae, test_rmse, time_per_mol])

