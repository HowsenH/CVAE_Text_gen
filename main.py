import dataset as mydataset
from typing import List, Tuple
from model import CVAE
from train import train
import argparse
import torch as T
import tqdm
import utils as ut

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--label_cats', type=int, default=10)
parser.add_argument('--max_words', type=int, default=20000)
parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--z_dim', type=int, default=512)
parser.add_argument('--n_z', type=int, default=100)
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--save_iter', type=int, default=5000)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--embedding', type=str, default='glove.6B.300d')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--rec_coef', type=float, default=7)
parser.add_argument('--n_highway_layers', type=int, default=2)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--dataset', type=str, default='pitchfork')

opt = parser.parse_args()

raw_data : List[Tuple[List[str], str, float]] = mydataset.tokenize_pitchfork_data()
raw_data = list(filter(lambda x : len(x[0]) <= opt.max_len, raw_data))
print("Load %d datapoints." % len(raw_data))

train_data, val_data, vocab = mydataset.make_dataset(raw_data, opt)
train_iter, val_iter = mydataset.make_iterator((train_data, val_data), opt)
device = T.device(opt.device)

layout = [
    ('model={:s}',  'cvae'),
    ('z={:02d}',  opt.z_dim),
    ('run={:04d}', opt.max_iter)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
writer = ut.prepare_writer(model_name, overwrite_existing=True)

model = CVAE(opt).to(device)
model.embedding.weight.data.copy_(vocab.vectors).to(device)

train_gen = mydataset.make_loader(train_iter, opt)
val_gen = mydataset.make_loader(val_iter, opt)
if opt.mode == 'train':
    train(model=model,
          data_loader = train_gen,
          tqdm=tqdm.tqdm,
          device=device,
          writer=writer,
          iter_max=opt.max_iter,
          iter_save=opt.save_iter)
elif opt.mode == 'test':
    ut.load_model_by_name(model, global_step=opt.max_iter)
    generated_list = model.generate_samples(opt, vocab)
    for sentence in generated_list:
        print(sentence)
