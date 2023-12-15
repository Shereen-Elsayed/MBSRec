import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
tf.compat.v1.disable_eager_execution()


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0003, type=float) # it was 0.0003 from 0.9 - 1.5 then from 1.6 till 2.0 0.00009
parser.add_argument('--maxlen', default=70, type=int)
parser.add_argument('--hidden_units', default=70, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=1001, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.4, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--projection_size', default=8, type=int)

'''
 Best params
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0007, type=float)
parser.add_argument('--maxlen', default=70, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=801, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--projection_size', default=8, type=int)
'''

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

print('here')
dataset = data_partition_yelp(args.dataset)
print('here')
[user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum] = dataset
print(usernum,'-',itemnum)
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print ('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

sampler = WarpSampler(user_train, Beh, Beh_w, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.compat.v1.initialize_all_variables())

T = 0.0
t0 = time.time()


for epoch in range(1, args.num_epochs + 1):
    total_loss = 0
    #for step in tqdm(range(int(num_batch)), total=int(num_batch), ncols=70, leave=False, unit='b'):
    for step in  range (0, int(num_batch) ): 
        u, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, recency = sampler.next_batch()
        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                 model.is_training: True, model.seq_cxt:seq_cxt, model.pos_cxt:pos_cxt, model.pos_weight:pos_weight, 
                                 model.neg_weight:neg_weight, model.recency:recency})
        total_loss = total_loss+ loss
    print('loss in epoch...', epoch, ' is  ', total_loss/int(num_batch) )
    if epoch % 10 == 0:
        t1 = time.time() - t0
        T += t1
        print ('Evaluating')
        t_valid = evaluate_valid(model, dataset, args, sess, Beh)

        print ('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f)' % (
        epoch, T, t_valid[0], t_valid[1]))


sampler.close()
f.close()

print("Done")
