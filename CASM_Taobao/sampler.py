import numpy as np
from multiprocessing import Process, Queue
import matplotlib as plt


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, Beh, Beh_w, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        recency_alpha = 0.5
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        recency = np.zeros([maxlen], dtype=np.float32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            recency[idx] = recency_alpha**(maxlen-idx) 
            #print('recency[idx]...', recency[idx])
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        #print(abc)
        seq_cxt = list()
        pos_cxt = list()
        pos_weight = list()
        neg_weight = list()

        for i in seq :
            seq_cxt.append(Beh[(user,i)])

        for i in pos :
            pos_cxt.append(Beh[(user,i)])

        for i in pos :
            pos_weight.append(Beh_w[(user,i)])
            neg_weight.append(1.0)

        seq_cxt = np.asarray(seq_cxt)  
        pos_cxt = np.asarray(pos_cxt)    
        pos_weight = np.asarray(pos_weight)  


        return (user, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight , recency)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, Beh, Beh_w, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      Beh,
                                                      Beh_w,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
