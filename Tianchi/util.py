import sys
import copy
import random
import numpy as np
import pickle
import ast

from collections import defaultdict

tstInt = None
with open('data/Tianchi_tst_int', 'rb') as fs:
    tstInt = np.array(pickle.load(fs))

tstStat = (tstInt!=None)
tstUsrs = np.reshape(np.argwhere(tstStat!=False), [-1])
tstUsrs = tstUsrs + 1
print(len(tstUsrs))

def data_partition_tmall(fname):
    usernum = 0
    itemnum = 0
    interactions = 0
    User = defaultdict(list)
    user_train = {}
    user_last_indx = {}
    user_valid = {}
    user_test = {}
    Beh = {}
    Beh_w = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    next(f)
    for line in f:
        interactions+=1
        u, i, b = line.rstrip().split(' ')
        #print( 'data type of user ....', u, '  ',i,'    ', b)
        #print(abc)
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if b == 'buy':
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx
            Beh[(u,i)] = [1,0,0,0]
            Beh_w[(u,i)] = 0.7

        elif b == 'cart':
            Beh[(u,i)] = [0,0,1,0]
            Beh_w[(u,i)] = 0.1

        elif b == 'fav':
            Beh[(u,i)] = [0,0,0,1]
            Beh_w[(u,i)] = 0.1
            
        elif b == 'pv':
            Beh[(u,i)] = [0,1,0,0]
            Beh_w[(u,i)] = 0.1
            
        User[u].append(i)
    print('Total Number of interactions is .....', interactions)
    for user in User:
        Beh[(user,0)] = [0,0,0,0]
        Beh_w[(user,0)] = 0
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            last_item_indx = user_last_indx[user]
            last_item = User[user][last_item_indx]
            items_list = User[user]
            del items_list[last_item_indx]

            #user_train[user] = items_list
            #user_train[user] = [value for value in items_list if value != last_item]
            truncated_item_list = items_list[:last_item_indx]       
            user_train[user] = [value for value in truncated_item_list if value != last_item]
            user_valid[user] = []
            user_valid[user].append(last_item)
            user_test[user] = []
            user_test[user].append(last_item)
    return [user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum]

def data_partition_yelp(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_last_indx = {}
    user_valid = {}
    user_test = {}
    Beh = {}
    Beh_w = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, b = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if b == 'pos':
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx
            Beh[(u,i)] = [1,0,0,0]
            Beh_w[(u,i)] = 1.0

        elif b == 'neutral':
            Beh[(u,i)] = [0,1,0,0]
            Beh_w[(u,i)] = 0.0

        elif b == 'neg':
            Beh[(u,i)] = [0,0,1,0]
            Beh_w[(u,i)] = 0.0

        elif b == 'tip':
            Beh[(u,i)] = [0,0,0,1]
            Beh_w[(u,i)] = 0.0
        User[u].append(i)

    for user in User:
        Beh[(user,0)] = [0,0,0,0]
        Beh_w[(user,0)] = 0
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            last_item_indx = user_last_indx[user]
            last_item = User[user][last_item_indx]
            items_list = User[user]
            del items_list[last_item_indx]

            #user_train[user] = items_list
            truncated_item_list = items_list[:last_item_indx]       
            user_train[user] = [value for value in truncated_item_list if value != last_item]
            #user_train[user] = [value for value in items_list if value != last_item]
            user_valid[user] = []
            user_valid[user].append(last_item)
            user_test[user] = []
            user_test[user].append(last_item)
    return [user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum]

def data_partition2(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-1]
            user_valid[user] = []
            user_valid[user].append(User[user][-1])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = tstUsrs #random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)



        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess, Beh, epoch):
    [train, valid, test, Beh, Beh_w, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = tstUsrs #random.sample(range(1, usernum + 1), 10000)
        print(len(users))
    else:
        users = range(1, usernum + 1)
    for u in users:
        seq_cxt = list()

        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        for i in seq :
            seq_cxt.append(Beh[(u,i)])
        seq_cxt = np.asarray(seq_cxt) 


        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        
        testitemscxt = list()
        testitemscxt.append(Beh[(u,valid[u][0])])


        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            testitemscxt.append(Beh[(u,valid[u][0])])



        predictions = -model.predict(sess, [u], [seq], item_idx, [seq_cxt],testitemscxt)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            #if  epoch<200:
                     #with open('/home/elsayed/MultiBehaviour/Tianchi_CUT_Results/Tianchi_CUT_Without_Behaviours_'+str(epoch)+'.txt', 'a') as f:
                     #user  HR     NDCG
                     #f.write(str(u)+';'+str(1)+';'+str(1/np.log2(rank + 2))+'\n')
        if valid_user % 100 == 0:
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
