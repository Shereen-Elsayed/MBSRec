from modules import *
#import tensorflow_addons as tfa


#获取行为嵌入，物品嵌入，位置嵌入,上下文嵌入
# 全连接层降维
#
class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.u = tf.compat.v1.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.compat.v1.placeholder(tf.int32, shape=(None, args.maxlen))
        self.seq_cxt = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen, 4))
        self.pos_cxt = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen, 4))
        self.pos_weight = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.neg_weight = tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        self.recency= tf.compat.v1.placeholder(tf.float32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.input_seq, 0)), -1)

        
            # sequence embedding, item embedding table
        #输入input_seq获取交互的嵌入，并返回嵌入表item_emb_table（词汇表）
        self.seq, item_emb_table = embedding(self.input_seq,
                                             vocab_size=itemnum + 1,
                                             num_units=args.hidden_units,  #生成嵌入量的纬度
                                             zero_pad=True,
                                             scale=True,   #对生成的嵌入向量进行某种缩放操作
                                             l2_reg=args.l2_emb, #L2 正则化的系数等参数
                                             scope="input_embeddings",
                                             with_t=True,
                                             reuse=reuse
                                             )

        #输入seq_cxt（行为类型的独热编码）,经过一个全连接层映射到hidden_units的纬度
        #kernel_initializer:全连接层权重矩阵（也就是连接输入和输出的参数矩阵）的初始化方式，这里采用的是均值为 0、标准差为 0.01 的正态分布来初始化权重
        self.seq_cxt_emb = tf.compat.v1.layers.dense(inputs=self.seq_cxt , units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="cxt_emb")

        #self.seq = tf.concat([self.seq , self.seq_cxt_emb], -1)
        #cxt
        #全连接层，将行为嵌入映射到hidden_units空间
        self.seq = tf.compat.v1.layers.dense(inputs=self.seq, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
 

        # Positional Encoding  位置编码
        t, pos_emb_table = embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
            vocab_size=args.maxlen,   #位置编码最多能处理长度为 args.maxlen 的序列
            num_units=args.hidden_units,  #位置嵌入向量的维度大小
            zero_pad=False,
            scale=False,
            l2_reg=args.l2_emb,
            scope="positional",
            reuse=reuse,
            with_t=True
        )
        self.seq += t   #将生成的位置嵌入 t 累加到 self.seq 上

        
#神经网络中的正则化技术，通过在训练过程中随机地将神经元的输出设置为 0，以防止过拟合
        # Dropout
        self.seq = tf.compat.v1.layers.dropout(self.seq,
                                     rate=args.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))
        self.seq *= mask

     # Build blocks
        for i in range(args.num_blocks):
            with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                
                
                # Pack-attention
                self.seq = multihead_attention(queries=self.seq,
                                               keys=self.seq,
                                               num_units=args.hidden_units,
                                               num_heads=args.num_heads,
                                               dropout_rate=args.dropout_rate,
                                               is_training=self.is_training,
                                               causality=True,
                                               reuse=False,
                                               res= True,
                                               scope="self_attention")
        
                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                       dropout_rate=args.dropout_rate, is_training=self.is_training)
                                       
                
                self.seq *= mask

        self.seq = normalize(self.seq)

    #将他们的形状重塑为一维的
    #例如，如果tf.shape(self.input_seq)[0]为 32（表示一次处理 32 个样本），args.maxlen为 10（表示序列最大长度为 10），
    #那么[tf.shape(self.input_seq)[0] * args.maxlen]就是 320，这些张量将被重塑为长度为 320 的一维张量。
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_weight = tf.reshape(self.pos_weight, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg_weight = tf.reshape(self.neg_weight, [tf.shape(self.input_seq)[0] * args.maxlen])
        recency = tf.reshape(self.recency, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])

#目标样本
        trgt_cxt = tf.reshape(self.pos_cxt, [tf.shape(self.input_seq)[0] * args.maxlen, 4])
        trgt_cxt_emb = tf.compat.v1.layers.dense(inputs=trgt_cxt , units=args.hidden_units,activation=None, reuse=True, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="cxt_emb")


#分别根据pos和neg从item_emb_table（物品嵌入表）中查找对应的嵌入向量，得到正样本和负样本的嵌入表示
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)

#对正样本和负样本的嵌入向量进行全连接层转换，将其映射到args.hidden_units维空间。
        pos_emb = tf.compat.v1.layers.dense(inputs=pos_emb, reuse=True, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
        neg_emb = tf.compat.v1.layers.dense(inputs=neg_emb, reuse=True, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
    
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.compat.v1.placeholder(tf.int32, shape=(100))
        self.test_item_cxt = tf.compat.v1.placeholder(tf.float32, shape=(100, 4))

#测试   物品上下文嵌入
        test_item_cxt_emb  = tf.compat.v1.layers.dense(inputs=self.test_item_cxt  , units=args.hidden_units,activation=None, reuse=True, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="cxt_emb")

#测试  物品嵌入
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        test_item_emb = tf.compat.v1.layers.dense(inputs=test_item_emb, reuse=True, units=args.hidden_units,activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.01) , name="feat_emb")
        

#将test_item_emb与seq_emb做矩阵乘法
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 100])
#取最后一个时间步（[:, -1, :]）的得分，作为最终的测试得分
        self.test_logits = self.test_logits[:, -1, :]
    
        
    # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

    # ignore padding items (0)
        istarget = tf.reshape(tf.compat.v1.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.compat.v1.log(tf.sigmoid(self.pos_logits) + 1e-24)*pos_weight * istarget -
            tf.compat.v1.log(1 - tf.sigmoid(self.neg_logits) + 1e-24)*neg_weight * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.compat.v1.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.compat.v1.summary.scalar('auc', self.auc)
            self.global_step = tf.compat.v1.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            #self.optimizer = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=0.00001)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            #self.train_op = self.optimizer.minimize(self.loss, var_list= self.optimizer.get_weights())
        else:
            tf.compat.v1.summary.scalar('test_auc', self.auc)

        self.merged = tf.compat.v1.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, seq_cxt, test_item_cxt):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False, self.seq_cxt:seq_cxt, self.test_item_cxt:test_item_cxt})
