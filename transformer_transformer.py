import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import collections




# 基于位置的前馈神经网络
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# 加法和规范化
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# 编码器块
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, num_hiddens, num_heads, dropout, norm_shape, ffn_num_input, ffn_num_hiddens, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block{i}",
                EncoderBlock(num_hiddens=num_hiddens,
                             num_heads=num_heads,
                             dropout=dropout,
                             norm_shape=norm_shape,
                             ffn_num_input=ffn_num_input,
                             ffn_num_hiddens=ffn_num_hiddens,
                             use_bias=use_bias))

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

# 解码器块
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, num_hiddens, num_heads, dropout, norm_shape, ffn_num_input, ffn_num_hiddens, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state



# 自定义EncoderDecoder类以确保返回解码器的输出和状态
class MyEncoderDecoder(d2l.EncoderDecoder):
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# 修改后的TransformerDecoder确保返回输出和状态
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block{i}",
                DecoderBlock(num_hiddens=num_hiddens,
                             num_heads=num_heads,
                             dropout=dropout,
                             norm_shape=norm_shape,
                             ffn_num_input=ffn_num_input,
                             ffn_num_hiddens=ffn_num_hiddens,
                             i=i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state  # 确保返回两个值：输出和状态

# 超参数
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
norm_shape = [32]

# 加载数据
def read_data_nmt():
    data_path = r'jpn-eng\jpn.txt'
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 2:
                src, tgt = parts[0].strip(), parts[1].strip()
                if src and tgt:  # 排除空字符的句子对
                    pairs.append((src, tgt))


    return pairs


def preprocess_nmt(sentence):
    """预处理单个句子"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    sentence = sentence.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = []
    for i, char in enumerate(sentence):
        if i > 0 and no_space(char, sentence[i-1]):
            out.append(' ')
        out.append(char)
    return ''.join(out)

def tokenize_nmt(text_pairs):
    source, target = [], []
    for src, tgt in text_pairs:
        # 英语按空格分词
        source.append(src.split())
        # 日语按字符分词
        target.append(list(tgt))
    return source, target


def load_data_nmt(batch_size, num_steps, num_examples=10000):
    """返回翻译数据集的迭代器和词表"""
    raw_pairs = read_data_nmt()[:num_examples]  # 读取前num_examples个例子
    # 预处理每个句子对
    preprocessed_pairs = []
    for src, tgt in raw_pairs:
        pre_src = preprocess_nmt(src)
        pre_tgt = preprocess_nmt(tgt)
        preprocessed_pairs.append((pre_src, pre_tgt))
    # 分词
    source, target = tokenize_nmt(preprocessed_pairs)
    # 构建词表
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 转换为张量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def truncate_pad(line, max_len, padding_token):
    if len(line) > max_len:
        return line[:max_len]
    return line + [padding_token] * (max_len - len(line))

def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    cleaned_lines = []
    for l in lines:
        cleaned = [token for token in l if token != vocab['<pad>']]  # 移除所有填充符（确保逻辑正确）
        if len(cleaned) == 0:  # 如果全为填充符，则跳过
            continue
        cleaned_lines.append(cleaned)
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in cleaned_lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps,num_examples = 100000)

# 使用自定义的MyEncoderDecoder替换原d2l.EncoderDecoder
encoder = TransformerEncoder(
    len(src_vocab), num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout)
net = MyEncoderDecoder(encoder, decoder)  # 使用自定义类



def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# 继续训练
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
engs = ['hello .', "i am a man .", 'i have a room .', 'wish you happiness .']

jpn = ['こんにちは .', 'アイ・アム・ア・マン .', '私には部屋がある .', 'ご多幸を祈ります .']
for eng, jpn in zip(engs, jpn):
    translation, attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, jpn, k=2):.3f}')

