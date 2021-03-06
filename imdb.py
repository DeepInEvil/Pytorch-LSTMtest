from torchtext import data
from torchtext import datasets
from torch.autograd import Variable
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tqdm
from CustomLSTMCell import LSTMCell

torch.manual_seed(666)
max_sent_len = 200
n_epoch = 50
cuda = True
run_cell = True


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    pred = [1 if p > 0.5 else 0 for p in pred]
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


# Normal LSTM
class LSTM(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)
        self.drop = nn.Dropout(0.4)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1)
        self.linearOut = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.drop(x)
        h0, c0 = self.init_hidden(inputs.size(1))
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        x = hn.squeeze()
        y = self.linearOut(x)
        return F.sigmoid(y).squeeze()

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda(), Variable(
            torch.zeros(1, batch_size, self.hidden_dim)).cuda()


class LSTMC(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMC, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
        self.drop = nn.Dropout(0.4)
        self.lstm = LSTMCell(embedding_dim, hidden_dim, batch_first=False)
        self.linearOut = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.drop(x)
        h0, c0 = self.init_hidden(inputs.size(1))
        lstm_out, hn = self.lstm(x, (h0, c0))
        if isinstance(hn, tuple):
            y = self.linearOut(hn[0])
        else:
            y = self.linearOut(hn)
        return F.sigmoid(y).squeeze()

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(), Variable(
            torch.zeros(batch_size, self.hidden_dim)).cuda()


# set up fields
TEXT = data.Field(lower=True, pad_token=0, fix_length=200)
LABEL = data.Field(sequential=False)

train, test = datasets.IMDB.splits(TEXT, LABEL)
# build the vocabulary
TEXT.build_vocab(train) #get rid of vectors=GloVe(name='6B', dim=300)
LABEL.build_vocab(train)
label_2_idx = {1: 0, 2: 1}
batch_size = 256
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_sizes=(batch_size, batch_size),
    shuffle=True, device=None, repeat=False)

print (len(TEXT.vocab))
# Initialize the lstm model
if run_cell:
    model = LSTMC(50, 64, len(TEXT.vocab))
else:
    model = LSTM(50, 64, len(TEXT.vocab))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
if cuda:
    model.cuda()


# Train the model
def train(data):
    model.train()
    for i in range(n_epoch):
        train_acc = 0.0
        print ("Epoch: " + str(i))
        tr_itr = enumerate(data)
        for iter, mb in tr_itr:
            sent, label = mb.text, mb.label
            label = Variable(torch.from_numpy(np.array([label_2_idx[l.data[0]] for l in label])))
            if cuda:
                sent, label = sent.cuda(), label.cuda()
            out = model(sent)
            loss = F.binary_cross_entropy(out, label.type(torch.cuda.FloatTensor))
            loss.backward()
            optimizer.step()
            acc = get_accuracy(label.cpu().data.numpy(), out.cpu().data.numpy())
            train_acc += acc
        print ("Accuracy for this epoch:")
        print (train_acc/len(data))


# Evaluation
def test(data):
    print ("Testing.............................")
    model.eval()
    test_acc = 0.0
    test_iter = enumerate(data)
    for iter, mb in test_iter:
        sent, label = mb.text, mb.label
        label = Variable(torch.from_numpy(np.array([label_2_idx[l.data[0]] for l in label])))
        if cuda:
            sent, label = sent.cuda(), label.cuda()
        out = model(sent)
        acc = get_accuracy(label.cpu().data.numpy(), out.cpu().data.numpy())
        test_acc += acc
    print (test_acc/len(data))


if __name__ == '__main__':
    train(train_iter)
    test(test_iter)