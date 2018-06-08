from torchtext import data
from torchtext import datasets
from torch.autograd import Variable
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tqdm

max_sent_len = 200
n_epoch = 20
cuda = True


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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linearOut = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        h0, c0 = self.init_hidden(inputs.size(1))
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        x = hn.squeeze()
        y = self.linearOut(x)
        return F.sigmoid(y).squeeze()

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda(), Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda()


# set up fields
TEXT = data.Field(lower=True, pad_token=0, fix_length=200)
LABEL = data.Field(sequential=False)

train, test = datasets.IMDB.splits(TEXT, LABEL)
# build the vocabulary
TEXT.build_vocab(train) #get rid of vectors=GloVe(name='6B', dim=300)
LABEL.build_vocab(train)
label_2_idx = {1: 0, 2: 1}
batch_size = 1024
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_sizes=(batch_size, batch_size),
    shuffle=True, device=None)

print (len(TEXT.vocab))
# Initialize the lstm model
model = LSTM(50, 32, len(TEXT.vocab))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
if cuda:
    model.cuda()


# Train the model
def train(data):
    model.train()
    train_acc = 0.0
    for i in range(n_epoch):
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
            if (iter // 100) == 0:
                print (acc)
            print ("finished epoch" + str(iter))
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