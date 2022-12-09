import sys
sys.path.append('../')
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from model.cnn_lstm import CNN_LSTM
import torchmetrics

PADDING_VALUE = 0
UNK_VALUE     = 1

def split_train_val_test(df, props=[.8, .1, .1]):
    assert round(sum(props), 2) == 1 and len(props) >= 2
    train_df = df.iloc[:int(len(df)*props[0])                                ,:]
    val_df   = df.iloc[int(len(df)*props[0]):int(len(df)*(props[0]+props[1])),:]
    test_df  = df.iloc[int(len(df)*(props[0]+props[1])):                     ,:]    
    return train_df, val_df, test_df

def generate_vocab_map(df, cutoff=2):
    vocab          = {"": PADDING_VALUE, "UNK": UNK_VALUE}
    reversed_vocab = None
    wordCnt = dict()

    for word in df["tokenized"]:
    # for word in df["disabled_omit"]:
        wordCount = Counter(word)
        for (id, cnt) in wordCount.items():
            if id in wordCnt:
                wordCnt[id] += cnt
            else:
                wordCnt[id] = cnt
    
    value = 2
    for (id, cnt) in wordCnt.items():
        if cnt >= cutoff:
            vocab[id] = value
            value += 1

    return vocab

class HeadlineDataset(Dataset):
    
    def __init__(self, vocab, df, max_length=50):
        self.vocab = vocab
        self.df = df
        self.max_length = max_length
        return 
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        word_list = list()

        for i, word in enumerate(self.df.iloc[index].loc['tokenized']):
        # for i, word in enumerate(self.df.iloc[index].loc['disabled_omit']):
            if i == self.max_length:
                break
            if word not in self.vocab:
                value = self.vocab['UNK']
            else:
                value = self.vocab[word]
            word_list.append(value)       

        tokenized_word_tensor = torch.LongTensor(word_list)     
        curr_label = self.df.iloc[index].loc['toxic_class'] # label 0/1
        return tokenized_word_tensor, curr_label

def collate_fn(batch, padding_value=PADDING_VALUE):

    texts, labels = [], []
    for src, tgt in batch:
        texts.append(src)
        labels.append(int(tgt[0]))

    padded_tokens = pad_sequence(texts, batch_first=True, padding_value=padding_value)
    padded_tokens = torch.LongTensor(padded_tokens)
    y_labels = torch.FloatTensor(labels)

    return padded_tokens, y_labels

def accuracy(out: torch.Tensor, target: torch.Tensor):
    sig_out = torch.sigmoid(out)
    N, C = sig_out.shape
    sig_out[sig_out >= 0.5] = 1
    sig_out[sig_out < 0.5] = 0
    acc = (sig_out == target).sum() / (N * C) * 100
    return acc

def precision(prec, out: torch.Tensor, target: torch.Tensor):
    sig_out = torch.sigmoid(out)
    N, C = sig_out.shape
    sig_out[sig_out >= 0.5] = 1
    sig_out[sig_out < 0.5] = 0
    return prec(sig_out, target)


def recall(rec, out: torch.Tensor, target: torch.Tensor):
    sig_out = torch.sigmoid(out)
    N, C = sig_out.shape
    sig_out[sig_out >= 0.5] = 1
    sig_out[sig_out < 0.5] = 0
    return rec(sig_out, target)

def f1_score(f1, out: torch.Tensor, target: torch.Tensor):
    sig_out = torch.sigmoid(out)
    N, C = sig_out.shape
    sig_out[sig_out >= 0.5] = 1
    sig_out[sig_out < 0.5] = 0
    return f1(sig_out, target)

def eval_step(device: torch.device,
        valid_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        criterion: nn.CrossEntropyLoss,
        steps: int,
        writer: SummaryWriter,
        prec,
        rec,
        f1):
    if steps % 1 == 0:
        model.eval()     
        total_loss = 0.0
        total_acc = 0.0

        total_prec = torch.tensor([0.0,0.0,0.0,0.0]).to(device)
        total_recall = torch.tensor([0.0,0.0,0.0,0.0]).to(device)
        total_f1 = torch.tensor([0.0,0.0,0.0,0.0]).to(device)

        for input, target in tqdm(valid_loader, leave=False):
            input = input.to(device)
            target = target.to(torch.int64)-1
            target = F.one_hot(target, num_classes=4).to(torch.float)
            target = target.to(device)
            out = model(input).squeeze()
            total_loss += criterion(out, target)
            total_acc += accuracy(out, target)
            # print("out shape: ", out.shape)
            # print("target shape: ", target.shape)
            total_prec += precision(prec, out, target)
            total_recall += recall(rec, out, target)
            total_f1 += f1_score(f1, out, target)

        mean_loss = total_loss / len(valid_loader)
        acc = total_acc / len(valid_loader)
        mean_prec = total_prec / len(valid_loader)
        mean_rec = total_recall / len(valid_loader)
        mean_f1_score = total_f1 / len(valid_loader)

        writer.add_scalar('test/cross entropy', mean_loss, global_step=steps)
        writer.add_scalar('test/accuracy', acc, global_step=steps)

        writer.add_scalar('test/precision_class0', mean_prec[0], global_step=steps)
        writer.add_scalar('test/recall_class0', mean_rec[0], global_step=steps)
        writer.add_scalar('test/f1_score_class0', mean_f1_score[0], global_step=steps)
        
        writer.add_scalar('test/precision_class1', mean_prec[1], global_step=steps)
        writer.add_scalar('test/recall_class1', mean_rec[1], global_step=steps)
        writer.add_scalar('test/f1_score_class1', mean_f1_score[1], global_step=steps)

        writer.add_scalar('test/precision_class2', mean_prec[2], global_step=steps)
        writer.add_scalar('test/recall_class2', mean_rec[2], global_step=steps)
        writer.add_scalar('test/f1_score_class2', mean_f1_score[2], global_step=steps)        
        
        writer.add_scalar('test/precision_class3', mean_prec[3], global_step=steps)
        writer.add_scalar('test/recall_class3', mean_rec[3], global_step=steps)
        writer.add_scalar('test/f1_score_class3', mean_f1_score[3], global_step=steps)
        

def train_step(device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        steps: int,
        writer: SummaryWriter,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        scheduler: torch.optim.lr_scheduler._LRScheduler):
    model.train()
    # training
    for input, target in tqdm(train_loader, leave=False):
        input = input.to(device)
        target = target.to(torch.int64)-1
        target = F.one_hot(target, num_classes=4).to(torch.float)
        target = target.to(device)
        out = model(input).squeeze()

        # calculate loss
        loss = criterion(out, target)
        
        # log
        if steps % 100 == 0:
            writer.add_scalar('train/cross entropy', loss,
                                global_step=steps)
            writer.add_scalar('train/learning_rate',
                                scheduler.get_last_lr()[0],
                                global_step=steps)

        # update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        steps += 1
    return steps

def train_val(train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device: torch.device,
        writer: SummaryWriter,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_epoch: int,
        precision,
        recall,
        f1):

    steps: int = 0
    for epoch in tqdm(range(num_epoch)):
        # eval
        eval_step(device,
                valid_loader,
                model,
                criterion,
                steps,
                writer,
                precision,
                recall,
                f1) 
        # training
        steps = train_step(device,
                        train_loader,
                        model,
                        steps,
                        writer,
                        optimizer,
                        criterion,
                        scheduler)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train = pd.read_pickle("../data/train.pkl")
    # train1 = pd.read_pickle("../data/train_disability_omit1.pkl")
    # train2 = pd.read_pickle("../data/train_disability_omit1.pkl")  
    # train3 = pd.read_pickle("../data/train_disability_omit1.pkl")
    # train = pd.concat([train1, train2, train3])
    df = train.sample(frac=1)
    train_df, val_df, test_df = split_train_val_test(df, props=[.8, .1, .1])
    train_vocab = generate_vocab_map(train_df)

    train_dataset = HeadlineDataset(train_vocab, train_df)
    val_dataset   = HeadlineDataset(train_vocab, val_df)
    test_dataset  = HeadlineDataset(train_vocab, test_df)
    
    train_sampler = RandomSampler(train_dataset)
    val_sampler   = RandomSampler(val_dataset)
    test_sampler  = RandomSampler(test_dataset)

    BATCH_SIZE = 2048

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn, drop_last=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = CNN_LSTM(voc_size = max(train_vocab.values())+1)
    model.to(device)
    optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-3,
                weight_decay=0.0,
                betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1e5,
        gamma=0.99)
    
    PATH = "./toxicity_classifier"
    writer = SummaryWriter(PATH)

    prec = torchmetrics.Precision(task='multiclass', average='none' , num_classes=4).to(device)
    rec = torchmetrics.Recall(task='multiclass', average='none' , num_classes=4).to(device)
    f1 = torchmetrics.F1Score(task='multiclass', average='none' , num_classes=4).to(device)

    num_epoch = 20
    train_val(train_loader,
            test_loader,
            model,
            device,
            writer,
            optimizer,
            criterion,
            scheduler,
            num_epoch,
            prec,
            rec,
            f1)

    torch.save(model.state_dict(), PATH + F"/classifier{num_epoch}epoch.pth")
    # torch.save(model.state_dict(), PATH + F"/omit_classifier{num_epoch}epoch.pth")