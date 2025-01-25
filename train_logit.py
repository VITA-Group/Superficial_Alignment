import torch
import torch.nn as nn
import wandb
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
import logging
from datetime import datetime
import copy
import glob
import re
from tqdm import tqdm, trange



def get_topk_softmax(x, topk):
    x = torch.softmax(x, dim=-1)
    top_values, top_indices = torch.topk(x, topk, dim=1)
    result = torch.zeros_like(x).to(x.device)
    result.scatter_(1, top_indices, top_values)
    return result

def get_topk_logsoftmax(x, topk):
    x = torch.log_softmax(x, dim=-1)
    top_values, top_indices = torch.topk(x, topk, dim=1)
    result = torch.zeros_like(x).to(x.device)
    result.scatter_(1, top_indices, top_values)
    return result

def get_topk(x, topk):
    top_values, top_indices = torch.topk(x, topk, dim=1)
    result = torch.zeros_like(x)
    result.scatter_(1, top_indices, top_values)
    return result

def get_topk_one(x, topk):
    top_values, top_indices = torch.topk(x, topk, dim=1)
    result = torch.zeros_like(x)
    result.scatter_(1, top_indices, 1)
    return result


def merge_prefix(prefix):
    new_prefix = []
    start=None
    end=None
    for p in prefix:
        p=p.split("_")
        if start is None:
            start=p[0]
            end=p[1]
        else:
            if end==p[0]:
                end=p[1]
            else:
                new_prefix.append(f"{start}_{end}")
                start=p[0]
                end=p[1]
    new_prefix.append(f"{start}_{end}")
    return new_prefix


parser = argparse.ArgumentParser()
parser.add_argument('--topk', default=40, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--model', default='hidden-logit-residual', type=str)
parser.add_argument('--emb', default=None, type=str)
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--epoch', default=1000, type=int)
parser.add_argument('--bs', default=1000, type=int)
parser.add_argument('--save_epoch', default=-1, type=int)
parser.add_argument('--data_dir', default="logits/gsm/dexperts-7B-chat7B/", type=str)
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument(
        "--prefix",
        type=str,
        nargs='+',
        default =  ["0_1000"]
    )
args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu}')

if args.emb is None:
    args.emb = os.path.join(args.data_dir, "emb.pt")


data_dir = args.data_dir
sufixes = args.prefix
expert_logits= []
expert_topks = []
antiexpert_logits = []
antiexpert_topks = []
antiexpert_hiddens = []

for sufix in sufixes:
    expert = torch.load(os.path.join(data_dir, f"expert_{sufix}.bin"))
    antiexpert = torch.load(os.path.join(data_dir, f"antiexpert_{sufix}.bin"))
    antiexpert_hidden = torch.load(os.path.join(data_dir, f"antiexpert_hidden_{sufix}.bin"))
    expert_logit = expert["logit"]
    expert_topk = expert["indices"]
    antiexpert_topk = antiexpert["indices"]
    expert_logits.append(expert_logit)
    expert_topks.append(expert_topk)
    antiexpert_topks.append(antiexpert_topk)
    antiexpert_hiddens.append(antiexpert_hidden)
expert_logits = torch.cat(expert_logits, dim=0)
expert_topks = torch.cat(expert_topks, dim=0)
antiexpert_topks = torch.cat(antiexpert_topks, dim=0)
antiexpert_hiddens = torch.cat(antiexpert_hiddens, dim=0).float()


prefix_name = "_".join(merge_prefix(args.prefix))
if "top" in args.model:
    prefix_name += f"-top{args.topk}"
prefix_name += f"-{args.model}"
save_dir = os.path.join(data_dir, "logit_model", prefix_name, f"lr_{args.lr}_bs_{args.bs}")
os.makedirs(save_dir, exist_ok=True)
print(save_dir)


# logger
log_dir = os.path.join(save_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger('logit')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(log_dir, f'{current_time}.log'))
file_handler.setLevel(logging.INFO)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设置控制台日志级别


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(console_handler)


logger.info(f"Use emb {args.emb}")

W = torch.load(args.emb).float().t()
out_size = W.shape[1]
print(f"out size: {out_size}")

expert_max = expert_topks[:, 0]
antiexpert_max = antiexpert_topks[:, 0]



if args.shuffle:
    N = expert_topks.shape[0]
    shuffle_idx = torch.randperm(N)
    assert shuffle_idx.shape[0] == expert_topks.shape[0]
    expert_logits = expert_logits[shuffle_idx]
    expert_topks = expert_topks[shuffle_idx]
    antiexpert_topks = antiexpert_topks[shuffle_idx]
    antiexpert_hiddens = antiexpert_hiddens[shuffle_idx]




N =expert_topks.shape[0]
train_num = int(N*0.95)
test_num = N-train_num


hidden_size = W.shape[0]
ms = args.model.split('-')
assert len(ms) == 3
if ms[1] == "hidden":
    out_hidden = hidden_size
elif ms[1] == "logit":
    out_hidden = out_size
else:
    raise ValueError

if "hidden" in ms[0]:
    in_hidden = hidden_size
elif "logit" in ms[0]:
    in_hidden = out_size
else:
    raise ValueError

model = nn.Linear(in_hidden, out_hidden, bias=False)


model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.KLDivLoss(reduction="batchmean")
Y = expert_topks[:, 0].to(device)
original_logit = torch.mm(antiexpert_hiddens, W).detach()

if ms[0] == "hidden":
    X = antiexpert_hiddens
elif ms[0] == "logit":
    X = original_logit
elif ms[0] == "toplogit":
    X = get_topk(original_logit, args.topk)
elif ms[0] == "toplogitone":
    X = get_topk_one(original_logit, args.topk)
elif ms[0] == "toploglogit":
    X = get_topk_logsoftmax(original_logit, args.topk)
else:
    raise ValueError




model.train()
W = W.detach().to(device)


def extract_number(filename):
    match = re.search(r'epoch_(\d+).pt', filename)
    if match:
        return int(match.group(1))
    return 0






best_test_acc = 0
best_model = None
best_epoch = -1


start_epoch=0

for e in trange(1+start_epoch, args.epoch + 1):
    model.train()
    train_index = torch.randperm(train_num)
    total_loss = 0
    correct=0
    for i in range(0, train_num, args.bs):
        batch_index = train_index[i:i + args.bs]
        batch_X, batch_y, batch_logits = X[batch_index].to(device), Y[batch_index].to(device), original_logit[batch_index].detach().to(device)

        optimizer.zero_grad()
        pred = model(batch_X)
        if ms[1] == "hidden":
            pred = torch.mm(pred, W)
        if ms[2]=="residual":
            pred = pred+batch_logits
        batch_expert_logits = torch.softmax(expert_logits[batch_index], dim=-1)
        batch_expert_topks = expert_topks[batch_index]
        n = batch_expert_logits.shape[0]
        batch_distill_Y = torch.zeros((n, out_size))
        batch_distill_Y.scatter_(1, batch_expert_topks, batch_expert_logits)
        batch_distill_Y = batch_distill_Y.detach().to(device)
        loss = criterion(F.log_softmax(pred, dim=1), batch_distill_Y)
        loss.backward()
        optimizer.step()
        _, train_predicted_labels = torch.max(pred, 1)
        correct += (train_predicted_labels == batch_y).sum().item()

        total_loss += loss.item()
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_index = torch.arange(train_num, N)
        for i in  range(0, test_num, args.bs):
            batch_index = test_index[i:i + args.bs]
            batch_X, batch_y, batch_logits = X[batch_index].to(device), Y[batch_index].to(device), original_logit[batch_index].to(device)
            pred = model(batch_X)
            if ms[1] == "hidden":
                pred = torch.mm(pred, W)
            if ms[2] == "residual":
                pred = pred + batch_logits
            _, test_predicted_labels = torch.max(pred, 1)
            test_correct += (test_predicted_labels == batch_y).sum().item()
        test_accuracy = test_correct / test_num
    if args.save_epoch > 0 and e % args.save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{e}.pt"))
    logger.info(f'Epoch {e}, Loss: {total_loss / (train_num // args.bs):.4f}, Train ACC: {correct/train_num :.4f}, Test ACC: {test_accuracy :.4f}')

