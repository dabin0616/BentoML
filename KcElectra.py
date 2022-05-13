import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_bert import BertConfig
from pretrained.tokenization_ranked import FullTokenizer as KBertRankedTokenizer
from transformers import BertTokenizer
from model.net import SentenceClassifier
from model.data import Corpus
from model.utils import PreProcessor, PadSequence
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main():



if __name__ == '__main__':
    main()