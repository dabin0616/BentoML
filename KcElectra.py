import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter


def dataloader(args, step):

    if step == 'train':

        train_df = pd.read_csv('%d/%d'%(args.data_dir, args.train_data), sep='\t')
        train_sample = RandomSampler(train_df)
        train, dev = train_test_split()
    elif step == 'test':
        test_df = pd.read_csv('%d/%d' % (args.data_dir, args.test_dat), sep='\t')
        test_sample = RandomSampler(test_df)

def train():



def test(args):



def main(parser):

    args = parser.parse_args()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='NLI', help="수행하는 task 이름")
    parser.add_argument('--data_dir', default='data/KorNLI', help="데이터의 디렉토리")
    parser.add_argument('--train_data', default='multinli.train.ko.tsv', help="사용하는 파일 이름")
    parser.add_argument('--test_data', default='multinli.train.ko.tsv', help="사용하는 파일 이름")
    parser.add_argument('--batch', default=5, help="batch 크기")
    parser.add_argument('--epoch', default=1, help="학습을 진행하는 횟수")
    parser.add_argument('--max_len', default=300, help="입력의 문장 길이")
    parser.add_argument('--dropout', default=0.1, help="dropout 비율")
    parser.add_argument('--warmup_ratio', default='', help="")
    parser.add_argument('--max_gram_norm', default='', help="")
    parser.add_argument('--lr', default='', help="")
    parser.add_argument('--log_interval', default='', help="")
    parser.add_argument('--', default='', help="")


    main(parser)