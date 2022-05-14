import argparse
import pandas as pd
from kobert.pytorch_kobert import get_pytorch_kobert_model
import mlflow
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter


def dataloader(args, step):

    label_enc = LabelEncoder()

    if step == 'train':

        train_df = pd.read_csv('%d/%d'%(args.data_dir, args.train_data), sep='\t')
        train_df = train_df.dropna()
        train_x, train_y = train_df["text"].tolist(), train_df["label"].tolist()

        train_x, val_x, train_y, val_y = train_test_split(train_x,
                                                          train_y,
                                                          test_size=args.text_size,
                                                          stratify=train_y)

        train_y = label_enc.fit_transform(train_y)
        val_y = label_enc.transform(val_y)

        return train_x, train_y, val_x, val_y, label_enc

    elif step == 'test':
        test_df = pd.read_csv('%d/%d' % (args.data_dir, args.test_dat), sep='\t')
        test_df = test_df.dropna()
        test_x, test_y = test_df['text'].tolist(), test_df['label'].tolist()
        test_y = label_enc.transform(test_y)

        return test_x, test_y, label_enc


def train(args, model):

    best_acc = 0.0

    for i in range(args.epoch):

        train_acc , val_acc = 0.0, 0.0
        model.train()


def test(args, model):



def main(parser):

    args = parser.parse_args()

    mlflow.set_experiment('')
    with mlflow.start_run() as run:

        train_x, train_y, val_x, val_y, label_enc = dataloader(args, 'train')
        Koelectra_model, vocab = get_pytorch_kobert_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='NLI', help="수행하는 task 이름")
    parser.add_argument('--data_dir', default='data/KorNLI', help="데이터의 디렉토리")
    parser.add_argument('--train_data', default='multinli.train.ko.tsv', help="사용하는 파일 이름")
    parser.add_argument('--test_data', default='multinli.train.ko.tsv', help="사용하는 파일 이름")
    parser.add_argument('--batch', default=5, help="batch 크기")
    parser.add_argument('--test_size', default=0.2, help="val 데이터 비율")
    parser.add_argument('--epoch', default=1, help="학습을 진행하는 횟수")
    parser.add_argument('--max_len', default=300, help="입력의 문장 길이")
    parser.add_argument('--dropout', default=0.1, help="dropout 비율")
    parser.add_argument('--warmup_ratio', default='', help="")
    parser.add_argument('--max_gram_norm', default='', help="")
    parser.add_argument('--lr', default='', help="")
    parser.add_argument('--log_interval', default='', help="")
    # parser.add_argument('--', default='', help="")


    main(parser)