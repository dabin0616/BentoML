

def calculate_accuracy(x,y):

    acc = 0
    return acc

def dataloader(args, stage):

    if stage=="train":
        pass
    elif stage=='test':
        pass

def train(args, model, train_x, train_y, val_x, val_y):
    best_model = ''
    return best_model

def test():
    return 0

def main(parser):

    args = parser.parse_args()

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
    parser.add_argument('--lr', default='', help="learning rate")
    parser.add_argument('--log_interval', default='', help="")
    parser.add_argument('--device', default='cuda', help="")
    parser.add_argument('--model_dir', default='./model', help="모델 저장하는 디렉토리")
    parser.add_argument('--tok_dir', default='', help="tokenizer 불러오는 디렉토리")
    parser.add_argument('--label_enc_dir', default='', help="label encoder 불러오는 디렉토리")

    main(parser)