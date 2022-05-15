import argparse
import pandas as pd
from KoBERT.pytorch_kobert import get_pytorch_kobert_model
from KoBERT.utils import get_tokenizer
import mlflow
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
import gluonnlp as nlp
import pandas as pd
import torch
import mlflow
import pickle
from KoBERT.pytorch_kobert import get_pytorch_kobert_model
from KoBERT.utils import get_tokenizer
from BERT import BERTDataset, BERTClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from KoBERT_service import SentimnetClassifier


def calculate_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def dataloader(args, stage):

    label_enc = LabelEncoder()

    if stage == 'train':

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

    elif stage == 'test':
        test_df = pd.read_csv('%d/%d' % (args.data_dir, args.test_dat), sep='\t')
        test_df = test_df.dropna()
        test_x, test_y = test_df['text'].tolist(), test_df['label'].tolist()
        test_y = label_enc.transform(test_y)

        return test_x, test_y, label_enc


def train(args, model, train_loader, val_loader, optimizer, loss_fn, scheduler):

    best_acc = 0.0

    for i in range(args.epoch):

        train_acc , val_acc = 0.0, 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_y = model(token_ids.long().to(args.device), valid_length, segment_ids.long().to(args.device))
            y = label.long().to(args.device)
            loss = loss_fn(pred_y, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_acc += calculate_accuracy(pred_y, y)

            if batch_id % (len(train_loader) // 10) == 0: print(batch_id, len(train_loader))

        print("epoch {} train acc {}".format(i + 1, train_acc / (batch_id + 1)))

        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(val_loader):
                pred_y = model(token_ids.long().to(args.device), valid_length, segment_ids.long().to(args.device))
                y = label.long().to(args.device)
                val_acc += calculate_accuracy(pred_y, y)

        val_score = val_acc / (batch_id + 1)
        print("epoch {} val acc {}".format(i + 1, val_score))

        if val_score > best_acc:
            best_acc = val_score
            best_model = model

    return best_model


def test(args, model,test_x, test_y, test_loader, label_enc):

    test_y, pred_y = label_enc.inverse_transform(test_y), []

    model.eval()
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in test_loader:
            output = model(token_ids.long().to(args.device), valid_length, segment_ids.long().to(device))
            _, output = torch.max(output, 1)
            output = label_enc.inverse_transform(output.cpu())
            pred_y.extend(output)

    df_result = pd.DataFrame({"sentence": test_x, "pred_y": pred_y, "test_y": test_y})
    df_result.to_csv("result.csv", index=False, encoding="utf8")

    f1 = f1_score(test_y, pred_y, average='weighted')
    precision = precision_score(test_y, pred_y, average='weighted')
    recall = recall_score(test_y, pred_y, average='weighted')


    return f1, precision, recall


def main(parser):

    args = parser.parse_args()

    mlflow.set_experiment(args.task)
    with mlflow.start_run() as run:

        mlflow.log_params(vars(args))

        train_x, train_y, val_x, val_y, label_enc = dataloader(args, stage='train')

        kobert_model, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        train_set = BERTDataset(train_x, train_y, tokenizer, args.max_len, pad=True, pair=False)
        val_set = BERTDataset(val_x, val_y, tokenizer, args.max_len, pad=True, pair=False)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, shuffle=False)

        model = BERTClassifier(kobert_model, num_classes=len(label_enc.classes_), dr_rate=args.lr).to(args.device)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        loss_fn = nn.CrossEntropyLoss()
        total_step = len(train_set) * args.epoch
        warmup_step = int(total_step * args.warmup_ratio)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_step)

        model = train(args, kobert_model,train_loader, val_loader, optimizer, loss_fn, scheduler)

        test_x, test_y, label_enc = dataloader(args, stage='test')
        test_set = BERTDataset(test_x, test_y, tokenizer, args.max_len, pad=True, pair=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False)
        f1, percision, recall = test(args, model, test_x, test_y, test_loader, label_enc)

        mlflow.log_param('F1', f1)
        mlflow.log_param('Percision', percision)
        mlflow.log_param('Recall', recall)

        print("5. save file")
        torch.save(model.state_dict(), args.model_dir)
        pickle.dump(tokenizer, open(args.tok_dir, 'wb'))
        pickle.dump(label_enc, open(args.label_enc_dir, 'wb'))

        print("5. packing")
        review_classifier = SentimnetClassifier()
        review_classifier.pack('model', model)
        review_classifier.pack('tokenizer', tokenizer)
        review_classifier.pack('max_len', args.max_len)
        review_classifier.pack('BERTDataset', BERTDataset)
        review_classifier.pack('label_enc', label_enc)
        saved_path = review_classifier.save()

        with open("bentoml_model_dir.txt", "w") as f:
            f.write(saved_path)
        mlflow.log_artifact("result.csv")
        mlflow.log_artifact("bentoml_model_dir.txt")

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