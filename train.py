"""
Disabled ANCE: commented out saving ranks in numpy file
"""


import os
import torch
import numpy as np
from utils import get_datasets, linear_learning_rate_scheduler, evaluate
from transformers import AutoTokenizer
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
import time
import argparse

from models.coil import COIL
from models.colbert import ColBERT
from models.mores_plus import MORES_PLUS
from models.mores import MORES

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='coil',
                        help='Enter model name: coil, colbert, mores, mores+')

    parser.add_argument('--dataset', type=str, default='mimic3',
                        help='Enter dataset name: mimic3 (default), mimic3-50')

    parser.add_argument('--version_name', type=str, required=True,
                        help='Enter version name for logging and checkpoints')
    parser.add_argument('--load_from_ckpt', type=str, default='',
                        help='If given, the model will be loaded from the given string.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Start epoch number for continuous logging and checkpoint saving purpose.')
    parser.add_argument('--evaluate_first', action='store_true',
                        help='Evaluate first before training (When model is loaded from a checkpoint).')
    parser.add_argument('--evaluate_period_in_epochs', type=int, default=1,
                        help='Evaluate will be held every k epochs when k is given as the input')
    parser.add_argument('--test', action='store_true')

    # learning rate
    parser.add_argument('--lr', type=float, default=3e-6,
                        help='Enter learning rate (default: 3e-6)')
    parser.add_argument('--target_epoch', type=int, default=3,
                        help='total training epochs')
    parser.add_argument('--lr_warmup_ratio', type=float, default=0.1,
                        help='linearly warmup learning rate from 0 for the given ratio of total training steps')
    parser.add_argument('--lr_decay', action='store_true',
                        help='linearly decay learning rate once warmup is finished')

    # used for pre-trained parameters and tokenizer
    parser.add_argument('--config_name', type=str, default="yikuan8/Clinical-Longformer",
                        help="""configuration name for pretrained models.\n
                        \t For ColBERT, COIL, and MORES:
                        \t\t RoBERTa-base-PM-M3-Voc-distill-align-hf, RoBERTa-large-PM-M3-Voc-hf \n
                        \t\t yikuan8/Clinical-Longformer \n
                        \t For MORES+: \n
                        \t\t 'GanjinZero/biobart-v2-large' """)

    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help='loss function type: cross_entropy (cross entropy loss), bce (binary cross entropy loss)')

    # data size and sample method
    parser.add_argument('--truncate_length', type=int, default=4000,
                        help='Truncation length for discharge notes')
    parser.add_argument('--label_truncate_length', type=int, default=30,
                        help='Truncation length for code synonyms') ###### later upgrade to 32
    parser.add_argument('--term_count', type=int, default=4,
                        help='Number of code synonyms used for each code')
    parser.add_argument('--sample_method', type=str, default='pos_neg_random',
                        help="""Sampling methods for positive and negative code synonyms for training \n
                        \t ance: approximate nearest neighbor negative contrastive learning (ANCE)
                        \t pos_neg_top300: localized contrastive learning using top 300'
                        \t pos_neg_random: alternate positive code synonyms and randomly sample negative samples
                        \t 'one': use exactly one datapoint of code synonyms with the given n_pos, n_neg ratio
                        \t 'in-batch': in-batch training (n_pos will be used as the batch size) 
                        """)
    parser.add_argument('--n_pos', type=int, default=1,
                        help='Number of positive codes for each training step')
    parser.add_argument('--n_neg', type=int, default=19,
                        help='Number of negative codes for each training step')
    parser.add_argument('--switch_qd', action='store_true',
                        help='Use discharge notes as the document and use code synonyms as the query')
    parser.add_argument('--use_filter', action='store_true',
                        help='Filter punctuations in encoded discharge notes in ColBERT')

    # ance
    parser.add_argument('--ance_warmup_unit', type=str, default='none',
                        help='Warmup unit for ANCE: step, epoch, none') # 'step' 'epoch', 'none'
    parser.add_argument('--ance_warmup', type=int, default=0,
                        help='Warmup number: Enter an integer')
    parser.add_argument('--ance_update_interval', type=int, default=10,
                        help='Update interval for ANCE in training steps')
    parser.add_argument('--n_ranks_ance', type=int, default=200,
                        help='Number of top ranked documents to choose negative samples from (ANCE)')

    parser.add_argument('--log_interval', type=int, default=100,
                        help='logging interval in training steps')
    parser.add_argument('--save_checkpoints', type=int, default=1,
                        help='Save checkpoints if not 0 (default: save)')

    # model args
    parser.add_argument('--vector_dim', type=int, default=768,
                        help='This will be automatically changed (No need to modify).')
    parser.add_argument('--coil_token_dim', type=int, default=768,
                        help='Token dimension for COIL')
    parser.add_argument('--coil_cls_dim', type=int, default=768,
                        help='CLS dimension for COIL')
    parser.add_argument('--colbert_token_dim', type=int, default=768,
                        help='Token dimension for ColBERT')
    parser.add_argument('--colbert_cls_dim', type=int, default=0,
                        help='Token dimension for ColBERT (default: not using CLS token)') # default: no cls
    parser.add_argument('--self_attention', action='store_true',
                        help='Use self attention in MORES')

    args = parser.parse_args()
    assert args.dataset in ['mimic3', 'mimic3-50']
    assert args.loss in ['cross_entropy', 'bce']
    assert args.sample_method in ['ance', 'pos_neg_top300', 'pos_neg_random', 'one', 'in-batch']
    assert args.ance_warmup_unit in ['step', 'epoch', 'none']

    if args.sample_method == 'one':
        assert args.loss == 'bce'

    if args.sample_method == 'in-batch':
        if args.model_name != 'colbert':
            raise NotImplementedError()

    if args.model_name == 'mores+':
        args.config_name = 'GanjinZero/biobart-v2-large'

    if 'large' in args.config_name:
        args.vector_dim = 1024

    return args

def collate_fn(data):
    return data[0]

def main():
    args = parse()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    tokenizer.convert_tokens_to_ids(['<s>', '</s>', '<pad>'])

    chunk_notes =  args.config_name != "yikuan8/Clinical-Longformer"
    if args.use_filter and chunk_notes:
        raise NotImplementedError()

    mimic3_train_dataset, mimic3_dev_dataset, mimic3_test_dataset = get_datasets(
        config_name=args.config_name,
        version=args.dataset,
        model_name=args.model_name,
        truncate_length=args.truncate_length,
        label_truncate_length=args.label_truncate_length,
        term_count=args.term_count,
        sample_method=args.sample_method,
        n_ranks_ance=args.n_ranks_ance,
        n_samples=(args.n_pos, args.n_neg),
        return_tensors="pt",
        loss=args.loss,
        switch_qd=args.switch_qd,
        chunk_notes=chunk_notes
    )
    print(len(mimic3_train_dataset), len(mimic3_dev_dataset), len(mimic3_test_dataset))

    if args.test:
        train_dataset = torch.utils.data.ConcatDataset([mimic3_train_dataset, mimic3_dev_dataset])
        eval_dataset = mimic3_test_dataset
    else:
        train_dataset = mimic3_train_dataset
        eval_dataset = mimic3_dev_dataset
    print(len(train_dataset), len(eval_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # train_dataset has its own shuffle (train_dataset.perms)

    if args.model_name == 'coil':
        model = COIL(args, device, args.loss, args.switch_qd, chunk_notes)
        print('COIL')
    elif args.model_name == 'colbert':
        model = ColBERT(args, device, chunk_notes)
        model.encoder.resize_token_embeddings(len(mimic3_train_dataset.tokenizer))
        print('ColBERT')
    elif args.model_name == 'mores':
        model = MORES(args, device, chunk_notes)
        print('MORES')
    elif args.model_name == 'mores+':
        model = MORES_PLUS(args, device, args.loss, args.switch_qd)
        print('MORES+')
    else:
        raise ValueError()
    print('config_name:', args.config_name)

    if args.load_from_ckpt:
        model.load_state_dict(torch.load(args.load_from_ckpt + '.pt'))
        print('model loaded from', args.load_from_ckpt + '.pt')
        try:
            train_dataset.ranks = np.load(args.load_from_ckpt + '.npy')
            print('train_dataset.ranks loaded from', args.load_from_ckpt + '.npy')
        except:
            print('(exception) train_dataset.ranks load from npy file failed')

    model = model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    log_dir = os.path.join('logs', args.version_name)
    os.makedirs(log_dir, mode=0o777, exist_ok=True)
    writer = SummaryWriter(log_dir)

    k = args.log_interval # print result interval
    ance_update_interval = args.ance_update_interval

    if args.evaluate_first:
        evaluate(model, train_dataset, eval_dataset, writer, args.start_epoch, args)

    n_step = args.start_epoch * len(train_dataset) + 1
    warmup_steps = int(len(train_dataset) * args.target_epoch * args.lr_warmup_ratio)
    total_steps = len(train_dataset) * args.target_epoch
    decay_steps = total_steps if args.lr_decay else 0
    for epoch in range(args.start_epoch+1, args.target_epoch+1): # 1-index matches with the warm-up count and evaluate_period

        model.train(True)

        loss_k, n_correct_k, n_total_k = 0, 0, 0
        loss_epoch, n_correct_epoch, n_total_epoch = 0, 0, 0
        time_epoch = time.time()

        if args.sample_method == 'in-batch':
            train_dataset.shuffle_indices()

        time_k = time.time()
        for i, data in enumerate(train_dataloader):

            # exception: medical note has no relevant code description (4 in the training set)
            if data == -1:
                continue

            optimizer, running_lr = linear_learning_rate_scheduler(optimizer, n_step, args.lr, warmup_steps, decay_steps)
            optimizer.zero_grad()

            loss, n_correct, n_total = model(data)

            # Backpropagate
            loss.backward()
            optimizer.step()

            loss_k += loss.tolist()
            loss_epoch += loss.tolist()
            n_correct_k += n_correct
            n_correct_epoch += n_correct
            n_total_k += n_total
            n_total_epoch += n_total

            if (i+1) % k == 0:
                loss_k /= min(i+1, k)
                acc_k = n_correct_k / n_total_k
                writer.add_scalar('loss_train', loss_k, n_step)
                writer.add_scalar('acc_train', acc_k, n_step)
                writer.add_scalar('lr', running_lr, n_step)

                print('[epoch %d, step %d], loss: %.6f, acc: %.6f, time: %.6f' % (epoch, i+1, loss_k, acc_k, time.time()-time_k))
                writer.flush()

                loss_k, n_correct_k, n_total_k = 0, 0, 0
                time_k = time.time()

            if args.sample_method == 'ance' and (i+1) % ance_update_interval == 0:
                if args.ance_warmup_unit == 'epoch' and epoch <= args.ance_warmup: # 1-indexed
                    continue
                if args.ance_warmup_unit == 'step' and n_step < args.ance_warmup: # 0-indexed
                    continue
                doc_vectors = model.encode_code_synonyms(train_dataset, batch_size=200)
                time_rank = time.time()
                model.rank(
                    dataset=train_dataset,
                    c_descs_vectors=doc_vectors,
                    return_eval=False,
                    save_topk=True)
                print('rank', i+1, i+ance_update_interval+1, '%.6f' % (time.time() - time_rank))

            n_step += 1
        print(len(train_dataset), n_total_epoch)
        print('[train epoch %d] avg_loss: %.6f, avg_acc: %.6f, total_time: %.6f' % \
                  (epoch, loss_epoch / len(train_dataset), n_correct_epoch / n_total_epoch , time.time()-time_epoch))

        model.train(False)
        writer.add_scalar('loss_epoch_train', loss_epoch / len(train_dataset), epoch)
        writer.add_scalar('acc_epoch_train', n_correct_epoch / n_total_epoch, epoch)
        writer.flush()

        if args.save_checkpoints:
            checkpoint_dir = os.path.join('checkpoints', args.version_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = os.path.join(checkpoint_dir, 'epoch%d.pt' % epoch)
            torch.save(model.state_dict(), checkpoint_name)

        loss_epoch, n_correct_epoch, n_total_epoch = 0, 0, 0
        time_epoch = time.time()
        for i, data in enumerate(eval_dataloader):
            with torch.no_grad():
                loss, n_correct, n_total = model(data)

                loss_epoch += loss.tolist()
                loss_epoch += loss.tolist()
                n_correct_epoch += n_correct
                n_total_epoch += n_total

        print('[dev epoch %d] avg_loss: %.6f, avg_acc: %.6f, total_time: %.6f' % \
                  (epoch, loss_epoch / len(eval_dataset), n_correct_epoch / n_total_epoch, time.time()-time_epoch))
        writer.add_scalar('loss_epoch_dev', loss_epoch / len(eval_dataset), epoch)
        writer.add_scalar('acc_epoch_dev', n_correct_epoch / n_total_epoch, epoch)
        writer.flush()

        if epoch % args.evaluate_period_in_epochs == 0:
            evaluate(model, train_dataset, eval_dataset, writer, epoch, args)

    writer.close()


if __name__ == '__main__':
    main()