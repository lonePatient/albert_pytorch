from __future__ import absolute_import, division, print_function
import argparse
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from model.file_utils import WEIGHTS_NAME, CONFIG_NAME
from model.modeling_albert import BertConfig
from model.optimization import AdamW, WarmupLinearSchedule
from common.tools import seed_everything
from common.tools import logger, init_logger
from configs.base import config
from model.modeling_albert import BertForSequenceClassification
from callback.progressbar import ProgressBar
from lcqmc_progressor import BertProcessor
from common.metrics import Accuracy
from common.tools import AverageMeter


def train(args, train_dataloader, eval_dataloader, metrics, model):
    """ Train the model """

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    args.warmup_steps = t_total * args.warmup_proportion
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_acc = 0
    model.zero_grad()
    seed_everything(args.seed)
    for epoch in range(int(args.num_train_epochs)):
        tr_loss = AverageMeter()
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss.update(loss.item(), n=1)
            pbar(step, info={"loss": loss.item()})
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        train_log = {'loss': tr_loss.avg}
        eval_log = evaluate(args, model, eval_dataloader, metrics)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)

        if logs['eval_acc'] > best_acc:
            logger.info(f"\nEpoch {epoch}: eval_acc improved from {best_acc} to {logs['eval_acc']}")
            logger.info("save model to disk.")
            best_acc = logs['eval_acc']
            print("Valid Entity Score: ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_file = args.model_save_path
            output_file.mkdir(exist_ok=True)
            output_model_file = output_file / WEIGHTS_NAME
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = output_file / CONFIG_NAME
            with open(str(output_config_file), 'w') as f:
                f.write(model_to_save.config.to_json_string())


def evaluate(args, model, eval_dataloader, metrics):
    # Eval!
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = AverageMeter()
    metrics.reset()
    preds = []
    targets = []
    pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
    for bid, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            eval_loss.update(loss.item(), n=batch[0].size()[0])
        preds.append(logits.cpu().detach())
        targets.append(inputs['labels'].cpu().detach())
        pbar(bid)
    preds = torch.cat(preds, dim=0).cpu().detach()
    targets = torch.cat(targets, dim=0).cpu().detach()
    metrics(preds, targets)
    eval_log = {"eval_acc": metrics.value(),
                'eval_loss': eval_loss.avg}
    return eval_log


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", default='albert_xlarge', type=str)
    parser.add_argument('--task_name', default='lcqmc', type=str)
    parser.add_argument("--train_max_seq_len", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_max_seq_len", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--share_type', default='all', type=str, choices=['all', 'attention', 'ffn', 'None'])
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=int,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    args.model_save_path = config['checkpoint_dir'] / f'{args.arch}'
    args.model_save_path.mkdir(exist_ok=True)

    # Setudistant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    init_logger(log_file=config['log_dir'] / 'finetuning.log')
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)
    # --------- data
    processor = BertProcessor(vocab_path=config['albert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    bert_config = BertConfig.from_pretrained(str(config['albert_config_path']),
                                             share_type=args.share_type, num_labels=num_labels)

    logger.info("Training/evaluation parameters %s", args)
    metrics = Accuracy(topK=1)
    # Training
    if args.do_train:
        train_data = processor.get_train(config['data_dir'] / "train.txt")
        train_examples = processor.create_examples(lines=train_data, example_type='train',
                                                   cached_examples_file=config[
                                                                            'data_dir'] / f"cached_train_examples_{args.arch}")
        train_features = processor.create_features(examples=train_examples, max_seq_len=args.train_max_seq_len,
                                                   cached_features_file=config[
                                                                            'data_dir'] / "cached_train_features_{}_{}".format(
                                                       args.train_max_seq_len, args.arch
                                                   ))
        train_dataset = processor.create_dataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        valid_data = processor.get_dev(config['data_dir'] / "dev.txt")
        valid_examples = processor.create_examples(lines=valid_data, example_type='valid',
                                                   cached_examples_file=config[
                                                                            'data_dir'] / f"cached_valid_examples_{args.arch}")
        valid_features = processor.create_features(examples=valid_examples, max_seq_len=args.eval_max_seq_len,
                                                   cached_features_file=config[
                                                                            'data_dir'] / "cached_valid_features_{}_{}".format(
                                                       args.eval_max_seq_len, args.arch
                                                   ))
        valid_dataset = processor.create_dataset(valid_features)
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

        model = BertForSequenceClassification.from_pretrained(config['bert_dir'], config=bert_config)
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        model.to(args.device)
        train(args, train_dataloader, valid_dataloader, metrics, model)

    if args.do_test:
        test_data = processor.get_train(config['data_dir'] / "test.txt")
        test_examples = processor.create_examples(lines=test_data,
                                                  example_type='test',
                                                  cached_examples_file=config[
                                                                           'data_dir'] / f"cached_test_examples_{args.arch}")
        test_features = processor.create_features(examples=test_examples,
                                                  max_seq_len=args.eval_max_seq_len,
                                                  cached_features_file=config[
                                                                           'data_dir'] / "cached_test_features_{}_{}".format(
                                                      args.eval_max_seq_len, args.arch
                                                  ))
        test_dataset = processor.create_dataset(test_features)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
        model = BertForSequenceClassification.from_pretrained(args.model_save_path, config=bert_config)
        model.to(args.device)
        test_log = evaluate(args, model, test_dataloader, metrics)
        print(test_log)


if __name__ == "__main__":
    main()
