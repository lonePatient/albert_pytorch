import torch
import json
import time
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
from tempfile import TemporaryDirectory
from tools.common import logger, init_logger
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tools.common import AverageMeter
from metrics.custom_metrics import LMAccuracy
from torch.nn import CrossEntropyLoss
from model.modeling_albert import AlbertForPreTraining, AlbertConfig
from model.file_utils import CONFIG_NAME
from model.tokenization_bert import BertTokenizer
from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from tools.common import seed_everything

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1
    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids
    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, file_id, tokenizer, data_name, reduce_memory=False):
        self.tokenizer = tokenizer
        self.file_id = file_id
        data_file = training_path / f"{data_name}_file_{self.file_id}.json"
        metrics_file = training_path / f"{data_name}_file_{self.file_id}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logger.info(f"Loading training examples for {str(data_file)}")
        with data_file.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logger.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))
def main():
    parser = ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--config_path", default=None, type=str, required=True)
    parser.add_argument("--vocab_path",default=None,type=str,required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_path", default='', type=str)
    parser.add_argument('--data_name', default='albert', type=str)
    parser.add_argument("--file_num", type=int, default=10,
                        help="Number of dynamic masking to pregenerate (with different masks)")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs to train for")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--num_eval_steps', default=1000)
    parser.add_argument('--num_save_steps', default=2000)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument("--learning_rate", default=0.000176, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)

    pregenerated_data = args.data_dir / "corpus/train"
    init_logger(log_file=str(args.output_dir/ "train_albert_model.log"))
    assert pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by prepare_lm_data_mask.py!"

    samples_per_epoch = 0
    for i in range(args.file_num):
        data_file = pregenerated_data / f"{args.data_name}_file_{i}.json"
        metrics_file = pregenerated_data / f"{args.data_name}_file_{i}_metrics.json"
        if data_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch += metrics['num_training_examples']
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            break
    logger.info(f"samples_per_epoch: {samples_per_epoch}")
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(f"cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} , distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    seed_everything(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=args.do_lower_case)
    total_train_examples = samples_per_epoch * args.epochs

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    args.warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)

    bert_config = AlbertConfig.from_pretrained(args.config_path)
    model = AlbertForPreTraining(config=bert_config)
    if args.model_path:
        model = AlbertForPreTraining.from_pretrained(args.model_path)
    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.model_path:
        optimizer.load_state_dict(torch.load(args.model_path + "/optimizer.bin"))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    global_step = 0
    mask_metric = LMAccuracy()
    sop_metric = LMAccuracy()
    tr_mask_acc = AverageMeter()
    tr_sop_acc = AverageMeter()
    tr_loss = AverageMeter()
    tr_mask_loss = AverageMeter()
    tr_sop_loss = AverageMeter()
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    train_logs = {}
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_examples}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = {num_train_optimization_steps}")
    logger.info(f"  warmup_steps = {args.warmup_steps}")
    start_time = time.time()
    seed_everything(args.seed)  # Added here for reproducibility
    for epoch in range(args.epochs):
        for idx in range(args.file_num):
            epoch_dataset = PregeneratedDataset(file_id=idx, training_path=pregenerated_data, tokenizer=tokenizer,
                                                reduce_memory=args.reduce_memory, data_name=args.data_name)
            if args.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
            model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                prediction_scores = outputs[0]
                seq_relationship_score = outputs[1]

                masked_lm_loss = loss_fct(prediction_scores.view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
                next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), is_next.view(-1))
                loss = masked_lm_loss + next_sentence_loss

                mask_metric(logits=prediction_scores.view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
                sop_metric(logits=seq_relationship_score.view(-1, 2), target=is_next.view(-1))

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                nb_tr_steps += 1
                tr_mask_acc.update(mask_metric.value(), n=input_ids.size(0))
                tr_sop_acc.update(sop_metric.value(), n=input_ids.size(0))
                tr_loss.update(loss.item(), n=1)
                tr_mask_loss.update(masked_lm_loss.item(), n=1)
                tr_sop_loss.update(next_sentence_loss.item(), n=1)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % args.num_eval_steps == 0:
                    now = time.time()
                    eta = now - start_time
                    if eta > 3600:
                        eta_format = ('%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60))
                    elif eta > 60:
                        eta_format = '%d:%02d' % (eta // 60, eta % 60)
                    else:
                        eta_format = '%ds' % eta
                    train_logs['loss'] = tr_loss.avg
                    train_logs['mask_acc'] = tr_mask_acc.avg
                    train_logs['sop_acc'] = tr_sop_acc.avg
                    train_logs['mask_loss'] = tr_mask_loss.avg
                    train_logs['sop_loss'] = tr_sop_loss.avg
                    show_info = f'[Training]:[{epoch}/{args.epochs}]{global_step}/{num_train_optimization_steps} ' \
                                f'- ETA: {eta_format}' + "-".join(
                        [f' {key}: {value:.4f} ' for key, value in train_logs.items()])
                    logger.info(show_info)
                    tr_mask_acc.reset()
                    tr_sop_acc.reset()
                    tr_loss.reset()
                    tr_mask_loss.reset()
                    tr_sop_loss.reset()
                    start_time = now

                if global_step % args.num_save_steps == 0:
                    if args.local_rank in [-1, 0] and args.num_save_steps > 0:
                        # Save model checkpoint
                        output_dir = args.output_dir / f'lm-checkpoint-{global_step}'
                        if not output_dir.exists():
                            output_dir.mkdir()
                        # save model
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(str(output_dir))
                        torch.save(args, str(output_dir / 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), str(output_dir / "optimizer.bin"))
                        # save config
                        output_config_file = output_dir / CONFIG_NAME
                        with open(str(output_config_file), 'w') as f:
                            f.write(model_to_save.config.to_json_string())
                        # save vocab
                        tokenizer.save_vocabulary(output_dir)
