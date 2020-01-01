import os
import json
import random
import collections
from tools.common import logger, init_logger
from argparse import ArgumentParser
from tools.common import seed_everything
from model.tokenization_bert import BertTokenizer
from callback.progressbar import ProgressBar
from pathlib import Path

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq, vocab_words):
    """Creates `TrainingInstance`s for a single document.
     This method is changed to create sentence-order prediction (SOP) followed by idea from paper of ALBERT, 2019-08-28, brightmart
    """
    document = all_documents[document_index]  # 得到一个文档

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:  # 有一定的比例，如10%的概率，我们使用比较短的序列长度，以缓解预训练的长序列和调优阶段（可能的）短序列的不一致情况
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    # 设法使用实际的句子，而不是任意的截断句子，从而更好的构造句子连贯性预测的任务
    instances = []
    current_chunk = []  # 当前处理的文本段，包含多个句子
    current_length = 0
    i = 0
    # print("###document:",document) # 一个document可以是一整篇文章、新闻、词条等. document:[['是', '爷', '们', '，', '就', '得', '给', '媳', '妇', '幸', '福'], ['关', '注', '【', '晨', '曦', '教', '育', '】', '，', '获', '取', '育', '儿', '的', '智', '慧', '，', '与', '孩', '子', '一', '同', '成', '长', '！'], ['方', '法', ':', '打', '开', '微', '信', '→', '添', '加', '朋', '友', '→', '搜', '号', '→', '##he', '##bc', '##x', '##jy', '##→', '关', '注', '!', '我', '是', '一', '个', '爷', '们', '，', '孝', '顺', '是', '做', '人', '的', '第', '一', '准', '则', '。'], ['甭', '管', '小', '时', '候', '怎', '么', '跟', '家', '长', '犯', '混', '蛋', '，', '长', '大', '了', '，', '就', '底', '报', '答', '父', '母', '，', '以', '后', '我', '媳', '妇', '也', '必', '须', '孝', '顺', '。'], ['我', '是', '一', '个', '爷', '们', '，', '可', '以', '花', '心', '，', '可', '以', '好', '玩', '。'], ['但', '我', '一', '定', '会', '找', '一', '个', '管', '的', '住', '我', '的', '女', '人', '，', '和', '我', '一', '起', '生', '活', '。'], ['28', '岁', '以', '前', '在', '怎', '么', '玩', '都', '行', '，', '但', '我', '最', '后', '一', '定', '会', '找', '一', '个', '勤', '俭', '持', '家', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '我', '不', '会', '让', '自', '己', '的', '女', '人', '受', '一', '点', '委', '屈', '，', '每', '次', '把', '她', '抱', '在', '怀', '里', '，', '看', '她', '洋', '溢', '着', '幸', '福', '的', '脸', '，', '我', '都', '会', '引', '以', '为', '傲', '，', '这', '特', '么', '就', '是', '我', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '干', '什', '么', '也', '不', '能', '忘', '了', '自', '己', '媳', '妇', '，', '就', '算', '和', '哥', '们', '一', '起', '喝', '酒', '，', '喝', '到', '很', '晚', '，', '也', '要', '提', '前', '打', '电', '话', '告', '诉', '她', '，', '让', '她', '早', '点', '休', '息', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '绝', '对', '不', '能', '抽', '烟', '，', '喝', '酒', '还', '勉', '强', '过', '得', '去', '，', '不', '过', '该', '喝', '的', '时', '候', '喝', '，', '不', '该', '喝', '的', '时', '候', '，', '少', '扯', '纳', '极', '薄', '蛋', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '必', '须', '听', '我', '话', '，', '在', '人', '前', '一', '定', '要', '给', '我', '面', '子', '，', '回', '家', '了', '咱', '什', '么', '都', '好', '说', '。'], ['我', '是', '一', '爷', '们', '，', '就', '算', '难', '的', '吃', '不', '上', '饭', '了', '，', '都', '不', '张', '口', '跟', '媳', '妇', '要', '一', '分', '钱', '。'], ['我', '是', '一', '爷', '们', '，', '不', '管', '上', '学', '还', '是', '上', '班', '，', '我', '都', '会', '送', '媳', '妇', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '交', '往', '不', '到', '1', '年', '，', '绝', '对', '不', '会', '和', '媳', '妇', '提', '过', '分', '的', '要', '求', '，', '我', '会', '尊', '重', '她', '。'], ['我', '是', '一', '爷', '们', '，', '游', '戏', '永', '远', '比', '不', '上', '我', '媳', '妇', '重', '要', '，', '只', '要', '媳', '妇', '发', '话', '，', '我', '绝', '对', '唯', '命', '是', '从', '。'], ['我', '是', '一', '爷', '们', '，', '上', 'q', '绝', '对', '是', '为', '了', '等', '媳', '妇', '，', '所', '有', '暧', '昧', '的', '心', '情', '只', '为', '她', '一', '个', '女', '人', '而', '写', '，', '我', '不', '一', '定', '会', '经', '常', '写', '日', '志', '，', '可', '是', '我', '会', '告', '诉', '全', '世', '界', '，', '我', '很', '爱', '她', '。'], ['我', '是', '一', '爷', '们', '，', '不', '一', '定', '要', '经', '常', '制', '造', '浪', '漫', '、', '偶', '尔', '过', '个', '节', '日', '也', '要', '送', '束', '玫', '瑰', '花', '给', '媳', '妇', '抱', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '手', '机', '会', '24', '小', '时', '为', '她', '开', '机', '，', '让', '她', '半', '夜', '痛', '经', '的', '时', '候', '，', '做', '恶', '梦', '的', '时', '候', '，', '随', '时', '可', '以', '联', '系', '到', '我', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '经', '常', '带', '媳', '妇', '出', '去', '玩', '，', '她', '不', '一', '定', '要', '和', '我', '所', '有', '的', '哥', '们', '都', '认', '识', '，', '但', '见', '面', '能', '说', '的', '上', '话', '就', '行', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '和', '媳', '妇', '的', '姐', '妹', '哥', '们', '搞', '好', '关', '系', '，', '让', '她', '们', '相', '信', '我', '一', '定', '可', '以', '给', '我', '媳', '妇', '幸', '福', '。'], ['我', '是', '一', '爷', '们', '，', '吵', '架', '后', '、', '也', '要', '主', '动', '打', '电', '话', '关', '心', '她', '，', '咱', '是', '一', '爷', '们', '，', '给', '媳', '妇', '服', '个', '软', '，', '道', '个', '歉', '怎', '么', '了', '？'], ['我', '是', '一', '爷', '们', '，', '绝', '对', '不', '会', '嫌', '弃', '自', '己', '媳', '妇', '，', '拿', '她', '和', '别', '人', '比', '，', '说', '她', '这', '不', '如', '人', '家', '，', '纳', '不', '如', '人', '家', '的', '。'], ['我', '是', '一', '爷', '们', '，', '陪', '媳', '妇', '逛', '街', '时', '，', '碰', '见', '熟', '人', '，', '无', '论', '我', '媳', '妇', '长', '的', '好', '看', '与', '否', '，', '我', '都', '会', '大', '方', '的', '介', '绍', '。'], ['谁', '让', '咱', '爷', '们', '就', '好', '这', '口', '呢', '。'], ['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。'], ['【', '我', '们', '重', '在', '分', '享', '。'], ['所', '有', '文', '字', '和', '美', '图', '，', '来', '自', '网', '络', '，', '晨', '欣', '教', '育', '整', '理', '。'], ['对', '原', '文', '作', '者', '，', '表', '示', '敬', '意', '。'], ['】', '关', '注', '晨', '曦', '教', '育', '[UNK]', '[UNK]', '晨', '曦', '教', '育', '（', '微', '信', '号', '：', 'he', '##bc', '##x', '##jy', '）', '。'], ['打', '开', '微', '信', '，', '扫', '描', '二', '维', '码', '，', '关', '注', '[UNK]', '晨', '曦', '教', '育', '[UNK]', '，', '获', '取', '更', '多', '育', '儿', '资', '源', '。'], ['点', '击', '下', '面', '订', '阅', '按', '钮', '订', '阅', '，', '会', '有', '更', '多', '惊', '喜', '哦', '！']]
    while i < len(document):  # 从文档的第一个位置开始，按个往下看
        segment = document[
            i]  # segment是列表，代表的是按字分开的一个完整句子，如 segment=['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。']
        # segment = get_new_segment(segment)  # whole word mask for chinese: 结合分词的中文的whole mask设置即在需要的地方加上“##”
        current_chunk.append(segment)  # 将一个独立的句子加入到当前的文本块中
        current_length += len(segment)  # 累计到为止位置接触到句子的总长度
        if i == len(document) - 1 or current_length >= target_seq_length:
            # 如果累计的序列长度达到了目标的长度，或当前走到了文档结尾==>构造并添加到“A[SEP]B“中的A和B中；
            if current_chunk:  # 如果当前块不为空
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:  # 当前块，如果包含超过两个句子，取当前块的一部分作为“A[SEP]B“中的A部分
                    a_end = random.randint(1, len(current_chunk) - 1)
                # 将当前文本段中选取出来的前半部分，赋值给A即tokens_a
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # 构造“A[SEP]B“中的B部分(有一部分是正常的当前文档中的后半部;在原BERT的实现中一部分是随机的从另一个文档中选取的，）
                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                # 有百分之50%的概率交换一下tokens_a和tokens_b的位置
                # print("tokens_a length1:",len(tokens_a))
                # print("tokens_b length1:",len(tokens_b)) # len(tokens_b) = 0
                if len(tokens_a) == 0 or len(tokens_b) == 0: continue
                if random.random() < 0.5:  # 交换一下tokens_a和tokens_b
                    is_random_next = True
                    temp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = temp
                else:
                    is_random_next = False
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # 把tokens_a & tokens_b加入到按照bert的风格，即以[CLS]tokens_a[SEP]tokens_b[SEP]的形式，结合到一起，作为最终的tokens; 也带上segment_ids，前面部分segment_ids的值是0，后面部分的值是1.
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                # 创建masked LM的任务的数据 Creates the predictions for the masked LM objective
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words)
                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels}
                instances.append(instance)
            current_chunk = []  # 清空当前块
            current_length = 0  # 重置当前文本块的长度
        i += 1  # 接着文档中的内容往后看
    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    mask_indices = sorted(random.sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.choice(vocab_list)
        masked_token_labels.append(MaskedLmInstance(index=index, label=tokens[index]))
        tokens[index] = masked_token
    assert len(masked_token_labels) <= num_to_mask
    masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_token_labels]
    masked_labels = [p.label for p in masked_token_labels]
    return tokens, mask_indices, masked_labels


def create_training_instances(input_file, tokenizer, max_seq_len, short_seq_prob,
                              masked_lm_prob, max_predictions_per_seq):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    f = open(input_file, 'r')
    lines = f.readlines()
    pbar = ProgressBar(n_total=len(lines), desc='read data')
    for line_cnt, line in enumerate(lines):
        line = line.strip()
        # Empty lines are used as document delimiters
        if not line:
            all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
            all_documents[-1].append(tokens)
        pbar(step=line_cnt)
    print(' ')
    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    random.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    pbar = ProgressBar(n_total=len(all_documents), desc='create instances')
    for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_len, short_seq_prob,
                masked_lm_prob, max_predictions_per_seq, vocab_words))
        pbar(step=document_index)
    print(' ')
    ex_idx = 0
    while ex_idx < 5:
        instance = instances[ex_idx]
        logger.info("-------------------------Example-----------------------")
        logger.info(f"id: {ex_idx}")
        logger.info(f"tokens: {' '.join([str(x) for x in instance['tokens']])}")
        logger.info(f"masked_lm_labels: {' '.join([str(x) for x in instance['masked_lm_labels']])}")
        logger.info(f"segment_ids: {' '.join([str(x) for x in instance['segment_ids']])}")
        logger.info(f"masked_lm_positions: {' '.join([str(x) for x in instance['masked_lm_positions']])}")
        logger.info(f"is_random_next : {instance['is_random_next']}")
        ex_idx += 1
    random.shuffle(instances)
    return instances


def main():
    parser = ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--vocab_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    parser.add_argument('--data_name', default='albert', type=str)
    parser.add_argument("--do_data", default=False, action='store_true')
    parser.add_argument("--do_split", default=False, action='store_true')
    parser.add_argument("--do_lower_case", default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--line_per_file", default=1000000000, type=int)
    parser.add_argument("--file_num", type=int, default=10,
                        help="Number of dynamic masking to pregenerate (with different masks)")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,  # 128 * 0.15
                        help="Maximum number of tokens to mask in each sequence")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.data_dir = Path(args.data_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir +"pregenerate_training_data.log")
    logger.info("pregenerate training data parameters:\n %s", args)
    tokenizer = BertTokenizer(vocab_file=args.vocab_path, do_lower_case=args.do_lower_case)

    # split big file
    if args.do_split:
        corpus_path = args.data_dir / "corpus/corpus.txt"
        split_save_path = args.data_dir / "/corpus/train"
        if not split_save_path.exists():
            split_save_path.mkdir(exist_ok=True)
        line_per_file = args.line_per_file
        command = f'split -a 4 -l {line_per_file} -d {corpus_path} {split_save_path}/shard_'
        os.system(f"{command}")

    # generator train data
    if args.do_data:
        data_path = args.data_dir / "corpus/train"
        files = sorted([f for f in data_path.parent.iterdir() if f.exists() and '.txt' in str(f)])
        for idx in range(args.file_num):
            logger.info(f"pregenetate {args.data_name}_file_{idx}.json")
            save_filename = data_path / f"{args.data_name}_file_{idx}.json"
            num_instances = 0
            with save_filename.open('w') as fw:
                for file_idx in range(len(files)):
                    file_path = files[file_idx]
                    file_examples = create_training_instances(input_file=file_path,
                                                              tokenizer=tokenizer,
                                                              max_seq_len=args.max_seq_len,
                                                              short_seq_prob=args.short_seq_prob,
                                                              masked_lm_prob=args.masked_lm_prob,
                                                              max_predictions_per_seq=args.max_predictions_per_seq)
                    file_examples = [json.dumps(instance) for instance in file_examples]
                    for instance in file_examples:
                        fw.write(instance + '\n')
                        num_instances += 1
            metrics_file = data_path / f"{args.data_name}_file_{idx}_metrics.json"
            print(f"num_instances: {num_instances}")
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()
