#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append('/mounts/data/proj/ayyoobbig/ofa/')
import argparse
import os

from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, XLMTokenizer, AutoTokenizer

from run_tag import run

from model_loader_extra import load_assembled_model, get_embedding_path


# update below
FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def tokenize_preprocess(args):
    def _preprocess_one_file(infile, outfile, idxfile, tokenizer, max_seq_len):
        if not os.path.exists(infile):
            print(f'{infile} not exists')
            return 0
        subword_len_counter = idx = 0
        special_tokens_count = 3 if isinstance(tokenizer, XLMRobertaTokenizer) or isinstance(tokenizer, RobertaTokenizer) else 2
        max_seq_len = max_seq_len - special_tokens_count
        with open(infile, "rt") as fin, open(outfile, "w") as fout, open(idxfile, "w") as fidx:
            for line in fin:
                line = line.strip()
                if not line:
                    fout.write('\n')
                    fidx.write('\n')
                    idx += 1
                    subword_len_counter = 0
                    continue

                items = line.split()
                token = items[0].strip()
                if len(items) == 2:
                    label = items[1].strip()
                else:
                    label = 'O'
                current_subwords_len = len(tokenizer.tokenize(token))

                if (current_subwords_len == 0 or current_subwords_len > max_seq_len) and len(token) != 0:
                    token = tokenizer.unk_token
                    current_subwords_len = 1

                if (subword_len_counter + current_subwords_len) > max_seq_len:
                    fout.write(f"\n{token}\t{label}\n")
                    fidx.write(f"\n{idx}\n")
                    subword_len_counter = current_subwords_len
                else:
                    fout.write(f"{token}\t{label}\n")
                    fidx.write(f"{idx}\n")
                    subword_len_counter += current_subwords_len
        return 1

    model_type = args.model_type
    TOKENIZERS = {
        'bert': BertTokenizer,
        'xlm': XLMTokenizer,
        'xlmr': XLMRobertaTokenizer,
        'roberta': RobertaTokenizer
    }

    # doing tokenization
    if not args.init_checkpoint:
        tokenizer = TOKENIZERS[model_type].from_pretrained(args.model_name_or_path, 
            do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)
        print(f"Loading {model_type} tokenizer ...")
    else:
        print(f"Loading glot500 tokenizer ...")
        if args.checkpoint_num == 0 and args.use_initialization:
            # loading from AutoTokenizer is very slow
            tokenizer = XLMRobertaTokenizer.from_pretrained('cis-lmu/glot500-base')
        else:
            tokenizer = XLMRobertaTokenizer.from_pretrained(args.init_checkpoint, do_lower_case=args.do_lower_case)

    for lang in args.predict_langs:
        out_dir = os.path.join(args.tokenized_dir, lang)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if lang == 'eng_Latn':
            files = ['dev', 'test', 'train']
        else:
            files = ['dev', 'test']
        for file in files:
            infile = os.path.join('/'.join(args.data_dir.split('/')[:-1]), f'{file}-{lang}.tsv')
            outfile = os.path.join(out_dir, "{}".format(file))
            idxfile = os.path.join(out_dir, "{}.idx".format(file))
            if os.path.exists(outfile) and os.path.exists(idxfile):
                print(f'{outfile} and {idxfile} exist')
            else:
                code = _preprocess_one_file(infile, outfile, idxfile, tokenizer, args.max_seq_len)
                if code > 0:
                    print(f'finish preprocessing {outfile}')

def main(): 
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the NER/POS task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_len", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_predict_dev", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--few_shot", default=-1, type=int,
                        help="num of few-shot exampes")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best_checkpoint", action="store_true",
                        help="Save only the best checkpoint during training")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--train_langs", default="eng_Latn", type=str,
                        help="The languages in the training sets.")
    parser.add_argument("--log_file", type=str, default=None, help="log file")
    parser.add_argument("--eval_patience", type=int, default=-1, help="wait N times of decreasing dev score before early stop during training")

    # update below
    parser.add_argument("--num_primitive", type=int, default=768)
    parser.add_argument("--embedding_path", type=str, 
        default="/mounts/data/proj/yihong/newhome/OFA/stored_factorization/updated"
    )
    parser.add_argument("--only_eng_vocab", type=bool_flag, default=False)
    parser.add_argument("--use_initialization", type=bool_flag, default=True)
    parser.add_argument("--random_initialization", type=bool_flag, default=False)
    # when checkpoint number is zero, loading the model without continue training
    parser.add_argument("--checkpoint_num", type=int, default=180000)
    parser.add_argument(
    "--tokenized_dir",
    default=None,
    type=str,
    required=False,
    help="The input data dir in which the tokenized are stored (optional).",
)

    args = parser.parse_args()
    
    args.predict_langs = []
    with open('ner_lang_list.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            args.predict_langs.append(line.strip().split('\t')[0])

    #updated below
    if args.init_checkpoint:
        # in this case, the tokenizer should all be glot500 tokenizer
        if args.use_initialization:
            embedding_name = get_embedding_path(args.model_name_or_path, args.num_primitive, args.only_eng_vocab, args.random_initialization)
            args.init_checkpoint += f"LM_ofa_{embedding_name}/checkpoint-{str(args.checkpoint_num)}"
        else:
            args.init_checkpoint += f"{args.model_name_or_path}/checkpoint-{str(args.checkpoint_num)}"
        args.tokenized_dir += '_glot500/'
    else:
        args.tokenized_dir += f"_{args.model_type}/"

    if not os.path.exists(args.tokenized_dir):
        os.makedirs(args.tokenized_dir)

    args.output_dir += f"{embedding_name}/checkpoint-{str(args.checkpoint_num)}" if args.use_initialization else args.model_name_or_path
    # args.output_dir += f"_checkpoint-{str(args.checkpoint_num)}" if args.init_checkpoint else ''

    args.model_save_name = embedding_name if args.use_initialization else args.model_name_or_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        args.do_train = False
    args.log_file = args.output_dir + '/train.log'
    
    tokenize_preprocess(args)
    run(args)

if __name__ == "__main__":
    main()

