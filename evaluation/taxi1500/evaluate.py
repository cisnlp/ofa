import sys
sys.path.append('/mounts/data/proj/ayyoobbig/ofa/')
import argparse
import glob
import logging
import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    #updated below
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

# update below
import copy
from modeling_xlmr_extra import XLMRobertaAssembledForSequenceClassification
from modeling_roberta_extra import RobertaAssembledForSequenceClassification
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


MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaAssembledForSequenceClassification),
    # update below
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaAssembledForSequenceClassification)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def read_data(file):
    print('Preparing dataset...')
    df = pd.read_csv(file, delimiter='\t', header=None,on_bad_lines='skip', engine='python')
    df.columns = ['id', 'label', 'verse']
    print('Number of sentences: {:,}\n'.format(df.shape[0]))
    df['label'].replace({'Recommendation': int(0), 'Faith': int(1), 'Violence': int(2),
                         'Grace': int(3), 'Sin': int(4), 'Description': int(5)},inplace=True)

    sentences = df.verse.values
    labels = df.label.values

    return sentences, labels


def encode(tokenizer, sentences, labels):
    print('Loading model tokenizer...')
    input_ids = []
    attention_masks = []
    for sent in sentences:
        sent = str(sent)
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=100,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def F1_score(preds, labels):
    pre = np.argmax(preds, axis=1)
    f1 = f1_score(labels, pre, average='macro')
    return f1


def train(args, model, optimizer, scheduler, train_dataloader, validation_dataloader):
    training_stats = []
    best_f1 = -1.0
    epochs = args.epochs
    device = args.device
    accum_iter = args.accum_iter
    output_dir = args.output_dir

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_train_loss = 0
        total_train_f1 = 0

        # training
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            with torch.set_grad_enabled(True):
                (loss, logits) = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=False)

                logits = logits.detach().cpu().numpy()
                b_label_ids = b_labels.to('cpu').numpy()

                total_train_loss += loss.item()
                total_train_f1 += F1_score(logits, b_label_ids)
                loss = loss / accum_iter
                loss.backward()
                if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_dataloader)):
                    optimizer.step()
                    scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_f1 = total_train_f1 / len(train_dataloader)
        print('train_f1: ', avg_train_f1)

        # evaluation
        model.eval()

        label_ids = []
        predictions = []

        total_eval_loss = 0
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                (loss, logits_current) = model(b_input_ids,
                                               attention_mask=b_input_mask,
                                               labels=b_labels,
                                               return_dict=False)
            total_eval_loss += loss.item()

            predictions += np.argmax(logits_current.detach().cpu().numpy(), axis=1).reshape(-1).tolist()
            label_ids += b_labels.to('cpu').numpy().reshape(-1).tolist()

        # f1 score and accuracy of the whole validation set
        total_eval_f1_score = f1_score(label_ids, predictions, average='macro')
        total_eval_accuracy = accuracy_score(label_ids, predictions)

        avg_val_accuracy = total_eval_accuracy
        avg_val_f1 = total_eval_f1_score
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print('avg_val_accuracy:', avg_val_accuracy, 'avg_val_f1: ', avg_val_f1, 'avg_val_loss', avg_val_loss)
        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            torch.save(model, f"{output_dir}/pytorch_model.bin")
            print('Better validation f1!')
        else:
            pass
        print('val_f1: ', avg_val_f1)

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Train Loss': avg_train_loss,
                'Val Loss': avg_val_loss,
                'Val Accu.': avg_val_accuracy,
                'val F1 ': avg_val_f1,
            }
        )

        print("")
        print("Training complete!")
    pd.set_option('display.precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)


def test(args, model, test_dataloader):
    device = args.device
    model.eval()

    predictions, true_labels = [], []

    # Predict
    result = {}
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]

        predictions += np.argmax(logits.detach().cpu().numpy(), axis=1).reshape(-1).tolist()
        true_labels += b_labels.to('cpu').numpy().reshape(-1).tolist()

    # calculate
    result['accuracy'] = accuracy_score(true_labels, predictions)
    result['f1'] = f1_score(true_labels, predictions, average='macro')

    return result


def run(args):
    # the below is the model loading part
    num_labels = args.num_labels
    args.model_type = args.model_type.lower()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    config_class, model_class, tokenizer_class, assembled_model_class = MODEL_CLASSES[args.model_type]

    # if the model is not initialized from a stored checkpoint
    if not args.init_checkpoint:
        config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        print(config)
        print(args.model_name_or_path)
        config.output_attentions = False
        config.output_hidden_states = False
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config)
    else:
        tokenizer_class = XLMRobertaTokenizer
        if args.checkpoint_num == 0 and args.use_initialization:
            config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
            config.output_attentions = False
            config.output_hidden_states = False
            print(f"Loading model from {args.embedding_path} without continue pretraining!")

            # the masked language model
            loaded_model = load_assembled_model(args.model_name_or_path, args.num_primitive, args.only_eng_vocab,
                                                args.embedding_path, args.random_initialization)
            config.num_primitive = args.num_primitive
            config.vocab_size = loaded_model.config.vocab_size

            model = assembled_model_class(config)
            model.roberta = loaded_model.roberta
            config = model.config
            # the following is the glot500 tokenizer
            tokenizer = XLMRobertaTokenizer.from_pretrained('cis-lmu/glot500-base')
            assert config.vocab_size == len(tokenizer)
            # change the model class for later using
            if args.num_primitive != 768:
                model_class = assembled_model_class
        else:
            config = config_class.from_pretrained(args.init_checkpoint)
            config.num_labels = num_labels
            config.output_attentions = False
            config.output_hidden_states = False
            tokenizer = XLMRobertaTokenizer.from_pretrained(
                args.init_checkpoint, do_lower_case=args.do_lower_case)
            if args.use_initialization and args.num_primitive != 768:
                model = assembled_model_class(config)
                print(f"Loading model from {args.init_checkpoint}/pytorch_model.bin")
                state_dict = torch.load(args.init_checkpoint + '/pytorch_model.bin')
                model.load_state_dict(state_dict, strict=False)
                uninitialized_params = {k: v for k, v in model.state_dict().items() if k not in state_dict}
                for k, v in uninitialized_params.items():
                    print(f"{k} is randomly initialized!")
                model_class = assembled_model_class
            else:
                model = model_class.from_pretrained(args.init_checkpoint, config=config)
    print(model)
    model.to(args.device)

    if args.do_train:
        train_file = f"{args.eng_data_dir}/eng_train.tsv"
        train_verse, train_label = read_data(train_file)
        train_input_ids, train_attention_masks, train_labels = encode(tokenizer, train_verse, train_label)
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

        dev_file = f"{args.eng_data_dir}/eng_dev.tsv"
        dev_verse, dev_label = read_data(dev_file)
        dev_input_ids, dev_attention_masks, dev_labels = encode(tokenizer, dev_verse, dev_label)
        val_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=args.train_batch_size,  # Trains with this batch size.
        )

        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=args.train_batch_size  # Evaluate with this batch size.
        )

        learning_rate = 1e-5
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        train(args, model, optimizer, scheduler, train_dataloader, validation_dataloader)

    best_checkpoint = f"{args.output_dir}/pytorch_model.bin"
    model = torch.load(best_checkpoint)
    model.to(args.device)
    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")

    # performing testing for all languages
    with open(output_test_results_file, "w") as result_writer:
        for lang in args.predict_langs:
            print('--------')
            print(f"Testing on {lang} ...")
            test_file = f"{args.test_data_dir}/{lang}_test.tsv"
            test_verse, test_label = read_data(test_file)
            test_input_ids, test_attention_masks, test_labels = encode(tokenizer, test_verse, test_label)
            test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

            test_dataloader = DataLoader(
                test_dataset,
                sampler=SequentialSampler(test_dataset),
                batch_size=args.test_batch_size,
            )

            result = test(args, model, test_dataloader)

            # Save results
            result_writer.write("=====================\nlanguage={}\n".format(lang))
            for key in sorted(result.keys()):
                result_writer.write("{} = {}\n".format(key, str(result[key])))
                print("{} = {}\n".format(key, str(result[key])))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint", default=None, type=str)

    ## training parameters
    parser.add_argument("--no_cuda", default=False, type=bool_flag)
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--accum_iter", default=2, type=int)
    parser.add_argument("--num_labels", default=6, type=int)
    parser.add_argument("--do_lower_case", action="store_true", 
                        help="Set this flag if you are using an uncased model.")


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

    parser.add_argument("--eng_data_dir", type=str,
                        default='/mounts/data/proj/chunlan/Taxi1500_data/Glot500_data/eng_data')
    parser.add_argument("--test_data_dir", type=str,
                        default='/mounts/data/proj/chunlan/Taxi1500_data/Glot500_data/lrs_test2')

    args = parser.parse_args()

    args.predict_langs = []

    files = os.listdir(args.test_data_dir)
    for file in files:
        lang = file.split('_')[0]
        args.predict_langs.append(lang)

    # updated below
    if args.init_checkpoint:
        # in this case, the tokenizer should all be glot500 tokenizer
        if args.use_initialization:
            embedding_name = get_embedding_path(args.model_name_or_path, args.num_primitive, args.only_eng_vocab,
                                                args.random_initialization)
            args.init_checkpoint += f"LM_ofa_{embedding_name}/checkpoint-{str(args.checkpoint_num)}"
        else:
            args.init_checkpoint += f"{args.model_name_or_path}/checkpoint-{str(args.checkpoint_num)}"

    args.output_dir += embedding_name if args.use_initialization else args.model_name_or_path
    args.output_dir += f"/checkpoint-{str(args.checkpoint_num)}" if args.init_checkpoint else ''

    args.model_save_name = embedding_name if args.use_initialization else args.model_name_or_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        args.do_train = True
    else:
        args.do_train = False
    args.log_file = args.output_dir + '/train.log'

    run(args)


if __name__ == "__main__":
    main()
