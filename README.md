# OFA

This is the repository for the pipeline of **O**ne **F**or **A**ll Framework, which aims to find **a good initialization of subword embeddings** when we want to adapt a monolingual or multilingual PLM to many languages. The framework optionally applies matrix factorization to the original PLM subword embeddings and replaces the new subword embeddings with two smaller matrices, which can largely reduce the number of parameters. Therefore, the OFA framework can boost efficient **large-scale multilingual continued pretraining**, which is especially helpful to a limited computation budget. Some of the code is based on [Glot500](https://github.com/cisnlp/Glot500), [WECHSEL](https://github.com/CPJKU/wechsel) and [FOCUS](https://github.com/konstantinjdobler/focus).  

Paper on arXiv: https://arxiv.org/abs/2311.08849  

```
.
├── README.md
├── evaluation
│   ├── retrieval
│   │   ├── bible_lang_list.txt
│   │   ├── evaluate_retrieval_bible.py
│   │   ├── evaluate_retrieval_bible_roberta.sh
│   │   ├── evaluate_retrieval_bible_xlm.sh
│   │   ├── evaluate_retrieval_tatoeba.py
│   │   ├── evaluate_retrieval_tatoeba_roberta.sh
│   │   ├── evaluate_retrieval_tatoeba_xlm.sh
│   │   └── tatoeba_lang_list.txt
│   ├── tagging
│   │   ├── evaluate_ner.py
│   │   ├── evaluate_ner.sh
│   │   ├── evaluate_ner_xlmr.sh
│   │   ├── evaluate_pos.py
│   │   ├── evaluate_pos.sh
│   │   ├── evaluate_pos_xlmr.sh
│   │   ├── ner_lang_list.txt
│   │   ├── pos_lang_list.txt
│   │   ├── run_tag.py
│   │   └── utils_tag.py
│   └── taxi1500
│       ├── evaluate.py
│       ├── evaluate.sh
│       ├── evaluate_xlmr.sh
│       └── texi1500_lang_list.txt
├── model_loader_extra.py
├── modeling_roberta_extra.py
├── modeling_xlmr_extra.py
├── ofa
│   ├── __init__.py
│   ├── ofa.py
│   ├── random_init.py
│   ├── run_ofa.bash
│   └── utils.py
├── requirements.txt
├── run_extra.py
├── train_bash_roberta.sh
└── train_bash_xlm_roberta.sh
```

## Subword Embedding Initialization

Initializing the subword embeddings using the OFA framework:

```
cd ofa
bash run_ofa.bash
```

This will create embedding matrices for the subwords in the target tokenizer under four different dimensions: \[100, 200, 400, 768\]. The embedding initialization is based on the vocabulary of the source and target tokenizer, the embedding layer of the source model, and the external multilingual embeddings. The multilingual word embeddings used in OFA can be downloaded [here](https://github.com/cisnlp/colexificationnet).

To randomly initialize the unseen subword embeddings, run the following code:

```
cd ofa
python random_init.py
```

## Continued Pretraining

We use the [Glot500-c](https://github.com/cisnlp/Glot500) corpus for continued-pretraining our models. The dataset contains more than 500 languages.

To continued-pretrain the model initialized with OFA (RoBERTa as the source model, i.e., monolingual as source), run:

```
bash train_bash_roberta.sh
```

To continued-pretrain the model initialized with OFA (XLM-R as the source model, i.e., multilingual as source), run:

```
bash train_bash_xlm_roberta.sh
```

You can change the .sh files for specifying ```--num_primitive``` with the latent embedding dimensions you want to use (in \[100, 200, 400, 768\]), ```--use_initialization``` with True and ```--random_initialization``` with False if you use OFA framework to initialize and with True if you use random initialization.


## Model Loading

We release our models on Huggingface, you can download [ofa-multi-100](https://huggingface.co/yihongLiu/ofa-multi-100), [ofa-multi-200](https://huggingface.co/yihongLiu/ofa-multi-200), [ofa-multi-400](https://huggingface.co/yihongLiu/ofa-multi-400) and [ofa-multi-768](https://huggingface.co/yihongLiu/ofa-multi-768). The current HuggingFace Transformer does not support the model architecture **except for ofa-multi-768**.  


To use **ofa-multi-768**, you could do something like the following, as its architecture is XLMRobertaForMaskedLM that HuggingFace supports:

```python
>>> from transformers import pipeline
>>> MODEL_PATH = 'your_saved_model_path'
>>> mask_filler = pipeline('fill-mask', model=MODEL_PATH)
>>> mask_filler("Hello I'm a <mask> model.", tok_k=3)
``` 

or

```python
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer

MODEL_PATH = 'your_saved_model_path'

model = XLMRobertaForMaskedLM.from_pretrained(MODEL_PATH)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)


text = "Hello I'm a <mask> model."
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()


for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

``` 

To use models **with smaller embedding dimensions**, you could do something like the following:


```python

# you have to import the architecture
from modeling_xlmr_extra import XLMRobertaAssembledForMaskedLM
from transformers import XLMRobertaTokenizer

MODEL_PATH = 'your_saved_model_path'

model = XLMRobertaAssembledForMaskedLM.from_pretrained(MODEL_PATH)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)


text = "Hello I'm a <mask> model."
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()


for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

``` 


## Evaluation

### Dataset Preparation

Please refer to [Glot500](https://github.com/cisnlp/Glot500) for downloading the datasets used for evaluation.

### Sentence Retrieval - Bible

For SR-B, first go to evaluation/retrieval'.  


If you want to evaluate the ofa-mono-xxx models, run:
```
bash evaluate_retrieval_bible_roberta.sh
```


If you want to evaluate the ofa-multi-xxx models, run:
```
bash evaluate_retrieval_bible_xlm.sh
```


### Sentence Retrieval - Tatoeba

For SR-T, first go to evaluation/retrieval'.  


If you want to evaluate the ofa-mono-xxx models, run:
```
bash evaluate_retrieval_tatoeba_roberta.sh
```


If you want to evaluate the ofa-multi-xxx models, run:
```
bash evaluate_retrieval_tatoeba_xlm.sh
```


### Text Classification - Taxi1500

First go to evaluation/taxi1500'.  


If you want to evaluate the ofa-mono-xxx models, run:
```
bash evaluate.sh
```


If you want to evaluate the ofa-multi-xxx models, run:
```
bash evaluate_xlmr.sh
```


### Named Entity Recognition

For NER, first go to evaluation/tagging'.  


If you want to evaluate the ofa-mono-xxx models, run:
```
bash evaluate_ner.sh
```


If you want to evaluate the ofa-multi-xxx models, run:
```
bash evaluate_ner_xlmr.sh
```

### Part-Of-Speech Tagging

For POS, first go to evaluation/tagging'.  


If you want to evaluate the ofa-mono-xxx models, run:
```
bash evaluate_pos.sh
```


If you want to evaluate the ofa-multi-xxx models, run:
```
bash evaluate_pos_xlmr.sh
```


## Citation

If you find our code, model, or data useful for your research, please considering citing:

```
@article{liu2023ofa,
 title={OFA: A Framework of Initializing Unseen Subword Embeddings for Efficient Large-scale Multilingual Continued Pretraining}
 author={Liu, Yihong and Lin, Peiqin and Wang, Mingyang and Sch{\"u}tze, Hinrich},
 journal={arXiv preprint arXiv:2311.08849},
 year={2023}
}
```

or

```
@inproceedings{imanigooghari-etal-2023-glot500,
	title        = {Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages},
	author       = {ImaniGooghari, Ayyoob  and Lin, Peiqin  and Kargaran, Amir Hossein  and Severini, Silvia  and Jalili Sabet, Masoud  and Kassner, Nora  and Ma, Chunlan  and Schmid, Helmut  and Martins, Andr{\'e}  and Yvon, Fran{\c{c}}ois  and Sch{\"u}tze, Hinrich},
	year         = 2023,
	month        = jul,
	booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
	publisher    = {Association for Computational Linguistics},
	address      = {Toronto, Canada},
	pages        = {1082--1117},
	url          = {https://aclanthology.org/2023.acl-long.61}
}
```

## Acknowledgements

This repository is built on top of [transformers](https://github.com/huggingface/transformers), [xtreme](https://github.com/google-research/xtreme), [Glot500](https://github.com/cisnlp/Glot500), [WECHSEL](https://github.com/CPJKU/wechsel) and [FOCUS](https://github.com/konstantinjdobler/focus).
