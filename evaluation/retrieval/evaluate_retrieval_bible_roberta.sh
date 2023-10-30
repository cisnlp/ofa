#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MODEL=${1:-cis-lmu/glot500-base}
MODEL="roberta-base"
GPU=${2:-3}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL_TYPE="roberta"

MAX_LENGTH=512
LC=""
BATCH_SIZE=128
DIM=768
NLAYER=12
LAYER=7
NUM_PRIMITIVE=100
checkpoint_num=0

# set checkpoint_num=0 to use models without continue pretraining

# set use_initialization "true" to use models with initialization

# set random_initialization "true" to use models with random initialization for embeddings of new words

# only comment init_checkpoint (i.e., set it None) if you want to the huggingface pretrained model, e.g., roberta
# otherwise always keep it uncommented

DATA_DIR="/mounts/data/proj/linpq/datasets/retrieval_bible_test/"
OUTPUT_DIR="/mounts/data/proj/ayyoobbig/ofa/evaluation/retrieval/bible/"
tokenized_dir="/mounts/data/proj/ayyoobbig/ofa/evaluation/retrieval/bible_tokenized"
init_checkpoint="/mounts/data/proj/ayyoobbig/ofa/trained_models/updated/"

python -u evaluate_retrieval_bible.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --embed_size $DIM \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_LENGTH \
    --num_layers $NLAYER \
    --dist cosine $LC \
    --specific_layer $LAYER \
    --only_eng_vocab "false" \
    --use_initialization "true" \
    --random_initialization "false" \
    --checkpoint_num $checkpoint_num \
    --num_primitive $NUM_PRIMITIVE \
    --tokenized_dir $tokenized_dir \
    --init_checkpoint $init_checkpoint

