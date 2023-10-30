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
MODEL="xlm-roberta-base"
# MODEL="facebook/xlm-v-base"
GPU=${2:-2}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL_TYPE="xlmr"

NUM_PRIMITIVE=100
checkpoint_num=290000

OUTPUT_DIR="/mounts/data/proj/ayyoobbig/ofa/evaluation/texi1500/results/"
init_checkpoint="/mounts/data/proj/ayyoobbig/ofa/trained_models/updated/"

python -u evaluate.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs 40 \
    --only_eng_vocab "false" \
    --use_initialization "true" \
    --random_initialization "false" \
    --checkpoint_num $checkpoint_num \
    --num_primitive $NUM_PRIMITIVE \
    --init_checkpoint $init_checkpoint
