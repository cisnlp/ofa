from transformers import AutoModelForMaskedLM, RobertaConfig
from modeling_xlmr_extra import *
from modeling_roberta_extra import *
from torch import nn
import numpy as np
import torch


def get_embedding_path(base_model_type, dim, only_eng_vocab=False, random_initialization=False):

    assert base_model_type in ['roberta-base', 'xlm-roberta-base', 'xlm-roberta-large']

    # if we use random_initialization we have to make sure the model is in full dimension
    if random_initialization:
        assert dim == 768 or dim == 1024

    embedding_path = ''

    if base_model_type == 'roberta-base':
        embedding_path += 'roberta'
    elif base_model_type == 'xlm-roberta-base':
        embedding_path += 'xlm'
    else:
        embedding_path += 'xlm_large'

    if random_initialization:
        embedding_path += '_rand'
    else:
        if only_eng_vocab:
            embedding_path += '_eng'
        else:
            embedding_path += '_all'
        embedding_path += f"_{str(dim)}"

    return embedding_path


def load_assembled_model(base_model_type, dim, only_eng_vocab=False,
                         path='/mounts/data/proj/yihong/newhome/OFA/stored_factorization/updated',
                         random_initialization=False):

    assert base_model_type in ['roberta-base', 'xlm-roberta-base', 'xlm-roberta-large']

    # if we use random_initialization we have to make sure the model is in full dimension
    if random_initialization:
        assert dim == 768 or dim == 1024

    factorize = True
    # all available reduced dimension
    if base_model_type != 'xlm-roberta-large':
        assert dim in [100, 200, 400, 768]
        if dim == 768:
            # for this we do not perform dimension reduction
            factorize = False

    else:
        assert dim in [100, 200, 400, 800, 1024]
        if dim == 1024:
            factorize = False

    embedding_path = ''

    # loading the base model
    if base_model_type == 'roberta-base':
        embedding_path += 'roberta'
        model = AutoModelForMaskedLM.from_pretrained('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base')
    elif base_model_type == 'xlm-roberta-base':
        embedding_path += 'xlm'
        model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
    else:
        embedding_path += 'xlm_large'
        model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-large')
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-large')

    if random_initialization:
        embedding_path += '_rand'
    else:
        if only_eng_vocab:
            embedding_path += '_eng'
        else:
            embedding_path += '_all'
        embedding_path += f"_{str(dim)}"

    primitive_embeddings = None

    if factorize:
        primitive_embeddings = np.load(f"{path}/{embedding_path}/primitive_embeddings.npy")
        config.num_primitive = dim

    target_matrix = np.load(f"{path}/{embedding_path}/target_matrix.npy")
    config.vocab_size = len(target_matrix)
    if factorize:
        if base_model_type == 'roberta-base':
            assembled_model = RobertaAssembledForMaskedLM(config=config)
        else:
            assembled_model = XLMRobertaAssembledForMaskedLM(config=config)

        # copy the encoder
        assembled_model.roberta.encoder = model.roberta.encoder

        # initializing / copying some embeddings
        assert np.shape(target_matrix)[1] == np.shape(primitive_embeddings)[0]
        assembled_model.roberta.embeddings.primitive_embeddings.weight.data = torch.from_numpy(primitive_embeddings.T)
        assembled_model.roberta.embeddings.target_coordinates.weight.data = torch.from_numpy(target_matrix)

        # regarding embeddings
        assembled_model.roberta.embeddings.token_type_embeddings = model.roberta.embeddings.token_type_embeddings
        assembled_model.roberta.embeddings.position_embeddings = model.roberta.embeddings.position_embeddings
        assembled_model.roberta.embeddings.LayerNorm = model.roberta.embeddings.LayerNorm
        assembled_model.roberta.embeddings.dropout = model.roberta.embeddings.dropout

        # regarding lm head
        assembled_model.lm_head.dense = model.lm_head.dense
        assembled_model.lm_head.layer_norm = model.lm_head.layer_norm
    else:
        assembled_model = model
        assembled_model.config.vocab_size = len(target_matrix)
        assembled_model.resize_token_embeddings(len(target_matrix))
        assembled_model.get_input_embeddings().weight.data = torch.from_numpy(target_matrix)

    return assembled_model


def print_model_stats(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
