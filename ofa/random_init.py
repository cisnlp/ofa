from utils import *
from transformers import AutoModelForMaskedLM, AutoTokenizer


def run_random_init(source_model_name, source_tokenizer, target_tokenizer, source_embeddings):

    overlapping_token_mapping = get_overlapping_tokens(target_tokenizer, source_tokenizer, fuzzy_search=True)

    mean, std = source_embeddings.mean(0), source_embeddings.std(0)
    # initialize using the source_embeddings
    random_fallback_matrix = \
        np.random.RandomState(114514).normal(mean, std, (len(target_tokenizer.vocab), source_embeddings.shape[1]))

    random_fallback_matrix = random_fallback_matrix.astype(np.float32)

    overlapping_tokens = {}
    for overlapping_token, (target_vocab_idx, source_vocab_idx) in overlapping_token_mapping.items():
        print(target_vocab_idx, source_vocab_idx)
        overlapping_tokens[overlapping_token] = target_vocab_idx
        random_fallback_matrix[target_vocab_idx] = source_embeddings[source_vocab_idx]

    print(f"Num overlapping tokens: {len(overlapping_tokens)}")
    # the subword tokens that need to be initialized
    additional_tokens = {token: idx for token, idx in target_tokenizer.get_vocab().items()
                         if token not in overlapping_tokens}
    print(f"Num additional tokens: {len(additional_tokens)}")
    assert len(overlapping_tokens) + len(additional_tokens) == len(target_tokenizer)

    final_target_matrix = random_fallback_matrix
    print(np.shape(final_target_matrix))
    model_name = ''
    if source_model_name == 'xlm-roberta-base':
        model_name += 'xlm_'
    elif source_model_name == 'roberta-base':
        model_name += 'roberta_'
    else:
        raise ValueError("Models other than xlm-r or roberta are not considered!")

    model_name += 'rand'

    model_path = '/mounts/data/proj/yihong/newhome/OFA/stored_factorization/updated/' + model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    np.save(f"{model_path}/target_matrix.npy", final_target_matrix)


source_model_name = 'roberta-base'

# loading tokenizers and source-model embeddings
source_tokenizer = AutoTokenizer.from_pretrained(source_model_name)  # source tok
target_tokenizer = AutoTokenizer.from_pretrained('cis-lmu/glot500-base')  # target tok

source_model = AutoModelForMaskedLM.from_pretrained(source_model_name)

source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
assert len(source_tokenizer) == len(source_embeddings)
run_random_init(source_model_name, source_tokenizer, target_tokenizer, source_embeddings)
print('done!')