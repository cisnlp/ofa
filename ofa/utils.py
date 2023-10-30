import numpy as np
from gensim.models import KeyedVectors
# from tqdm.auto import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
from transformers import PreTrainedTokenizer
from collections import OrderedDict


BPE_WHITESPACE = "Ġ"
XLMR_WHITESPACE = "▁"


def get_token_standardization_func(input_tokenizer: PreTrainedTokenizer):
    """Standardize tokens from different tokenizers.
    Standard output format should be Unicode-like output for non-ASCII chars.
    Beginning of word tokens should be prefixed with a space.

    We have to use .decode() to get "standardized" tokens (e.g. BytePairBPE represents non-ASCII tokens non-UNIcode-like internally).
    But XLM-R's tokenizer removes leading whitespace from tokens when using .decode().
    Se we add those back in manually.
    """

    def decode(tokenizer: PreTrainedTokenizer, token_id: int):
        """For BPE tokenizer and fallback"""
        return tokenizer.decode(token_id)

    def replace_space(tokenizer: PreTrainedTokenizer, token_id: int):
        """For XLM-R tokenizer (sentencepiece-style)"""
        return tokenizer.convert_ids_to_tokens(token_id).replace(XLMR_WHITESPACE, " ")

    def wordpiece(tokenizer: PreTrainedTokenizer, token_id: int):
        """For wordpiece (e.g. BERT or mBERT)"""
        token = tokenizer.decode(token_id)
        if token.startswith("##"):
            return token[2:]
        else:
            return " " + token

    # simple heuristics to avoid false positive
    if (
        len([k for k in input_tokenizer.get_vocab().keys() if k[0] == XLMR_WHITESPACE])
        > 100
    ):
        standardize_token = replace_space
    # simple heuristics to avoid false positive
    elif len([k for k in input_tokenizer.get_vocab().keys() if k[:2] == "##"]) > 100:
        standardize_token = wordpiece
    else:
        standardize_token = decode

    return standardize_token


def get_overlapping_tokens(target_tokenizer: PreTrainedTokenizer,
                           source_tokenizer: PreTrainedTokenizer, fuzzy_search=True, fuzzy_whitespace=False):
    """
    :param target_tokenizer:
    :param source_tokenizer:
    :param fuzzy_search: whether fuzzy search (for determine overlapping subwords) is used
    :param fuzzy_whitespace:
    :return:
    """

    target_vocab = target_tokenizer.get_vocab()  # a dictionary
    source_vocab = source_tokenizer.get_vocab()

    # standardizing all the subwords for both the source and the target vocabulary
    standardize_token = get_token_standardization_func(source_tokenizer)
    source_vocab = {
        standardize_token(source_tokenizer, idx): idx
        for idx in sorted(source_vocab.values())
    }

    standardize_token = get_token_standardization_func(target_tokenizer)
    target_vocab = {
        standardize_token(target_tokenizer, idx): idx
        for idx in sorted(target_vocab.values())
    }

    # Determine overlapping tokens between source and target vocab
    exact_overlap = {
        k: (target_vocab[k], source_vocab[k])
        for k in set(target_vocab) & set(source_vocab)
    }
    # exact_overlap: a dict of {'token': (token_id_source, token_id_target)}

    if not fuzzy_search:
        return {
            target_tokenizer.convert_ids_to_tokens(v[0]): v
            for k, v in sorted(exact_overlap.items())
        }

    # We do a greedy search for additional overlapping tokens.
    # NOTE: source_vocab order is random, need to sort for consistent results
    lowercase_source_vocab = {k.lower(): v for k, v in sorted(source_vocab.items())}
    fuzzy_overlap = exact_overlap  # initialize the fuzzy_overlap

    for target_token, target_token_idx in sorted(target_vocab.items()):
        lowercase_target_token = target_token.lower()
        if fuzzy_overlap.get(target_token):  # if the target_token is already in the stored overlapped tokens, skip
            continue
        if lowercase_source_vocab.get(lowercase_target_token):
            # same token but just **different case** found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[lowercase_target_token],
            )
        elif fuzzy_whitespace and lowercase_source_vocab.get(
            " " + lowercase_target_token
        ):
            # same token with **extra whitespace** found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[" " + lowercase_target_token],
            )
        elif fuzzy_whitespace and lowercase_source_vocab.get(
            lowercase_target_token.lstrip()
        ):
            # lstrip remove the spaces before a string
            # same token without **extra whitespace** found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[lowercase_target_token.lstrip()],
            )
    return {
        target_tokenizer.convert_ids_to_tokens(v[0]): v
        for k, v in fuzzy_overlap.items()
    }


def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class WordEmbedding:
    """
    An interface to gensim Word2Vec model.
    """

    def __init__(self, model):
        self.model = model
        if isinstance(model, KeyedVectors):
            self.kind = "word2vec"
        else:
            raise ValueError(
                f"{model} seems to be a Word2Vec model."
            )

        # the following variables change after factorization is applied

    def get_words(self):
        return list(self.model.key_to_index.keys())

    def get_dimension(self):
        return self.model.vector_size

    def get_word_vector(self, word):
        return self.model.get_vector(word)

    def get_word_id(self, word):
        return self.model.get_index(word)


def get_subword_embeddings_in_word_embedding_space(tokenizer, model: WordEmbedding,
                                                   multilingual=True,
                                                   languages_considered=None,
                                                   max_n_word_vectors=None,
                                                   return_not_covered_set=False):
    """
    :param tokenizer: the pre-trained tokenizer
    :param model: the embeddings
    :param multilingual: whether the multilingual embedding is used, if yes, we should remove the iso-codes
    :param languages_considered: the set of the languages that the tokenizer is trained on (or needs to be considered)
    :param max_n_word_vectors: maximum number of vocabulary to be considered
    :param return_not_covered_set: whether return the set of subword ids not covered by the embeddings
    :return:
    """
    words = model.get_words()

    if max_n_word_vectors is None:
        max_n_word_vectors = len(words)

    # a dictionary to store the subword-word_list pair
    # where the word_list is a list to store the tokens from which the subword can be tokenized
    sources = {}

    # a set to store the subwords that are not tokenized from any tokens (useful to calculate the coverage)
    not_covered_subwords = set()

    # the subword embedding matrix
    embs_matrix = np.zeros((len(tokenizer), model.get_dimension()))

    # for each subword (id), create a empty list to store
    # the possible token (id) from which the subword can be tokenized
    embs = {subword_id: [] for subword_id in tokenizer.get_vocab().values()}

    for i, word in enumerate(words[:max_n_word_vectors]):
        if i % 100000 == 0:
            print(i)
        # total=max_n_word_vectors,
        # disable=not verbose,
        if multilingual:
            # if the word is from other languages
            if len(word) > 4 and word[3] == ':':
                assert len(word.split(':')) == 2
            else:
                word = f"eng:{word}"
            lang, word = word[:3], word[4:]
            word_ori = word  # the original word
            word = word.replace('$', ' ')
            # when the tokenizer languages are provided,
            # we should skip those tokens that do not belong to the language
            if languages_considered is not None:
                if lang not in languages_considered:
                    continue

        # ways to cover more subwords
        for tokenized in [
            tokenizer.encode(word, add_special_tokens=False),
            tokenizer.encode(' ' + word, add_special_tokens=False),
            # because word[0] will be ' ', and  word[1] will be the actual subword
            tokenizer.encode(f"{word[1].upper() + word[2:]}", add_special_tokens=False)
            if multilingual and word_ori[0] == '$' else '',
            tokenizer.encode(f" {word[0].upper() + word[1:]}", add_special_tokens=False)
            if multilingual and word_ori[0] != '$' else '',
        ]:
            for subword_id in set(tokenized):
                embs[subword_id].append(i)

    for i in range(len(embs_matrix)):
        if len(embs[i]) == 0:
            # to record the uncovered subword
            not_covered_subwords.add(i)
            continue

        # in OFA, we simply do the average
        weight = np.array([1.0 for _ in embs[i]])
        weight = weight / weight.sum()

        vectors = [model.get_word_vector(words[idx]) for idx in embs[i]]

        sources[tokenizer.convert_ids_to_tokens([i])[0]] = embs[i]
        embs_matrix[i] = (np.stack(vectors) * weight[:, np.newaxis]).sum(axis=0)

    # embs_matrix is initialized subword embedding from the multilingual word embeddings
    if return_not_covered_set:
        return embs_matrix, sources, not_covered_subwords
    else:
        return embs_matrix, sources


# this function use the initialized source and target subword embeddings to initialize the target PLM embedding matrix
def create_target_embeddings(
    source_subword_embeddings,
    target_subword_embeddings,
    source_tokenizer,
    target_tokenizer,
    source_matrix,
    target_matrix=None,
    overlapping_tokens=None,
    additional_tokens=None,
    neighbors=10,
    temperature=0.1,
):
    """
    :param source_subword_embeddings: initialized source subword embeddings
    :param target_subword_embeddings: initialized source subword embeddings
    :param source_tokenizer:
    :param target_tokenizer:
    :param source_matrix: the source-language PLM subword embedding
    :param target_matrix: the initialized subword embedding for target languages
    :param overlapping_tokens: the overlapped tokens in source and target-language tokenizers
    :param additional_tokens: the subword tokens that need to be initialized
    :param neighbors: number of neighbors
    :param temperature:
    :return:
    """
    def get_n_closest(token_id, similarities, top_k):
        if (target_subword_embeddings[token_id] == 0).all():
            return None

        best_indices = np.argpartition(similarities, -top_k)[-top_k:]
        best_tokens = source_tokenizer.convert_ids_to_tokens(best_indices)

        best = sorted(
            [
                (token, similarities[idx])
                for token, idx in zip(best_tokens, best_indices)
            ],
            key=lambda x: -x[1],
        )

        return best

    source_vocab = source_tokenizer.vocab

    # all embeddings are initialized to zero first if no overlapped subword tokens are considered
    if target_matrix is None:
        target_matrix = np.zeros((len(target_tokenizer), source_matrix.shape[1]), dtype=source_matrix.dtype)
    else:
        # this makes sure that the shape of embeddings match
        assert np.shape(target_matrix) == (len(target_tokenizer), source_matrix.shape[1])

    mean, std = source_matrix.mean(0), source_matrix.std(0)
    random_fallback_matrix = \
        np.random.RandomState(114514).normal(mean, std, (len(target_tokenizer.vocab), source_matrix.shape[1]))

    batch_size = 1024
    n_matched = 0

    not_found = {}
    found = {}

    how_many_kept = 0
    how_many_updated = 0
    how_many_randomly_updated = 0

    for i in range(int(math.ceil(len(target_matrix) / batch_size))):
        # use a batch to perform the similarity, otherwise a lot of memory will be consumed
        start, end = (
            i * batch_size,
            min((i + 1) * batch_size, len(target_matrix)),
        )

        similarities = cosine_similarity(target_subword_embeddings[start:end], source_subword_embeddings)

        # here the token_id is actually the index of the target-language PLM embeddings
        for token_id in range(start, end):
            if target_tokenizer.convert_ids_to_tokens(token_id) in overlapping_tokens:
                # we only need to initialize additional_tokens
                found[token_id] = target_tokenizer.convert_ids_to_tokens(token_id)
                n_matched += 1
                how_many_kept += 1
                continue

            # for the token not overlapped, the initial embedding should be zero
            assert np.all(target_matrix[token_id] == 0)

            # get the closest neighbors of the subword token
            closest = get_n_closest(token_id, similarities[token_id - start], neighbors)

            # this corresponds to the case when the subword embedding is not zero
            if closest is not None:
                tokens, sims = zip(*closest)
                weights = softmax(np.array(sims) / temperature, 0)

                found[token_id] = target_tokenizer.convert_ids_to_tokens(token_id)

                emb = np.zeros(target_matrix.shape[1])

                for sim_i, close_token in enumerate(tokens):
                    emb += source_matrix[source_vocab[close_token]] * weights[sim_i]

                target_matrix[token_id] = emb

                n_matched += 1
                how_many_updated += 1
            else:
                # this is a random initialization
                target_matrix[token_id] = random_fallback_matrix[token_id]
                not_found[token_id] = target_tokenizer.convert_ids_to_tokens(token_id)
                how_many_randomly_updated += 1

    # this is to copy the special tokens
    # we only need to do this if we don't include overlapped tokens
    if additional_tokens is None and overlapping_tokens is None:
        for token in source_tokenizer.special_tokens_map.values():
            if isinstance(token, str):
                token = [token]

            for t in token:
                if t in target_tokenizer.vocab and t in additional_tokens:
                    target_matrix[target_tokenizer.vocab[t]] = source_matrix[source_tokenizer.vocab[t]]

    logging.info(
        f"Matching token found for {n_matched} of {len(target_matrix)} tokens."
    )

    print(f"kept: {how_many_kept}")
    print(f"Updated well: {how_many_updated}")
    print(f"Updated randomly: {how_many_randomly_updated}")
    return target_matrix, not_found, found


def perform_factorize(source_matrix, keep_dim=100):
    """
    :param source_matrix: E^s, the PLM embeddings in the source languages.
    :param keep_dim: the dimension after reduction, or the number of semantic primitives
    """
    try:
        # factorize the multilingual embedding using svd
        u, s, vh = np.linalg.svd(source_matrix, full_matrices=False)

        primitive_embeddings = np.matmul(vh.T[:, :keep_dim], np.diag(s[:keep_dim])).T
        # primitive_embeddings size: (keep_dim, vector_size)

        lower_coordinates = u[:, :keep_dim]
        # size: (num_words, keep_dim)
    except:
        raise ValueError("Cannot perform the factorization!")
    else:
        return primitive_embeddings, lower_coordinates
