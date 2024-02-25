import json
from typing import List

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# token_transform = get_tokenizer('spacy', language='en_core_web_sm')

def load_data(path: str):
    out = []
    with open(path, "r") as f:
        for line in f:
            instance: dict = json.loads(line)
            out.append(instance)
    return out

# helper function to yield list of tokens
def yield_tokens(json_path: str) -> List[str]:
    json_list = load_data(json_path)
    for data_dict in json_list:
        for i in range(len(data_dict["captions"])):
            yield data_dict["captions"][i].split()


def make_vocab(jsonl_path: str) -> torchtext.vocab.Vocab:
    """
    Create a vocab from a jsonl file.
    """
    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    # Create torchtext's Vocab object
    vocab_transform = build_vocab_from_iterator(yield_tokens(jsonl_path),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    vocab_transform.set_default_index(UNK_IDX)

    return vocab_transform


if __name__ == "__main__":
    vocab = make_vocab("data/VATEX_features/preprocessed_jsonl/training.jsonl")
    print(len(vocab))
    sentence = "this is a test sentence"
    print(vocab(sentence.split()))
