import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentences = positive + negative
        unique_words = set()
        for sentence in all_sentences:
            words = sentence.split()
            unique_words.update(words)
        sorted_vocab = sorted(unique_words)

        vocab_mapping = {}
 
        for idx, word in enumerate(sorted_vocab, start=1):
            vocab_mapping[word] = idx

        encoded_tensors = []

        for sentence in all_sentences:
            words = sentence.split()
            ids = [vocab_mapping[word] for word in words]
            encoded_tensors.append(torch.tensor(ids))
        padded_tensor = nn.utils.rnn.pad_sequence(encoded_tensors, batch_first=True)
        return padded_tensor
        
