import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 4

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# Encoder with summary tokens
enc = tiktoken.get_encoding("gpt2")
sum_tokens = {"<|sum|>" : 50257, "<|core|>" : 50258}
custom_enc = tiktoken.Encoding(name="gpt2_custom", 
                               pat_str=enc._pat_str, 
                               mergeable_ranks=enc._mergeable_ranks, 
                               special_tokens={**enc._special_tokens, **sum_tokens})
sumtok = sum_tokens["<|sum|>"]
coretok = sum_tokens["<|core|>"]

if __name__ == '__main__':
    split_dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    split_dataset.pop("test", None)
    split_dataset['val'] = split_dataset.pop('validation', None)

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        article = [coretok] + custom_enc.encode_ordinary(example['article']) # encode_ordinary ignores any special tokens
        summary = [sumtok] + custom_enc.encode_ordinary(example['highlights'])
        article.append(custom_enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        ids = summary + article 
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['article', 'highlights', 'id'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
