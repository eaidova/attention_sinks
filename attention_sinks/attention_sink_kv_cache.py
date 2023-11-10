"""
Adapted from https://github.com/mit-han-lab/streaming-llm
"""

from dataclasses import dataclass

import torch


def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start:int, end:int):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

@torch.jit.script_if_tracing
def get_slice(key:torch.Tensor, sink_window_size, k_seq_dim, cache_size, sink_size):
    if key.shape[k_seq_dim] > cache_size:
        return  torch.cat(
                    [
                        slice2d(key, 0, sink_size),
                        slice2d(key, key.shape[k_seq_dim] - sink_window_size, key.shape[k_seq_dim]),
                    ],
                    dim=k_seq_dim,
                )
    return key

@dataclass
class AttentionSinkKVCache:
    attention_sink_size: int = 4
    attention_sink_window_size: int = 1020
    k_seq_dim: int = 2
    v_seq_dim: int = 2

    def __post_init__(self):
        self.cache_size = self.attention_sink_size + self.attention_sink_window_size
        self.k_slice = DIM_TO_SLICE[self.k_seq_dim]
        self.v_slice = DIM_TO_SLICE[self.v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        #if seq_len <= self.cache_size:
        #    return past_key_values
        return tuple([
            tuple([get_slice(k, torch.tensor(self.attention_sink_window_size), torch.tensor(self.k_seq_dim), torch.tensor(self.cache_size), torch.tensor(self.attention_sink_size)),
                   get_slice(v, torch.tensor(self.attention_sink_window_size), torch.tensor(self.v_seq_dim), torch.tensor(self.cache_size), torch.tensor(self.attention_sink_size))]
            )
            for k, v in past_key_values
        ])

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.attention_sink_size),
                        self.k_slice(
                            k,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.attention_sink_size),
                        self.v_slice(
                            v,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
