import torch


class EncodeIndexModule(torch.nn.Module):
    def __init__(self, idx_dim: int, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + 2 * idx_dim + time_dim

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, raw_msg: torch.Tensor,
                t_enc: torch.Tensor, src_enc: torch.Tensor, dst_enc: torch.Tensor):

        return torch.cat([z_src, z_dst, raw_msg, src_enc, dst_enc, t_enc], dim=-1)