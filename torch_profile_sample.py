"""
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_modules=True, record_shapes=True, profile_memory=True) as prof:
    print(prof.key_averages().table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit=-1))
    #打印一个name非全名的op+kernel_list
    #这里放前向推理
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total",max_src_column_width=10000,max_name_column_width=10000,row_limit=-1))
    print(prof.key_averages().table(sort_by="self_cuda_time_total",max_src_column_width=10000,row_limit=-1))
    prof.export_chrome_trace(log_path)
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4096, 4096)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = Model().cuda().eval()
x = torch.randn(128, 4096, device="cuda")
log_path = "mlp_trace.json"

with torch.no_grad():
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_modules=True,
        record_shapes=True,
        profile_memory=True
    ) as prof:
        y = model(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
prof.export_chrome_trace(log_path)