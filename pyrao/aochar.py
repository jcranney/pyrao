#!/usr/bin/env python
import torch
from pydantic import BaseModel, ConfigDict
from pyrao import build_dpc, build_cpm, build_cpp, build_cmm


class AOChar(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    device: str = "cpu"
    order: int = 3
    res: torch.Tensor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dpc = torch.tensor(build_dpc(), device=self.device)
        cpm = [
            torch.tensor(build_cpm(t), device=self.device)
            for t in torch.arange(self.order)*1.0
        ]
        cpp = torch.tensor(build_cpp(), device=self.device)
        dc = dpc @ torch.linalg.solve(
            dpc.T @ dpc + 1e-5*torch.eye(dpc.shape[1]),
            dpc.T
        )
        cmm = torch.tensor(build_cmm(), device=self.device)
        cmm += 1e-4*torch.eye(cmm.shape[0], device=self.device)
        cmmi = torch.linalg.inv(cmm)
        res = dc @ torch.stack([
            cpm[i] @ cmmi @ cpm[i].T
            for i in range(self.order)
        ]).sum(dim=0)
        res = res @ cpp @ res.T
        self.res = res


if __name__ == "__main__":
    model_config = ConfigDict(arbitrary_types_allowed=True)
    device: str = "cpu"
    order: int = 5

    dpc = torch.tensor(build_dpc(), device=device)
    cpm = [
        torch.tensor(build_cpm(t), device=device)
        for t in torch.arange(order)*1.0
    ]
    cpp = torch.tensor(build_cpp(), device=device)
    dc = dpc @ torch.linalg.solve(
        dpc.T @ dpc + 1e-3*torch.eye(dpc.shape[1]),
        dpc.T
    )
    cmm = torch.tensor(build_cmm(), device=device)
    cmm += 1.0*torch.eye(cmm.shape[0], device=device)
    cmmi = torch.linalg.inv(cmm)
    cppi = torch.linalg.inv(cpp)
    tmp = dc @ torch.stack([
        cpm[i] @ cmmi @ cpm[i].T
        for i in range(1, order)
    ]).sum(dim=0) @ cppi
    tmp = torch.eye(tmp.shape[0], device=device) - tmp
    res = tmp @ cpp @ tmp.T
    w, v = torch.linalg.eigh(res)
    v = v[w > 0, :]
    w = w[w > 0]
    res_factor = v @ torch.diag(w**0.5)
    instance = res_factor @ torch.randn(res_factor.shape[1], device=device)
    import matplotlib.pyplot as plt
    plt.ion()
    plt.matshow(instance.reshape([32, 32]))
