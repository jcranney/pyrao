#!/usr/bin/env python
import torch
from pydantic import BaseModel, ConfigDict
from pyrao import build_system_matrices
import tqdm
import itertools


class AOSystemGeneric(BaseModel):
    # The generic variant of this class only initialises the minimal matrices
    # and doesn't provide any user-level API except for step and reset.
    # Also, this classes init method does not 
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cvv_factor: torch.Tensor = None
    cpp_factor: torch.Tensor = None
    dkp: torch.Tensor = None
    dmp: torch.Tensor = None
    dmc: torch.Tensor = None
    dpc: torch.Tensor = None
    _phi: torch.Tensor = None
    device: str = "cpu"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        matrices = build_system_matrices(verbose=False)

        cmp = torch.tensor(matrices.c_meas_phi, device=self.device)
        ckp = torch.tensor(matrices.c_phip1_phi, device=self.device)
        cpp = torch.tensor(matrices.c_phi_phi, device=self.device)
        self.dmc = torch.tensor(matrices.d_meas_com, device=self.device)
        self.dpc = torch.tensor(matrices.d_phi_com, device=self.device)
        # these are derived products needed to run the simulation
        self.dkp = torch.linalg.solve_ex(cpp, ckp, left=False)[0]
        _cvv = cpp-torch.einsum("ij,jk,lk->il", self.dkp, cpp, self.dkp)
        self.cvv_factor = torch.linalg.cholesky_ex(_cvv)[0]
        self.cpp_factor = torch.linalg.cholesky_ex(cpp)[0]
        self.dmp = torch.linalg.solve_ex(cpp, cmp, left=False)[0]
        self._phi_shape = (int(cpp.shape[0]**0.5),)*2
        self._phi = self._randmult(self.cpp_factor)

    def reset(self):
        self._phi[:] = 0.0
        self._phi += self._randmult(self.cpp_factor)

    def step(self):
        self._phi[:] = torch.einsum(
            "ij,j->i",
            self.dkp,
            self._phi,
        ) + self._randmult(self.cvv_factor)

    def _randmult(self, mat: torch.Tensor):
        return torch.einsum(
            "ij,j->i",
            mat,
            torch.randn([mat.shape[1]], device=self.device)
        )

    @property
    def phi_atm(self):
        return self._phi.reshape(self._phi_shape)

    @property
    def phi_cor(self):
        pass

    @property
    def phi_res(self):
        return self.phi_atm + self.phi_cor

    @property
    def perf(self):
        rms_wfe_rad = self.phi_res.std()
        return {
            "strehl": torch.exp(-rms_wfe_rad**2),
            "wfe":  rms_wfe_rad,
        }


class AOSystem(AOSystemGeneric):
    # The suffix-free variant of the AOSystem is the one intended to be
    # interacted with at the user-level in python.
    _com: torch.Tensor = None
    _meas: torch.Tensor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._com = torch.zeros(self.dmc.shape[1], device=self.device)
        self._meas = torch.zeros(self.dmc.shape[0], device=self.device)

    def reset(self):
        super().reset()
        self._com[:] = 0.0
        return self.step()  # updates internal measurement and returns it

    def step(self):
        super().step()
        self._meas[:] = torch.einsum(
            "ij,j->i",
            self.dmp,
            self._phi,
        ) + torch.einsum(
            "ij,j->i",
            self.dmc,
            self._com,
        )
        return self._meas

    def set_command(self, com):
        self._com[:] = com[:]

    @property
    def phi_cor(self):
        return (self.dpc @ self._com).reshape(self._phi_shape)


class AOSystemSHM(AOSystemGeneric):
    # The SHM variant is meant to run on SHM, waiting for commands and then
    # updating measurements. The main() function is blocking, so there is no
    # useful user interaction with this class normally.
    _com = None
    _meas = None
    _phi_display = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from pyMilk.interfacing.shm import SHM
        import numpy as np
        try:
            self._com = SHM("pyrao_com")
        except FileNotFoundError:
            self._com = SHM("pyrao_com", ((self.dmc.shape[1],), np.float32))
        try:
            self._meas = SHM("pyrao_meas")
        except FileNotFoundError:
            self._meas = SHM("pyrao_meas", ((self.dmc.shape[0],), np.float32))
        try:
            self._phi_display = SHM("pyrao_phi")
        except FileNotFoundError:
            self._phi_display = SHM("pyrao_phi", (self._phi_shape, np.float32))

    def reset(self):
        super().reset()
        self._com.set_data(self._com.get_data()*0.0)
        self.step()

    def step(self, blocking=False):
        super().step()
        com = torch.tensor(
            self._com.get_data(check=blocking),
            device=self.device
        )
        self._meas.set_data(
            (torch.einsum(
                "ij,j->i",
                self.dmp,
                self._phi,
            ) + torch.einsum(
                "ij,j->i",
                self.dmc,
                com,
            )).cpu().numpy()
        )

    def update_displays(self):
        self._phi_display.set_data(self.phi_res.cpu().numpy())

    def main(self, niter=None, blocking=False):
        if niter:
            pbar = tqdm.tqdm(range(niter))
        else:
            pbar = tqdm.tqdm(itertools.count())
        for _ in pbar:
            self.step(blocking=blocking)
            self.update_displays()

    @property
    def phi_cor(self):
        com = torch.tensor(self._com.get_data(), device=self.device)
        return (self.dpc @ com).reshape(self._phi_shape)