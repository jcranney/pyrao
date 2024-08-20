#!/usr/bin/env python
import torch
from pydantic import BaseModel, ConfigDict
from pyrao import ultimatestart_system_matrices
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
    _pm: torch.Tensor = None  # pupil masking vector for meas samples
    _phi: torch.Tensor = None
    pupil: torch.Tensor = None  # pupil masking vector for phase samples
    device: str = "cpu"
    noise: bool = False
    noise_sigma: float = 10.0

    def __init__(self, matrix_builder: callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        matrices = matrix_builder()

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
        self.pupil = (torch.tensor(
            matrices.p_phi,
            device=self.device,
        ).reshape(self._phi_shape) > 0.5)
        self._pm = torch.tensor(matrices.p_meas, device=self.device)

    def reset(self):
        self._phi[:] = 0.0
        self._phi += self._randmult(self.cpp_factor)

    def step(self):
        self._phi[:] = torch.einsum(
            "ij,j->i",
            self.dkp,
            self._phi,
        ) + self._randmult(self.cvv_factor)
        self._phi[:] -= self._phi[:].mean()

    def _randmult(self, mat: torch.Tensor):
        return torch.einsum(
            "ij,j->i",
            mat,
            self._randvec(mat.shape[1])
        )

    def _randvec(self, length):
        return torch.randn([length], device=self.device)

    @property
    def phi_atm(self):
        return self._phi.reshape(self._phi_shape)

    @property
    def phi_cor(self):
        pass

    @property
    def phi_res(self):
        return (self.phi_atm + self.phi_cor)*self.pupil

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

    def __init__(self, matrix_builder, *args, **kwargs):
        super().__init__(matrix_builder, *args, **kwargs)
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
        if self.noise:
            self._meas[:] *= self._pm
            self._meas[:] += self.noise_sigma*(
                self._randvec(self._meas.shape[0])
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

    def __init__(self, matrix_builder: callable, *args, **kwargs):
        super().__init__(matrix_builder, *args, **kwargs)
        from pyMilk.interfacing.shm import SHM
        import numpy as np
        n = "pyrao_com"
        try:
            self._com = SHM(n)
            if self._com.shape[0] != self.dmc.shape[1]:
                self._com = SHM(n, ((self.dmc.shape[1],), np.float32))
        except FileNotFoundError:
            self._com = SHM(n, ((self.dmc.shape[1],), np.float32))

        n = "pyrao_meas"
        try:
            self._meas = SHM(n)
            if self._meas.shape[0] != self.dmc.shape[0]:
                self._meas = SHM(n, ((self.dmc.shape[0],), np.float32))
        except FileNotFoundError:
            self._meas = SHM(n, ((self.dmc.shape[0],), np.float32))

        n = "pyrao_phi"
        try:
            self._phi_display = SHM(n)
            dims = zip(self._phi_display.shape, self._phi_shape)
            if not all([a == b for a, b in dims]):
                self._phi_display = SHM(n, (self._phi_shape, np.float32))
        except FileNotFoundError:
            self._phi_display = SHM(n, (self._phi_shape, np.float32))

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
        meas = (
            torch.einsum(
                "ij,j->i",
                self.dmp,
                self._phi,
            ) + torch.einsum(
                "ij,j->i",
                self.dmc,
                com,
            )
        )
        if self.noise:
            meas[:] *= self._pm
            meas[:] += self.noise_sigma*(
                self._randvec(self._meas.shape[0])
            )
        self._meas.set_data(meas.cpu().numpy())

    def update_displays(self):
        phi_res = self.phi_res
        phi_res[self.pupil] -= phi_res[self.pupil].mean()
        phi_res[~self.pupil] = 0.0
        self._phi_display.set_data(phi_res.cpu().numpy())

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


class SubaruLTAO(AOSystemSHM):
    def __init__(self, *args, **kwargs):
        super().__init__(ultimatestart_system_matrices, *args, **kwargs)
