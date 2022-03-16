from typing import Dict, Optional
from rockpool.timeseries import TSContinuous, TSEvent
import numpy as np
import torch
from matplotlib import pyplot as plt
from rockpool.nn.modules.torch.torch_module import TorchModule


# Plotting
_to_numpy = lambda _tensor: _tensor.detach().numpy().flatten()


def plot_signal(tensor: str, name: str, unit: str, dt: float):
    Ix = TSContinuous.from_clocked(_to_numpy(tensor), dt=dt, name=name)
    plt.figure()
    plt.ylabel(unit)
    Ix.plot()


def plot_raster(tensor: str, name: str, dt: float):
    spikes_ts = TSEvent.from_raster(_to_numpy(tensor), dt=dt, name=name)
    plt.figure()
    spikes_ts.plot()


def plot_LIF_record(
    rec: Dict[str, torch.Tensor],
    plot_Vmem: bool = True,
    plot_Isyn: bool = True,
    plot_spikes: bool = True,
    plot_Irec: bool = False,
    threshold: Optional[float] = None,
    dt: float = 1e-3,
) -> None:

    if plot_Irec:
        plot_signal(rec["irec"].detach(), "$I_{rec}$", "Current", dt)

    if plot_Isyn:
        plot_signal(rec["isyn"].detach(), "$I_{syn}$", "Current", dt)

    if plot_Vmem:
        vmem = rec["vmem"]
        if threshold is not None:
            spike_idx = rec["spikes"].detach().nonzero(as_tuple=True)
            vmem[spike_idx] = threshold
            if len(rec["spikes"].detach()) > spike_idx[1].max().item() + 1:
                reset_idx = (spike_idx[0], spike_idx[1] + 1, spike_idx[2])
                vmem[reset_idx] = 0

            thr = np.ones_like(vmem.detach().flatten()) * threshold
            thr = TSContinuous.from_clocked(thr, dt=dt)

        plot_signal(vmem, "$V_{mem}$", "Voltage", dt)
        if threshold is not None:
            thr.plot(linestyle="dashed")
    if plot_spikes:
        plot_raster(rec["spikes"].detach(), "Output Spikes", dt)


# Data Generation


def one_spike_sample(
    t_spike: float = 10e-3,
    duration: float = 100e-3,
    dt: float = 1e-3,
) -> torch.Tensor:

    _n = lambda t: int(np.around(t / dt))
    _sample = torch.zeros((1, _n(duration), 1))
    _sample[0, _n(t_spike), 0] = 1

    return _sample


def constant_rate_spike_train(
    duration: float,
    rate: float,
    dt: float = 1e-3,
) -> torch.Tensor:
    """

    :param duration: The simulation duration in seconds
    :type duration: float
    :param rate: The spiking rate in Hertz(1/s)
    :type rate: float
    :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
    :type dt: float, optional
    :return: constant rate discrete spike train
    :rtype: torch.Tensor

    """
    _n = lambda _t, _s: int(np.around(_t / _s))
    steps = _n(duration, dt)
    _idx = _n(steps, rate)

    _sample = torch.zeros((1, steps, 1))
    _sample[0, range(_idx, steps, _idx), 0] = 1

    return _sample


def poisson_spike_train(
    duration: float, rate: float, dt: float = 1e-3, seed: Optional[int] = None
) -> torch.Tensor:
    """
    random_spike_train generate a Poisson frozen random spike train and

    :param duration: The simulation duration in seconds
    :type duration: float
    :param rate: The spiking rate in Hertz(1/s)
    :type rate: float
    :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
    :type dt: float, optional
    :raises ValueError: No spike generated due to low firing rate or very short simulation time]
    :return: randomly generated discrete spike train
    :rtype: torch.Tensor

    """
    np.random.seed(seed)
    steps = int(np.round(duration / dt))
    input_sp_raster = np.random.poisson(rate * dt, (1, steps, 1))
    if not any(input_sp_raster.flatten()):
        raise ValueError(
            "No spike generated at all due to low firing rate or short simulation time duration!"
        )
    input_sp_ts = torch.tensor(input_sp_raster, dtype=torch.float)
    return input_sp_ts


class RateReadout(TorchModule):
    def __init__(
        self,
        shape: tuple,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialise a LinearTorch layer

        Args:
            shape (tuple): The shape of this layer ``(Nin, Nout)``
        """
        # - Initialise superclass
        super().__init__(shape=shape, *args, **kwargs)

        # - Check arguments
        if len(self.shape) != 2:
            raise ValueError(
                "`shape` must specify input and output sizes for RateReadout."
            )

        self.dt = dt

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input, _ = self._auto_batch(input)

        n_batch, n_timesteps, n_channel = input.shape
        rate = torch.sum(input) / (n_timesteps * self.dt)
        return rate
