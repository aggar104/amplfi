from typing import Callable, Optional

import torch


class AmplfiPrior:
    def __init__(
        self,
        priors: dict[str, torch.distributions.Distribution],
        conversion_function: Optional[Callable] = None,
        chirp_distance_conv: bool = False,
    ):
        """
        A class for sampling parameters from a prior distribution

        Args:
            priors:
                A dictionary of parameter samplers that take an integer N
                and return a tensor of shape (N, ...) representing
                samples from the prior distribution
            conversion_function:
                A callable that takes a dictionary of sampled parameters
                and returns a dictionary of waveform generation parameters
        """
        super().__init__()
        self.priors = priors
        self.conversion_function = conversion_function or (lambda x: x)
        self.chirp_distance_conv = chirp_distance_conv

    def __call__(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Generates random samples from the prior

        Args:
            N: Number of samples to generate
            device: Device to place the samples
        """
        # sample parameters from prior
        parameters = {
            k: v.sample((N,)).to(device) for k, v in self.priors.items()
        }
        # perform any necessary conversions
        # to from sampled parameters to
        # waveform generation parameters
        parameters = self.conversion_function(parameters)
        return parameters

    def log_prob(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the log probability of samples under the prior

        Args:
            samples:
                Dictionary where key is parameter and
                value is tensor of samples
        """

        first = samples[list(samples.keys())[0]]
        log_probs = torch.ones(len(first), device=first.device)

        if self.chirp_distance_conv:
            dc = samples["distance"]
            mc = samples["chirp_mass"]
            mc_pow = mc.pow(5.0 / 6.0)
            dL = dc / mc_pow
            # base prior for luminosity distance
            log_probs = log_probs + self.priors["distance"].log_prob(dL).to(
                first.device
            )
            # Jacobian = -(5/6) log Mc
            log_probs = log_probs - (5.0 / 6.0) * torch.log(mc)

        for param, tensor in samples.items():
            if param == "distance":
                continue
            log_probs += self.priors[param].log_prob(tensor).to(first.device)
        return log_probs


class ParameterTransformer(torch.nn.Module):
    """
    Helper class for applying preprocessing
    transformations to inference parameters

    Args:
        transforms:
            Dictionary where key is the parameter and
            value is a conversion function e.g.
            amplfi.train.data.utils.transforms.sample_rescaled_distance,
            amplfi.train.data.utils.transforms.sample_chirp_distance
    """

    def __init__(
        self,
        transforms: dict[str, Callable],
    ):
        super().__init__()
        self.transforms = transforms

    def forward(
        self,
        parameters: dict[str, torch.Tensor],
    ):
        transformed = {}
        for k, v in self.transforms.items():
            if k == "distance":
                transformed[k] = v(parameters)
            else:
                transformed[k] = v(parameters[k])
        # update parameter dict
        parameters.update(transformed)
        return parameters
