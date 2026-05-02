from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerShiftBase, SemiCompleteAcceptedIdsMixin


@AutoSampler.register("dream")
class DreamSampler(SemiCompleteAcceptedIdsMixin, DllmSamplerShiftBase):
    pass
