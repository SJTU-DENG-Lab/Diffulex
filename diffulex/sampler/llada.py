from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase, SemiCompleteAcceptedIdsMixin


@AutoSampler.register("llada")
class LLaDASampler(SemiCompleteAcceptedIdsMixin, DllmSamplerNoShiftBase):
    pass
