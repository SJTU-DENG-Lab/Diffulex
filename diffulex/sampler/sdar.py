from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerNoShiftBase, SemiCompleteAcceptedIdsMixin


@AutoSampler.register("sdar")
@AutoSampler.register("sdar_moe")
class SDARSampler(SemiCompleteAcceptedIdsMixin, DllmSamplerNoShiftBase):
    accepted_ids_use_block_threshold = False
