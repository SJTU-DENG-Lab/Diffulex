from diffulex.config import Config
from diffulex.engine.request import AutoReq
from diffulex.engine.status import DllmReqStatus
from diffulex.sampling_params import SamplingParams
from diffulex.attention.metadata import is_warming_up
from diffulex.strategy.templates.token_merge.engine.request import (
    TokenMergeReqTemplate,
)


@AutoReq.register("dmax")
class DMaxReq(TokenMergeReqTemplate):
    """Req for DMax-style block diffusion with token merge."""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)
        if config is None:
            raise ValueError("DMaxReq requires config to initialize token-merge state.")
        self.dmax_force_prefill_active = bool(getattr(config, "dmax_force_prefill_active", False))
        if is_warming_up():
            # Used for warming up token merge
            self.init_token_merge(config)

    def lazy_activate(self):
        if not self.dmax_force_prefill_active:
            return super().lazy_activate()

        self.log_status()
        self.status = self.status_history[-1]
        if self.is_pending or self.is_decoding or self.is_prefilling:
            # Keep active-block iterations on the prefix-cache prefill path.
            # This matches reference generate_spd more closely than switching
            # to the 32-token decode path after the first NFE.
            self.status = DllmReqStatus.PREFILLING
