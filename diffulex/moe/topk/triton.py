import torch
import torch.nn.functional as F

from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.datatype import TopKOutput
from diffulex_kernel import fused_topk


class TritonFusedTopKRouter(TopKRouter):

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:

        topk_weights, topk_ids = fused_topk(
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
        )
        
        return TopKOutput(
            weights=topk_weights,
            ids=topk_ids,
            router_logits=router_logits,
        )
