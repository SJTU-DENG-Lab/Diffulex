import torch
import torch.nn.functional as F

from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.output import TopKOutput
from diffulex_kernel import fused_topk


class NaiveTopKRouter(TopKRouter):
    def _forward_naive(self, router_logits: torch.Tensor) -> TopKOutput:
        if self.scoring_func == "softmax":
            routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
        elif self.scoring_func == "sigmoid":
            routing_scores = torch.sigmoid(router_logits.float())
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func!r}.")

        top_k = min(self.top_k, routing_scores.shape[-1])
        topk_weights, topk_ids = torch.topk(routing_scores, top_k, dim=-1, sorted=False)
        if self.renormalize and top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        return TopKOutput(
            weights=topk_weights,
            ids=topk_ids.to(torch.int32),
            router_logits=router_logits,
        )

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        if not router_logits.is_cuda:
            return self._forward_naive(router_logits)

        topk_weights, topk_ids = fused_topk(
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
        )
        return TopKOutput(weights=topk_weights, ids=topk_ids, router_logits=router_logits)
