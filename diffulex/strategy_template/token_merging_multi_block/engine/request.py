from __future__ import annotations

from dataclasses import dataclass

from diffulex.config import Config
from diffulex.strategy_template.multi_block.engine.request import MultiBlockReqTemplate


@dataclass
class TokenMergeDescriptor:
    topk_ids: list[int]
    topk_probs: list[float]
    residual_prob: float

    def __post_init__(self) -> None:
        if not self.topk_ids:
            raise ValueError("topk_ids must not be empty")
        if len(self.topk_ids) != len(self.topk_probs):
            raise ValueError("topk_ids and topk_probs must have the same length")


class TokenMergingMultiBlockReqTemplate(MultiBlockReqTemplate):
    def init_token_merging_multi_block(self: TokenMergingMultiBlockReqTemplate, config: Config) -> None:
        self.init_multi_block(config)
        self.token_merge_enabled = True
        self.token_merge_mode = config.token_merge_mode
        self.token_merge_top_k = int(config.token_merge_top_k)
        self.token_merge_weight = float(config.token_merge_weight)
        self.token_merge_descriptors: dict[int, TokenMergeDescriptor] = {}

    def set_token_merge_descriptor(
        self,
        position: int,
        topk_ids: list[int],
        topk_probs: list[float],
        residual_prob: float,
    ) -> None:
        self.token_merge_descriptors[int(position)] = TokenMergeDescriptor(
            topk_ids=[int(token_id) for token_id in topk_ids],
            topk_probs=[float(prob) for prob in topk_probs],
            residual_prob=float(residual_prob),
        )

    def clear_token_merge_descriptor(self, position: int) -> None:
        self.token_merge_descriptors.pop(int(position), None)

    def clear_token_merge_descriptors(self, positions: list[int] | range) -> None:
        for position in positions:
            self.clear_token_merge_descriptor(int(position))

    def prune_token_merge_descriptors_to_running_sequence(self) -> None:
        if not self.token_merge_descriptors:
            return
        running_positions = {int(position) for position in self.running_position_ids}
        stale_positions = [position for position in self.token_merge_descriptors if position not in running_positions]
        for position in stale_positions:
            del self.token_merge_descriptors[position]

    def token_merge_descriptor_for_position(self, position: int) -> TokenMergeDescriptor | None:
        return self.token_merge_descriptors.get(int(position))
