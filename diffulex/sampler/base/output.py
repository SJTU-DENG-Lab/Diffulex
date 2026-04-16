from __future__ import annotations

from dataclasses import dataclass

from easydict import EasyDict as edict


@dataclass
class SampleOutputBase:
    true_local_ids_map: dict[str, dict[str, list[int]]]
    accepted_ids_map: dict[str, dict[str, list[int]]]
    sampled_tokens_map: dict[str, dict[str, list[int]]]
    mask_token_rel_ids_map: dict[str, dict[str, list[int]]] | None = None
    confidence_map: dict[str, dict[str, list[float]]] | None = None
    initial_confidence_map: dict[str, dict[str, list[float]]] | None = None
    edit_writes_map: dict[str, dict[str, dict[int, int]]] | None = None

    def __post_init__(self):
        req_ids = set(self.accepted_ids_map.keys())
        self.accepted_ids_map = edict(self.accepted_ids_map)
        self.sampled_tokens_map = edict(self.sampled_tokens_map)
        self.true_local_ids_map = edict(self.true_local_ids_map)
        self.mask_token_rel_ids_map = edict(self.mask_token_rel_ids_map or {})
        self.confidence_map = edict(self.confidence_map or {})
        self.initial_confidence_map = edict(self.initial_confidence_map or {})
        edit_writes_map = self.edit_writes_map or {}
        for req_id_str in req_ids:
            edit_writes_map.setdefault(req_id_str, {})
        self.edit_writes_map = edict(edit_writes_map)
