from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class DllmBlockEditMixin:
    editable_start: int = 0
    confidence_threshold: float = 0.9
    _previous_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    _current_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    _confidence_values: list[float] | None = field(default=None, init=False, repr=False)

    @property
    def editable_relative_ids(self) -> list[int]:
        return list(range(int(self.editable_start), self.block_size))

    def observe_edit_state(
        self,
        token_ids: list[int],
        confidences: list[float] | None = None,
    ) -> None:
        token_ids = [int(token_id) for token_id in token_ids]
        if len(token_ids) != int(self.block_size):
            raise ValueError(
                f"Expected {self.block_size} token ids for block {self.block_id}, "
                f"got {len(token_ids)}"
            )

        if confidences is not None and len(confidences) != int(self.block_size):
            raise ValueError(
                f"Expected {self.block_size} confidence values for block {self.block_id}, "
                f"got {len(confidences)}"
            )

        if self._current_token_ids is None:
            self._previous_token_ids = list(token_ids)
        else:
            self._previous_token_ids = self._current_token_ids
        self._current_token_ids = list(token_ids)
        self._confidence_values = None if confidences is None else [float(value) for value in confidences]

    @property
    def previous_token_ids(self) -> list[int] | None:
        return None if self._previous_token_ids is None else list(self._previous_token_ids)

    @property
    def current_token_ids(self) -> list[int] | None:
        return None if self._current_token_ids is None else list(self._current_token_ids)

    @property
    def confidence_values(self) -> list[float] | None:
        return None if self._confidence_values is None else list(self._confidence_values)

    @property
    def same_token_ratio(self) -> float:
        if self._previous_token_ids is None or self._current_token_ids is None:
            return 0.0

        editable_ids = self.editable_relative_ids
        if not editable_ids:
            return 1.0

        same_count = sum(
            self._previous_token_ids[rel_idx] == self._current_token_ids[rel_idx]
            for rel_idx in editable_ids
        )
        return same_count / len(editable_ids)

    @property
    def same_as_previous(self) -> bool:
        return self.same_token_ratio == 1.0

    @property
    def all_confident(self) -> bool:
        if self._confidence_values is None:
            return False
        editable_confidences = [
            self._confidence_values[rel_idx]
            for rel_idx in self.editable_relative_ids
            if self._current_token_ids is None or self._current_token_ids[rel_idx] != int(self.mask_token_id)
        ]
        if not editable_confidences:
            return True
        return all(value >= float(self.confidence_threshold) for value in editable_confidences)

    @property
    def commit_ready(self) -> bool:
        return self.same_as_previous or self.all_confident
