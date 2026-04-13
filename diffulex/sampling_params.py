from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 512
    max_nfe: int | None = 512
    max_repetition_run: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_nfe is not None and self.max_nfe <= 0:
            raise ValueError(f"max_nfe must be a positive integer when set, got: {self.max_nfe}")
        if self.max_repetition_run is not None and self.max_repetition_run <= 0:
            raise ValueError(
                "max_repetition_run must be a positive integer when set, "
                f"got: {self.max_repetition_run}"
            )
