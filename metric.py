from enum import StrEnum


class Metric(StrEnum):

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return tuple(cls)

    @classmethod
    def exists(cls, code: "Metric"):
        return code in cls

    @classmethod
    def cast(cls, code: str) -> "Metric":
        if code not in cls:
            raise ValueError(f"Unknown metric: {code}")
        return cls(code)
