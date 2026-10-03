"""SKILL.md loader with progressive disclosure: catalog first, body on demand."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: tuple[str, ...]
    body: str


class SkillLoader:
    def __init__(self, directory: Path):
        self._s: dict[str, Skill] = {}
        for f in sorted(Path(directory).glob("*/SKILL.md")):
            text = f.read_text()
            _, front, body = text.split("---", 2)
            meta = yaml.safe_load(front)
            self._s[meta["name"]] = Skill(meta["name"], meta["description"],
                                          tuple(meta.get("triggers", [])), body.strip())

    def names(self) -> list[str]:
        return sorted(self._s)

    def catalog(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self._s.values())

    def get(self, name: str) -> Skill:
        if name not in self._s:
            raise KeyError(f"unknown skill '{name}'")
        return self._s[name]
