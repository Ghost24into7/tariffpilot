"""Versioned prompt registry. Prompts live in prompts/*.yaml, never inline in code."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    tier: str
    system: str
    template: str
    hash: str

    def render(self, variables: dict) -> tuple[str, str]:
        safe = {k: str(v).replace("</product_description>", "") for k, v in variables.items()}
        return self.system.strip(), Template(self.template).substitute(safe).strip()


class PromptRegistry:
    def __init__(self, directory: Path):
        self._p: dict[str, Prompt] = {}
        for f in sorted(Path(directory).glob("*.yaml")):
            d = yaml.safe_load(f.read_text())
            for key in ("id", "version", "tier", "system", "template"):
                if key not in d:
                    raise ValueError(f"{f.name}: missing '{key}'")
            if d["tier"] not in ("lite", "flash"):
                raise ValueError(f"{f.name}: tier must be lite|flash")
            h = hashlib.sha256((d["system"] + d["template"]).encode()).hexdigest()[:12]
            self._p[d["id"]] = Prompt(d["id"], int(d["version"]), d["tier"], d["system"], d["template"], h)

    def get(self, pid: str) -> Prompt:
        if pid not in self._p:
            raise KeyError(f"unknown prompt '{pid}'")
        return self._p[pid]

    def versions(self) -> dict[str, str]:
        return {p.id: f"v{p.version}:{p.hash}" for p in self._p.values()}
