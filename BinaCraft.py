#!/usr/bin/env python3
"""
BinaCraft - a tiny binary sandbox and logic-circuit prototype.
"""

from __future__ import annotations

import ast
import json
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Boolean expression engine
# ---------------------------------------------------------------------------

_ALLOWED_BOOL_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
)

class BoolExprError(ValueError):
    pass

class BoolExpr:
    def __init__(self, expression: str):
        self.expression = expression.strip()
        if not self.expression:
            raise BoolExprError("empty boolean expression")
        self._tree = self._parse_and_validate(self.expression)

    @staticmethod
    def _parse_and_validate(expr: str) -> ast.AST:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise BoolExprError(f"syntax error in expression: {expr!r}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_BOOL_NODES):
                raise BoolExprError(f"disallowed expression element: {type(node).__name__}")
            if isinstance(node, ast.Name):
                if node.id not in {"N", "E", "S", "W", "ANY", "ALL", "A", "B", "C", "D"}:
                    raise BoolExprError(f"unknown symbol: {node.id}")
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (bool, int)):
                    raise BoolExprError("only boolean/int constants are allowed")
        return tree

    def eval(self, values: Dict[str, int]) -> int:
        def walk(node: ast.AST) -> bool:
            if isinstance(node, ast.Expression): return walk(node.body)
            if isinstance(node, ast.Constant): return bool(int(node.value))
            if isinstance(node, ast.Name): return bool(int(values.get(node.id, 0)))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not): return not walk(node.operand)
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return all(walk(v) for v in node.values)
                if isinstance(node.op, ast.Or):
                    return any(walk(v) for v in node.values)
            raise BoolExprError("unsupported expression")
        return int(walk(self._tree))

    def to_json(self) -> str:
        return self.expression

# ---------------------------------------------------------------------------
# Game model
# ---------------------------------------------------------------------------

KIND_EMPTY, KIND_POWER, KIND_WIRE, KIND_SWITCH = "empty", "power", "wire", "switch"
KIND_AND, KIND_OR, KIND_NOT, KIND_MACRO = "and", "or", "not", "macro"

KIND_CHARS = {
    KIND_EMPTY: ".", KIND_POWER: "P", KIND_WIRE: "#", KIND_SWITCH: "S",
    KIND_AND: "A", KIND_OR: "O", KIND_NOT: "N", KIND_MACRO: "M",
}

@dataclass
class Cell:
    kind: str = KIND_EMPTY
    state: int = 0
    macro: Optional[str] = None
    label: str = ""

    def output(self, section: "Section", x: int, y: int, previous: List[List[int]], project: "Project") -> int:
        if self.kind == KIND_EMPTY: return 0
        if self.kind == KIND_POWER: return 1
        if self.kind == KIND_SWITCH: return int(bool(self.state))
        if self.kind == KIND_WIRE: return int(any(section.neighbor_outputs(previous, x, y)))
        if self.kind == KIND_AND:
            return int(bool(section.in_dir(previous, x, y, "N")) and bool(section.in_dir(previous, x, y, "E")))
        if self.kind == KIND_OR:
            return int(any(bool(section.in_dir(previous, x, y, d)) for d in "NESW"))
        if self.kind == KIND_NOT:
            return int(not bool(section.in_dir(previous, x, y, "N")))
        if self.kind == KIND_MACRO:
            if not self.macro or not (m := project.macros.get(self.macro)): return 0
            return m.evaluate(section, x, y, previous, project)
        return 0

    def symbol(self, state: int) -> str:
        chars = {KIND_EMPTY: ".", KIND_POWER: "P", KIND_SWITCH: "T" if state else "t",
                 KIND_WIRE: "#" if state else ":", KIND_AND: "A" if state else "a",
                 KIND_OR: "O" if state else "o", KIND_NOT: "N" if state else "n",
                 KIND_MACRO: "M" if state else "m"}
        return chars.get(self.kind, "?")

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "state": self.state, "macro": self.macro, "label": self.label}

@dataclass
class MacroDef:
    name: str
    expression: BoolExpr
    description: str = ""

    def evaluate(self, section: "Section", x: int, y: int, previous: List[List[int]], project: "Project") -> int:
        return self.expression.eval(section.inputs_dict(previous, x, y))

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "expression": self.expression.to_json(), "description": self.description}

@dataclass
class Section:
    name: str
    width: int
    height: int
    grid: List[List[Cell]] = field(init=False)
    last_state: Optional[List[List[int]]] = None

    def __post_init__(self) -> None:
        self.grid = [[Cell() for _ in range(self.width)] for _ in range(self.height)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def place(self, x: int, y: int, kind: str, *, macro: Optional[str] = None, state: int = 0, label: str = "") -> None:
        if not self.in_bounds(x, y): return
        self.grid[y][x] = Cell(kind=kind, state=int(state), macro=macro, label=label)

    def in_dir(self, previous: List[List[int]], x: int, y: int, direction: str) -> int:
        dx, dy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[direction]
        nx, ny = x + dx, y + dy
        return previous[ny][nx] if self.in_bounds(nx, ny) else 0

    def neighbor_outputs(self, previous: List[List[int]], x: int, y: int) -> List[int]:
        return [self.in_dir(previous, x, y, d) for d in "NESW"]

    def inputs_dict(self, previous: List[List[int]], x: int, y: int) -> Dict[str, int]:
        d = {k: self.in_dir(previous, x, y, k) for k in "NESW"}
        d.update({"ANY": int(any(d.values())), "ALL": int(all(d.values())),
                  "A": d["N"], "B": d["E"], "C": d["S"], "D": d["W"]})
        return d

    def snapshot(self) -> List[List[int]]:
        if self.last_state is not None: return [row[:] for row in self.last_state]
        return [[(1 if c.kind == KIND_POWER or (c.kind == KIND_SWITCH and c.state) else 0) for c in row] for row in self.grid]

    def tick(self, project: "Project") -> List[List[int]]:
        prev = self.snapshot()
        return [[self.grid[y][x].output(self, x, y, prev, project) for x in range(self.width)] for y in range(self.height)]

    def step(self, project: "Project", count: int = 1) -> None:
        for _ in range(max(0, count)): self.last_state = self.tick(project)

    def render_mixed(self, project: "Project") -> str:
        s, lines = self.snapshot(), [f"Section {self.name!r} [{self.width}x{self.height}]"]
        for y, row in enumerate(self.grid):
            lines.append(" ".join(c.symbol(c.output(self, x, y, s, project)) for x, c in enumerate(row)))
        return "\n".join(lines)

    def render_signals(self, project: "Project") -> str:
        s, lines = self.snapshot(), [f"Section {self.name!r} signals:"]
        for y, row in enumerate(self.grid):
            lines.append(" ".join(str(c.output(self, x, y, s, project)) for x, c in enumerate(row)))
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "width": self.width, "height": self.height,
                "grid": [[cell.to_json() for cell in row] for row in self.grid]}

@dataclass
class Project:
    name: str
    sections: Dict[str, Section] = field(default_factory=dict)
    macros: Dict[str, MacroDef] = field(default_factory=dict)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "sections": {n: s.to_json() for n, s in self.sections.items()},
                       "macros": {n: m.to_json() for n, m in self.macros.items()}}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
            p = cls(d["name"])
            for n, sd in d.get("sections", {}).items():
                sec = Section(n, sd["width"], sd["height"])
                for y, row in enumerate(sd["grid"]):
                    for x, cd in enumerate(row): sec.grid[y][x] = Cell(**cd)
                p.sections[n] = sec
            for n, md in d.get("macros", {}).items():
                p.macros[n] = MacroDef(n, BoolExpr(md["expression"]), md.get("description", ""))
            return p

# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class AbstrangInterpreter:
    def __init__(self): self.project: Optional[Project] = None
    def _clear(self): os.system('cls' if os.name == 'nt' else 'clear')

    def run_line(self, line: str):
        t = line.replace("=", " = ").split()
        if not t: return
        cmd = t[0].lower()
        if cmd == "clear": self._clear(); return
        if cmd == "help": print("project, section, define, place, tick, show, signals, clear, save, load, quit"); return
        if cmd == "project": self.project = Project(t[1]); print(f"Project {t[1]} created"); return
        if not self.project: print("Error: No project"); return
        try:
            if cmd == "section": self.project.sections[t[1]] = Section(t[1], int(t[2]), int(t[3])); print(f"Section {t[1]} added")
            elif cmd == "define": self.project.macros[t[1]] = MacroDef(t[1], BoolExpr(" ".join(t[t.index("=")+1:])))
            elif cmd == "place":
                sec = self.project.sections[t[1]]
                kind, mc, st = t[4].lower(), (t[5] if t[4].lower() == "macro" else None), (1 if "on" in t[5:] else 0)
                sec.place(int(t[2]), int(t[3]), kind, macro=mc, state=st)
            elif cmd == "tick": self.project.sections[t[1]].step(self.project, int(t[2]) if len(t)>2 else 1)
            elif cmd == "show": print(self.project.sections[t[1]].render_mixed(self.project))
            elif cmd == "signals": print(self.project.sections[t[1]].render_signals(self.project))
            elif cmd == "save": self.project.save(t[1])
            elif cmd == "load": self.project = Project.load(t[1])
        except Exception as e: print(f"Error: {e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEMO = """
project BinaCraft
section main 12 6
define XOR = (N and not E) or (not N and E)
place main 1 1 power
place main 2 1 wire
place main 3 1 switch on
place main 4 1 and
place main 5 1 or
place main 6 1 not
place main 7 1 macro XOR
tick main 1
show main
"""

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repl", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    interp = AbstrangInterpreter()
    if args.demo:
        for l in DEMO.strip().splitlines(): interp.run_line(l)
    else:
        print("BinaCraft REPL ('help' for info)"); 
        while True:
            try:
                l = input(">> ").strip()
                if l.lower() in {"quit", "exit"}: break
                interp.run_line(l)
            except (EOFError, KeyboardInterrupt): break

if __name__ == "__main__": main()
