#!/usr/bin/env python3
"""
BinaCraft - a tiny binary sandbox and logic-circuit prototype.

Features:
- 2D sections made of binary cells
- Built-in blocks: power, wire, switch, AND, OR, NOT
- Reusable custom switches ("macros") defined with boolean expressions
- Projects containing multiple sections
- A compact Abstrang-like script language
- JSON save/load
- Terminal rendering
- OS-aware screen clearing
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
    """Safe boolean expression compiler for reusable custom switches."""

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

    @staticmethod
    def _to_bool(value: Any) -> bool:
        return bool(int(value))

    def eval(self, values: Dict[str, int]) -> int:
        def walk(node: ast.AST) -> bool:
            if isinstance(node, ast.Expression):
                return walk(node.body)
            if isinstance(node, ast.Constant):
                return self._to_bool(node.value)
            if isinstance(node, ast.Name):
                return self._to_bool(values.get(node.id, 0))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not walk(node.operand)
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    result = True
                    for value in node.values:
                        result = result and walk(value)
                        if not result: break
                    return result
                if isinstance(node.op, ast.Or):
                    result = False
                    for value in node.values:
                        result = result or walk(value)
                        if result: break
                    return result
            raise BoolExprError("unsupported expression")
        return int(walk(self._tree))

    def to_json(self) -> str:
        return self.expression

    @classmethod
    def from_json(cls, data: str) -> "BoolExpr":
        return cls(data)

# ---------------------------------------------------------------------------
# Game model
# ---------------------------------------------------------------------------

KIND_EMPTY = "empty"
KIND_POWER = "power"
KIND_WIRE = "wire"
KIND_SWITCH = "switch"
KIND_AND = "and"
KIND_OR = "or"
KIND_NOT = "not"
KIND_MACRO = "macro"

KIND_CHARS = {
    KIND_EMPTY: ".",
    KIND_POWER: "P",
    KIND_WIRE: "#",
    KIND_SWITCH: "S",
    KIND_AND: "A",
    KIND_OR: "O",
    KIND_NOT: "N",
    KIND_MACRO: "M",
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
            n = section.in_dir(previous, x, y, "N")
            e = section.in_dir(previous, x, y, "E")
            return int(bool(n) and bool(e))
        if self.kind == KIND_OR:
            n = section.in_dir(previous, x, y, "N")
            e = section.in_dir(previous, x, y, "E")
            s = section.in_dir(previous, x, y, "S")
            w = section.in_dir(previous, x, y, "W")
            return int(bool(n) or bool(e) or bool(s) or bool(w))
        if self.kind == KIND_NOT:
            n = section.in_dir(previous, x, y, "N")
            return int(not bool(n))
        if self.kind == KIND_MACRO:
            if not self.macro: return 0
            macro = project.macros.get(self.macro)
            if not macro: return 0
            return macro.evaluate(section, x, y, previous, project)
        return 0

    def symbol(self, state: int) -> str:
        if self.kind == KIND_EMPTY: return "."
        if self.kind == KIND_SWITCH: return "T" if state else "t"
        if self.kind == KIND_POWER: return "P"
        if self.kind == KIND_WIRE: return "#" if state else ":"
        if self.kind == KIND_AND: return "A" if state else "a"
        if self.kind == KIND_OR: return "O" if state else "o"
        if self.kind == KIND_NOT: return "N" if state else "n"
        if self.kind == KIND_MACRO: return "M" if state else "m"
        return "?"

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "state": self.state, "macro": self.macro, "label": self.label}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Cell":
        return cls(kind=data.get("kind", KIND_EMPTY), state=int(data.get("state", 0)),
                   macro=data.get("macro"), label=data.get("label", ""))

@dataclass
class MacroDef:
    name: str
    expression: BoolExpr
    description: str = ""

    def evaluate(self, section: "Section", x: int, y: int, previous: List[List[int]], project: "Project") -> int:
        values = section.inputs_dict(previous, x, y)
        return int(self.expression.eval(values))

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "expression": self.expression.to_json(), "description": self.description}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MacroDef":
        return cls(name=data["name"], expression=BoolExpr.from_json(data["expression"]),
                   description=data.get("description", ""))

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

    def cell(self, x: int, y: int) -> Cell:
        if not self.in_bounds(x, y): raise IndexError(f"cell out of bounds: ({x}, {y})")
        return self.grid[y][x]

    def set_cell(self, x: int, y: int, cell: Cell) -> None:
        if not self.in_bounds(x, y): raise IndexError(f"cell out of bounds: ({x}, {y})")
        self.grid[y][x] = cell

    def place(self, x: int, y: int, kind: str, *, macro: Optional[str] = None, state: int = 0, label: str = "") -> None:
        if kind not in {KIND_EMPTY, KIND_POWER, KIND_WIRE, KIND_SWITCH, KIND_AND, KIND_OR, KIND_NOT, KIND_MACRO}:
            raise ValueError(f"unknown block kind: {kind}")
        self.set_cell(x, y, Cell(kind=kind, state=int(state), macro=macro, label=label))

    def toggle(self, x: int, y: int) -> None:
        c = self.cell(x, y)
        if c.kind != KIND_SWITCH: raise ValueError("only switches can be toggled")
        c.state = 0 if c.state else 1

    def set_switch(self, x: int, y: int, state: int) -> None:
        c = self.cell(x, y)
        if c.kind != KIND_SWITCH: raise ValueError("only switches can be set")
        c.state = int(bool(state))

    def neighbor_outputs(self, previous: List[List[int]], x: int, y: int) -> List[int]:
        outs = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny): outs.append(int(previous[ny][nx]))
        return outs

    def in_dir(self, previous: List[List[int]], x: int, y: int, direction: str) -> int:
        mapping = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
        dx, dy = mapping[direction]
        nx, ny = x + dx, y + dy
        return int(previous[ny][nx]) if self.in_bounds(nx, ny) else 0

    def inputs_dict(self, previous: List[List[int]], x: int, y: int) -> Dict[str, int]:
        n, e, s, w = [self.in_dir(previous, x, y, d) for d in "NESW"]
        vals = {"N": n, "E": e, "S": s, "W": w, "A": n, "B": e, "C": s, "D": w}
        vals["ANY"] = int(bool(n or e or s or w))
        vals["ALL"] = int(bool(n and e and s and w))
        return vals

    def snapshot(self) -> List[List[int]]:
        if self.last_state is not None: return [row[:] for row in self.last_state]
        return [[(1 if c.kind == KIND_POWER or (c.kind == KIND_SWITCH and c.state) else 0) for c in row] for row in self.grid]

    def tick(self, project: "Project") -> List[List[int]]:
        previous = self.snapshot()
        next_state = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                next_state[y][x] = self.grid[y][x].output(self, x, y, previous, project)
        return next_state

    def step(self, project: "Project", count: int = 1) -> None:
        for _ in range(max(0, int(count))):
            self.last_state = self.tick(project)

    def render_kinds(self) -> str:
        lines = [f"Section {self.name!r} [{self.width}x{self.height}] kinds:"]
        for row in self.grid: lines.append(" ".join(KIND_CHARS.get(cell.kind, "?") for cell in row))
        return "\n".join(lines)

    def render_signals(self, project: "Project") -> str:
        state = self.snapshot()
        lines = [f"Section {self.name!r} [{self.width}x{self.height}] signals:"]
        for y, row in enumerate(self.grid):
            lines.append(" ".join(str(c.output(self, x, y, state, project)) for x, c in enumerate(row)))
        return "\n".join(lines)

    def render_mixed(self, project: "Project") -> str:
        state = self.snapshot()
        lines = [f"Section {self.name!r} [{self.width}x{self.height}]"]
        for y, row in enumerate(self.grid):
            lines.append(" ".join(c.symbol(c.output(self, x, y, state, project)) for x, c in enumerate(row)))
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "width": self.width, "height": self.height,
                "grid": [[cell.to_json() for cell in row] for row in self.grid]}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Section":
        sec = cls(name=data["name"], width=int(data["width"]), height=int(data["height"]))
        for y, row in enumerate(data.get("grid", [])[:sec.height]):
            for x, cell_data in enumerate(row[:sec.width]):
                sec.grid[y][x] = Cell.from_json(cell_data)
        return sec

@dataclass
class Project:
    name: str
    sections: Dict[str, Section] = field(default_factory=dict)
    macros: Dict[str, MacroDef] = field(default_factory=dict)

    def add_section(self, name: str, width: int, height: int) -> Section:
        sec = Section(name=name, width=width, height=height)
        self.sections[name] = sec
        return sec

    def get_section(self, name: str) -> Section:
        return self.sections[name]

    def define_macro(self, name: str, expression: str, description: str = "") -> MacroDef:
        macro = MacroDef(name=name, expression=BoolExpr(expression), description=description)
        self.macros[name] = macro
        return macro

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "sections": {n: s.to_json() for n, s in self.sections.items()},
                "macros": {n: m.to_json() for n, m in self.macros.items()}}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Project":
        proj = cls(name=data["name"])
        for n, sd in data.get("sections", {}).items(): proj.sections[n] = Section.from_json(sd)
        for n, md in data.get("macros", {}).items(): proj.macros[n] = MacroDef.from_json(md)
        return proj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

# ---------------------------------------------------------------------------
# Abstrang Interpreter
# ---------------------------------------------------------------------------

class ScriptError(RuntimeError): pass

class AbstrangInterpreter:
    def __init__(self):
        self.project: Optional[Project] = None

    def _clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def run_line(self, line: str) -> None:
        tokens = line.replace("=", " = ").split()
        if not tokens: return
        cmd = tokens[0].lower()

        if cmd == "clear":
            self._clear_screen()
            return
        if cmd == "help":
            print(self.help_text()); return
        if cmd == "project":
            self.project = Project(tokens[1])
            print(f"created project {tokens[1]!r}"); return
        
        if self.project is None: raise ScriptError("create or load a project first")

        if cmd == "section":
            self.project.add_section(tokens[1], int(tokens[2]), int(tokens[3]))
            print(f"added section {tokens[1]!r}"); return
        if cmd in {"define", "switchdef"}:
            eq = tokens.index("=")
            self.project.define_macro(tokens[1], " ".join(tokens[eq+1:]))
            print(f"defined macro {tokens[1]!r}"); return
        if cmd == "place":
            sec = self.project.get_section(tokens[1])
            kind = tokens[4].lower()
            st, mc, lb = 0, None, ""
            extra = [t.lower() for t in tokens[5:]]
            if kind == KIND_SWITCH and extra: st = 1 if extra[0] in {"on", "1"} else 0
            elif kind == KIND_MACRO: mc = tokens[5]
            elif extra: lb = " ".join(tokens[5:])
            sec.place(int(tokens[2]), int(tokens[3]), kind, macro=mc, state=st, label=lb)
            print(f"placed {kind} at ({tokens[2]},{tokens[3]})"); return
        if cmd == "toggle":
            self.project.get_section(tokens[1]).toggle(int(tokens[2]), int(tokens[3]))
            print("toggled"); return
        if cmd == "tick":
            cnt = int(tokens[2]) if len(tokens) == 3 else 1
            self.project.get_section(tokens[1]).step(self.project, cnt)
            print(f"advanced {cnt} tick(s)"); return
        if cmd == "show":
            print(self.project.get_section(tokens[1]).render_mixed(self.project)); return
        if cmd == "signals":
            print(self.project.get_section(tokens[1]).render_signals(self.project)); return
        if cmd == "save":
            self.project.save(tokens[1]); print("saved"); return
        if cmd == "load":
            self.project = Project.load(tokens[1]); print("loaded"); return
        if cmd == "list":
            print(self.list_project()); return
        raise ScriptError(f"unknown command: {cmd}")

    def help_text(self) -> str:
        return "project NAME, section NAME W H, define NAME = EXPR, place SEC X Y KIND, tick SEC [N], show SEC, signals SEC, clear, list, save/load FILE, quit"

    def list_project(self) -> str:
        if not self.project: return "(no project)"
        return f"Project: {self.project.name}\nSections: {list(self.project.sections.keys())}\nMacros: {list(self.project.macros.keys())}"

# ---------------------------------------------------------------------------
# Main / REPL
# ---------------------------------------------------------------------------

def repl() -> None:
    interp = AbstrangInterpreter()
    print("BinaCraft REPL (type 'help' or 'quit')")
    while True:
        try:
            line = input(">> ").strip()
            if not line: continue
            if line.lower() in {"quit", "exit"}: break
            interp.run_line(line)
        except EOFError: break
        except Exception as e: print(f"error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repl", action="store_true")
    args = parser.parse_args()
    repl() if args.repl or sys.stdin.isatty() else None
