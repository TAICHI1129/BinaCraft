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

The simulation is synchronous:
each tick reads the previous tick state of neighboring cells and computes
the next state for every cell at once.
"""

from __future__ import annotations

import ast
import json
import sys
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
                        if not result:
                            break
                    return result
                if isinstance(node.op, ast.Or):
                    result = False
                    for value in node.values:
                        result = result or walk(value)
                        if result:
                            break
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
    state: int = 0  # used by switches
    macro: Optional[str] = None
    label: str = ""

    def output(self, section: "Section", x: int, y: int, previous: List[List[int]], project: "Project") -> int:
        if self.kind == KIND_EMPTY:
            return 0
        if self.kind == KIND_POWER:
            return 1
        if self.kind == KIND_SWITCH:
            return int(bool(self.state))
        if self.kind == KIND_WIRE:
            return int(any(section.neighbor_outputs(previous, x, y)))
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
            if not self.macro:
                return 0
            macro = project.macros.get(self.macro)
            if not macro:
                return 0
            return macro.evaluate(section, x, y, previous, project)
        return 0

    def symbol(self, state: int) -> str:
        if self.kind == KIND_EMPTY:
            return "."
        if self.kind == KIND_SWITCH:
            return "T" if state else "t"
        if self.kind == KIND_POWER:
            return "P"
        if self.kind == KIND_WIRE:
            return "#" if state else ":"
        if self.kind == KIND_AND:
            return "A" if state else "a"
        if self.kind == KIND_OR:
            return "O" if state else "o"
        if self.kind == KIND_NOT:
            return "N" if state else "n"
        if self.kind == KIND_MACRO:
            return "M" if state else "m"
        return "?"

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "state": self.state,
            "macro": self.macro,
            "label": self.label,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Cell":
        return cls(
            kind=data.get("kind", KIND_EMPTY),
            state=int(data.get("state", 0)),
            macro=data.get("macro"),
            label=data.get("label", ""),
        )


@dataclass
class MacroDef:
    """Reusable custom switch built from a boolean expression."""

    name: str
    expression: BoolExpr
    description: str = ""

    def evaluate(self, section: "Section", x: int, y: int, previous: List[List[int]], project: "Project") -> int:
        values = section.inputs_dict(previous, x, y)
        return int(self.expression.eval(values))

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression.to_json(),
            "description": self.description,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MacroDef":
        return cls(
            name=data["name"],
            expression=BoolExpr.from_json(data["expression"]),
            description=data.get("description", ""),
        )


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
        if not self.in_bounds(x, y):
            raise IndexError(f"cell out of bounds: ({x}, {y})")
        return self.grid[y][x]

    def set_cell(self, x: int, y: int, cell: Cell) -> None:
        if not self.in_bounds(x, y):
            raise IndexError(f"cell out of bounds: ({x}, {y})")
        self.grid[y][x] = cell

    def place(self, x: int, y: int, kind: str, *, macro: Optional[str] = None, state: int = 0, label: str = "") -> None:
        if kind not in {KIND_EMPTY, KIND_POWER, KIND_WIRE, KIND_SWITCH, KIND_AND, KIND_OR, KIND_NOT, KIND_MACRO}:
            raise ValueError(f"unknown block kind: {kind}")
        self.set_cell(x, y, Cell(kind=kind, state=int(state), macro=macro, label=label))

    def toggle(self, x: int, y: int) -> None:
        c = self.cell(x, y)
        if c.kind != KIND_SWITCH:
            raise ValueError("only switches can be toggled")
        c.state = 0 if c.state else 1

    def set_switch(self, x: int, y: int, state: int) -> None:
        c = self.cell(x, y)
        if c.kind != KIND_SWITCH:
            raise ValueError("only switches can be set")
        c.state = int(bool(state))

    def neighbor_outputs(self, previous: List[List[int]], x: int, y: int) -> List[int]:
        outs = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                outs.append(int(previous[ny][nx]))
        return outs

    def in_dir(self, previous: List[List[int]], x: int, y: int, direction: str) -> int:
        mapping = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
        dx, dy = mapping[direction]
        nx, ny = x + dx, y + dy
        if not self.in_bounds(nx, ny):
            return 0
        return int(previous[ny][nx])

    def inputs_dict(self, previous: List[List[int]], x: int, y: int) -> Dict[str, int]:
        n = self.in_dir(previous, x, y, "N")
        e = self.in_dir(previous, x, y, "E")
        s = self.in_dir(previous, x, y, "S")
        w = self.in_dir(previous, x, y, "W")
        vals = {"N": n, "E": e, "S": s, "W": w}
        vals["ANY"] = int(bool(n or e or s or w))
        vals["ALL"] = int(bool(n and e and s and w))
        vals["A"] = n
        vals["B"] = e
        vals["C"] = s
        vals["D"] = w
        return vals

    def snapshot(self) -> List[List[int]]:
        if self.last_state is not None:
            return [row[:] for row in self.last_state]
        return [[cell_output(c) for c in row] for row in self.grid]

    def tick(self, project: "Project") -> List[List[int]]:
        previous = self.snapshot()
        next_state = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                next_state[y][x] = self.grid[y][x].output(self, x, y, previous, project)
        return next_state

    def step(self, project: "Project", count: int = 1) -> None:
        for _ in range(max(0, int(count))):
            next_state = self.tick(project)
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.grid[y][x]
                    if cell.kind == KIND_POWER:
                        continue
                    if cell.kind == KIND_SWITCH:
                        continue
                    # store transient output in label? no, grid stores kind/state only.
                    # The rendered output is derived from the grid and a separate snapshot.
            # In this prototype, cell types hold logic, not the last output state.
            # So step is intentionally a no-op for internal state changes other than switches.
            # The simulation output is computed during render() and inspect() using tick().
            # To make stateful elements possible later, this is where they would live.
            pass

    def render_kinds(self) -> str:
        lines = [f"Section {self.name!r} [{self.width}x{self.height}] kinds:"]
        for row in self.grid:
            lines.append(" ".join(KIND_CHARS.get(cell.kind, "?") for cell in row))
        return "\n".join(lines)

    def render_signals(self, project: "Project") -> str:
        state = self.snapshot()
        lines = [f"Section {self.name!r} [{self.width}x{self.height}] signals:"]
        for y, row in enumerate(self.grid):
            out_row = []
            for x, cell in enumerate(row):
                out_row.append(str(cell.output(self, x, y, state, project)))
            lines.append(" ".join(out_row))
        return "\n".join(lines)

    def render_mixed(self, project: "Project") -> str:
        state = self.snapshot()
        lines = [f"Section {self.name!r} [{self.width}x{self.height}]"]
        for y, row in enumerate(self.grid):
            out_row = []
            for x, cell in enumerate(row):
                out_row.append(cell.symbol(cell.output(self, x, y, state, project)))
            lines.append(" ".join(out_row))
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "grid": [[cell.to_json() for cell in row] for row in self.grid],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Section":
        sec = cls(name=data["name"], width=int(data["width"]), height=int(data["height"]))
        grid_data = data.get("grid", [])
        for y, row in enumerate(grid_data[: sec.height]):
            for x, cell_data in enumerate(row[: sec.width]):
                sec.grid[y][x] = Cell.from_json(cell_data)
        return sec


def cell_output(cell: Cell) -> int:
    if cell.kind == KIND_POWER:
        return 1
    if cell.kind == KIND_SWITCH:
        return int(bool(cell.state))
    return 0


@dataclass
class Project:
    name: str
    sections: Dict[str, Section] = field(default_factory=dict)
    macros: Dict[str, MacroDef] = field(default_factory=dict)

    def add_section(self, name: str, width: int, height: int) -> Section:
        if name in self.sections:
            raise ValueError(f"section already exists: {name}")
        sec = Section(name=name, width=width, height=height)
        self.sections[name] = sec
        return sec

    def get_section(self, name: str) -> Section:
        try:
            return self.sections[name]
        except KeyError as exc:
            raise KeyError(f"unknown section: {name}") from exc

    def define_macro(self, name: str, expression: str, description: str = "") -> MacroDef:
        macro = MacroDef(name=name, expression=BoolExpr(expression), description=description)
        self.macros[name] = macro
        return macro

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sections": {name: sec.to_json() for name, sec in self.sections.items()},
            "macros": {name: macro.to_json() for name, macro in self.macros.items()},
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Project":
        proj = cls(name=data["name"])
        for name, sec_data in data.get("sections", {}).items():
            proj.sections[name] = Section.from_json(sec_data)
        for name, macro_data in data.get("macros", {}).items():
            proj.macros[name] = MacroDef.from_json(macro_data)
        return proj

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_json(data)


# ---------------------------------------------------------------------------
# Abstrang-like script interpreter
# ---------------------------------------------------------------------------

class ScriptError(RuntimeError):
    pass


class AbstrangInterpreter:
    """
    Tiny command language.

    Commands:
      project NAME
      section NAME WIDTH HEIGHT
      define NAME = EXPRESSION
      place SECTION X Y KIND [extra]
      toggle SECTION X Y
      set SECTION X Y on|off
      show SECTION
      kinds SECTION
      signals SECTION
      save FILE
      load FILE
      help
      quit

    Examples:
      project BinaCraft
      section main 16 8
      define XOR = (N and not E) or (not N and E)
      place main 1 1 power
      place main 2 1 wire
      place main 3 1 switch on
      place main 4 1 and
      place main 5 1 macro XOR
      show main
    """

    def __init__(self):
        self.project: Optional[Project] = None

    @staticmethod
    def _tokens(line: str) -> List[str]:
        # Keep it simple: split by spaces, but allow '=' as its own token.
        raw = line.replace("=", " = ").split()
        return raw

    def run_lines(self, lines: Iterable[str]) -> None:
        for lineno, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                self.run_line(line)
            except Exception as exc:
                raise ScriptError(f"line {lineno}: {line}\n  {exc}") from exc

    def run_line(self, line: str) -> None:
        tokens = self._tokens(line)
        if not tokens:
            return

        cmd = tokens[0].lower()

        if cmd == "help":
            print(self.help_text())
            return

        if cmd == "project":
            if len(tokens) < 2:
                raise ScriptError("project name is required")
            self.project = Project(tokens[1])
            print(f"created project {tokens[1]!r}")
            return

        if self.project is None:
            raise ScriptError("create or load a project first")

        if cmd == "section":
            if len(tokens) != 4:
                raise ScriptError("usage: section NAME WIDTH HEIGHT")
            name = tokens[1]
            width = int(tokens[2])
            height = int(tokens[3])
            self.project.add_section(name, width, height)
            print(f"added section {name!r}")
            return

        if cmd in {"define", "switchdef"}:
            if "=" not in tokens:
                raise ScriptError("usage: define NAME = EXPRESSION")
            eq = tokens.index("=")
            if eq < 2:
                raise ScriptError("usage: define NAME = EXPRESSION")
            name = tokens[1]
            expr = " ".join(tokens[eq + 1 :])
            self.project.define_macro(name, expr)
            print(f"defined macro {name!r} = {expr}")
            return

        if cmd == "place":
            if len(tokens) < 5:
                raise ScriptError("usage: place SECTION X Y KIND [extra]")
            sec = self.project.get_section(tokens[1])
            x = int(tokens[2])
            y = int(tokens[3])
            kind = tokens[4].lower()
            state = 0
            macro = None
            label = ""

            extra = [t.lower() for t in tokens[5:]]
            if kind == KIND_SWITCH:
                if extra and extra[0] in {"on", "1", "true"}:
                    state = 1
                elif extra and extra[0] in {"off", "0", "false"}:
                    state = 0
                elif extra:
                    raise ScriptError("switch extra must be on/off")
            elif kind == KIND_MACRO:
                if not extra:
                    raise ScriptError("usage: place SECTION X Y macro MACRO_NAME")
                macro = tokens[5]
                if macro not in self.project.macros:
                    raise ScriptError(f"unknown macro: {macro}")
            elif extra:
                label = " ".join(tokens[5:])

            sec.place(x, y, kind, macro=macro, state=state, label=label)
            print(f"placed {kind!r} at {sec.name} ({x},{y})")
            return

        if cmd == "toggle":
            if len(tokens) != 4:
                raise ScriptError("usage: toggle SECTION X Y")
            sec = self.project.get_section(tokens[1])
            sec.toggle(int(tokens[2]), int(tokens[3]))
            print(f"toggled switch at {sec.name} ({tokens[2]},{tokens[3]})")
            return

        if cmd == "set":
            if len(tokens) != 5:
                raise ScriptError("usage: set SECTION X Y on|off")
            sec = self.project.get_section(tokens[1])
            state = 1 if tokens[4].lower() in {"on", "1", "true"} else 0
            sec.set_switch(int(tokens[2]), int(tokens[3]), state)
            print(f"set switch at {sec.name} ({tokens[2]},{tokens[3]}) = {state}")
            return

        if cmd == "show":
            if len(tokens) != 2:
                raise ScriptError("usage: show SECTION")
            sec = self.project.get_section(tokens[1])
            print(sec.render_mixed(self.project))
            return

        if cmd == "kinds":
            if len(tokens) != 2:
                raise ScriptError("usage: kinds SECTION")
            sec = self.project.get_section(tokens[1])
            print(sec.render_kinds())
            return

        if cmd == "signals":
            if len(tokens) != 2:
                raise ScriptError("usage: signals SECTION")
            sec = self.project.get_section(tokens[1])
            print(sec.render_signals(self.project))
            return

        if cmd == "tick":
            if len(tokens) not in {2, 3}:
                raise ScriptError("usage: tick SECTION [COUNT]")
            sec = self.project.get_section(tokens[1])
            count = int(tokens[2]) if len(tokens) == 3 else 1
            sec.step(self.project, count)
            print(f"advanced {sec.name} by {count} tick(s)")
            return

        if cmd == "save":
            if len(tokens) != 2:
                raise ScriptError("usage: save FILE")
            self.project.save(tokens[1])
            print(f"saved to {tokens[1]!r}")
            return

        if cmd == "load":
            if len(tokens) != 2:
                raise ScriptError("usage: load FILE")
            self.project = Project.load(tokens[1])
            print(f"loaded project {self.project.name!r}")
            return

        if cmd == "list":
            print(self.list_project())
            return

        raise ScriptError(f"unknown command: {cmd}")

    def help_text(self) -> str:
        return (
            "Commands:\n"
            "  project NAME\n"
            "  section NAME WIDTH HEIGHT\n"
            "  define NAME = EXPRESSION\n"
            "  place SECTION X Y KIND [extra]\n"
            "    kinds: empty, power, wire, switch, and, or, not, macro\n"
            "  toggle SECTION X Y\n"
            "  set SECTION X Y on|off\n"
            "  tick SECTION [COUNT]\n"
            "  show SECTION\n"
            "  kinds SECTION\n"
            "  signals SECTION\n"
            "  list\n"
            "  save FILE\n"
            "  load FILE\n"
            "  help\n"
            "  quit"
        )

    def list_project(self) -> str:
        if self.project is None:
            return "(no project)"
        lines = [f"Project: {self.project.name}"]
        if self.project.macros:
            lines.append("Macros:")
            for name, macro in self.project.macros.items():
                lines.append(f"  - {name}: {macro.expression.expression}")
        else:
            lines.append("Macros: (none)")
        if self.project.sections:
            lines.append("Sections:")
            for sec in self.project.sections.values():
                lines.append(f"  - {sec.name} {sec.width}x{sec.height}")
        else:
            lines.append("Sections: (none)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo / REPL
# ---------------------------------------------------------------------------

DEMO_SCRIPT = """
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
show main
signals main
"""


def repl() -> None:
    interp = AbstrangInterpreter()
    print("BinaCraft / Abstrang REPL")
    print("type 'help' for commands, 'quit' to exit")
    while True:
        try:
            line = input(">> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        try:
            interp.run_line(line)
        except Exception as exc:
            print(f"error: {exc}")


def run_script_text(text: str) -> None:
    interp = AbstrangInterpreter()
    interp.run_lines(text.splitlines())


def main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="BinaCraft", description="Binary sandbox prototype")
    parser.add_argument("--script", type=str, help="run an Abstrang script file")
    parser.add_argument("--save-demo", type=str, help="write the demo script to a file and exit")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo")
    parser.add_argument("--repl", action="store_true", help="start an interactive REPL")
    args = parser.parse_args(argv)

    if args.save_demo:
        Path(args.save_demo).write_text(DEMO_SCRIPT.strip() + "\n", encoding="utf-8")
        print(f"demo script written to {args.save_demo}")
        return 0

    if args.script:
        text = Path(args.script).read_text(encoding="utf-8")
        run_script_text(text)
        return 0

    if args.demo:
        run_script_text(DEMO_SCRIPT)
        return 0

    if args.repl or sys.stdin.isatty():
        repl()
        return 0

    # pipe mode: read script from stdin
    text = sys.stdin.read()
    if text.strip():
        run_script_text(text)
    else:
        print("No input. Use --demo, --repl, or --script FILE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
