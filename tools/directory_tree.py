#!/usr/bin/env python3
"""
Print a readable file tree for a directory.

Examples:
    python directory_tree.py
    python directory_tree.py /path/to/project
    python directory_tree.py . --ignore-dir .git --ignore-dir node_modules
    python directory_tree.py . --ignore-ext .pyc --ignore-ext .log
    python directory_tree.py . --ignore-pattern "*.tmp" --ignore-file secrets.txt
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class EntryView:
    path: Path
    name: str
    relative_path: str
    is_dir: bool
    is_symlink: bool


@dataclass(frozen=True)
class IgnoreRules:
    directory_names: Set[str]
    file_names: Set[str]
    extensions: Tuple[str, ...]
    patterns: Tuple[str, ...]
    ignore_hidden: bool
    ignore_case: bool

    def normalize(self, value: str) -> str:
        return value.casefold() if self.ignore_case else value

    def should_ignore(self, name: str, relative_path: str, is_dir: bool) -> bool:
        if self.ignore_hidden and name.startswith("."):
            return True

        normalized_name = self.normalize(name)
        normalized_relative = self.normalize(relative_path)

        if is_dir:
            if normalized_name in self.directory_names:
                return True
        else:
            if normalized_name in self.file_names:
                return True
            if any(
                normalized_name.endswith(extension) for extension in self.extensions
            ):
                return True

        for pattern in self.patterns:
            if fnmatch.fnmatchcase(normalized_name, pattern) or fnmatch.fnmatchcase(
                normalized_relative, pattern
            ):
                return True

        return False


@dataclass(frozen=True)
class TreeStyle:
    branch: str
    last_branch: str
    vertical: str
    empty: str


UNICODE_STYLE = TreeStyle(
    branch="├── ",
    last_branch="└── ",
    vertical="│   ",
    empty="    ",
)

ASCII_STYLE = TreeStyle(
    branch="|-- ",
    last_branch="`-- ",
    vertical="|   ",
    empty="    ",
)


def non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def normalize_extension(extension: str, ignore_case: bool) -> str:
    extension = extension.strip()
    if not extension:
        raise ValueError("extensions cannot be empty")
    if not extension.startswith("."):
        extension = "." + extension
    return extension.casefold() if ignore_case else extension


def normalize_pattern(pattern: str, ignore_case: bool) -> str:
    # Tree-relative paths always use "/" so ignore patterns are portable.
    pattern = pattern.replace("\\", "/")
    return pattern.casefold() if ignore_case else pattern


def stdout_supports_box_drawing() -> bool:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        "├── └── │".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


class TreePrinter:
    def __init__(
        self,
        rules: IgnoreRules,
        style: TreeStyle,
        max_depth: Optional[int],
        follow_symlinks: bool,
        directories_first: bool,
    ) -> None:
        self.rules = rules
        self.style = style
        self.max_depth = max_depth
        self.follow_symlinks = follow_symlinks
        self.directories_first = directories_first

    def print(self, root: Path) -> None:
        root_display = self._absolute_display_path(root)
        if root.is_dir() and not root_display.endswith(("/", "\\")):
            root_display += os.sep
        print(root_display)

        if not root.is_dir():
            return

        ancestors: Set[Tuple[int, int]] = set()
        root_identity, _ = self._directory_identity(root)
        if root_identity is not None:
            ancestors.add(root_identity)

        self._walk(
            directory=root,
            prefix="",
            relative_prefix="",
            depth=0,
            ancestors=ancestors,
        )

    def _walk(
        self,
        directory: Path,
        prefix: str,
        relative_prefix: str,
        depth: int,
        ancestors: Set[Tuple[int, int]],
    ) -> None:
        if self.max_depth is not None and depth >= self.max_depth:
            return

        entries, error = self._scan(directory, relative_prefix)
        if error is not None:
            print(prefix + self.style.last_branch + f"[{error}]")
            return

        for index, entry in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = self.style.last_branch if is_last else self.style.branch
            child_prefix = prefix + (
                self.style.empty if is_last else self.style.vertical
            )

            display_name = self._display_name(entry)
            recurse = entry.is_dir and (self.follow_symlinks or not entry.is_symlink)
            child_identity: Optional[Tuple[int, int]] = None
            identity_error: Optional[str] = None
            cycle = False

            if recurse:
                child_identity, identity_error = self._directory_identity(entry.path)
                if child_identity is not None and child_identity in ancestors:
                    cycle = True
                    recurse = False

            note = ""
            if cycle:
                note = " [cycle detected]"
            elif identity_error is not None:
                note = f" [{identity_error}]"
                recurse = False

            print(prefix + connector + display_name + note)

            if recurse:
                next_ancestors = set(ancestors)
                if child_identity is not None:
                    next_ancestors.add(child_identity)

                self._walk(
                    directory=entry.path,
                    prefix=child_prefix,
                    relative_prefix=entry.relative_path,
                    depth=depth + 1,
                    ancestors=next_ancestors,
                )

    def _scan(
        self, directory: Path, relative_prefix: str
    ) -> Tuple[List[EntryView], Optional[str]]:
        entries: List[EntryView] = []

        try:
            with os.scandir(directory) as iterator:
                for item in iterator:
                    relative_path = (
                        f"{relative_prefix}/{item.name}"
                        if relative_prefix
                        else item.name
                    )

                    try:
                        is_symlink = item.is_symlink()
                    except OSError:
                        is_symlink = False

                    try:
                        # Determine the target type even when symlink traversal is disabled.
                        # This lets directory ignore rules apply to symlinked directories too.
                        is_dir = item.is_dir(follow_symlinks=True)
                    except OSError:
                        is_dir = False

                    if self.rules.should_ignore(item.name, relative_path, is_dir):
                        continue

                    entries.append(
                        EntryView(
                            path=Path(item.path),
                            name=item.name,
                            relative_path=relative_path.replace(os.sep, "/"),
                            is_dir=is_dir,
                            is_symlink=is_symlink,
                        )
                    )
        except PermissionError:
            return [], "permission denied"
        except OSError as exc:
            message = exc.strerror or str(exc)
            return [], f"unable to read: {message}"

        if self.directories_first:
            entries.sort(
                key=lambda entry: (
                    0 if entry.is_dir else 1,
                    entry.name.casefold(),
                    entry.name,
                )
            )
        else:
            entries.sort(key=lambda entry: (entry.name.casefold(), entry.name))

        return entries, None

    def _display_name(self, entry: EntryView) -> str:
        if entry.is_symlink:
            try:
                target = os.readlink(entry.path)
            except OSError:
                target = "?"
            directory_marker = "/" if entry.is_dir else ""
            return f"{entry.name}{directory_marker}@ -> {target}"

        return entry.name + ("/" if entry.is_dir else "")

    @staticmethod
    def _directory_identity(
        path: Path,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        try:
            stat_result = path.stat()
        except PermissionError:
            return None, "permission denied"
        except OSError as exc:
            message = exc.strerror or str(exc)
            return None, f"unable to access: {message}"

        return (stat_result.st_dev, stat_result.st_ino), None

    @staticmethod
    def _absolute_display_path(path: Path) -> str:
        try:
            return str(path.resolve())
        except OSError:
            return str(path.absolute())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a readable file tree for a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s
  %(prog)s /path/to/project
  %(prog)s . --ignore-dir .git --ignore-dir node_modules
  %(prog)s . --ignore-ext pyc --ignore-ext .log
  %(prog)s . --ignore-file secrets.txt --ignore-pattern "*.tmp"
  %(prog)s . --ignore-pattern "build/*" --max-depth 3
""",
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="directory or file to show (default: current directory)",
    )
    parser.add_argument(
        "-d",
        "--ignore-dir",
        action="append",
        default=[],
        metavar="NAME",
        help="ignore a directory with this exact name; repeat as needed",
    )
    parser.add_argument(
        "-f",
        "--ignore-file",
        action="append",
        default=[],
        metavar="NAME",
        help="ignore a file with this exact name; repeat as needed",
    )
    parser.add_argument(
        "-e",
        "--ignore-ext",
        action="append",
        default=[],
        metavar="EXT",
        help='ignore files ending in this extension, such as ".pyc" or "log"; repeat as needed',
    )
    parser.add_argument(
        "-p",
        "--ignore-pattern",
        action="append",
        default=[],
        metavar="GLOB",
        help='ignore names or root-relative paths matching a glob, such as "*.tmp"; repeat as needed',
    )
    parser.add_argument(
        "--ignore-hidden",
        action="store_true",
        help='ignore dot-prefixed entries such as ".git" and ".env"',
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="make all ignore matching case-insensitive",
    )
    parser.add_argument(
        "--max-depth",
        type=non_negative_int,
        metavar="N",
        help="show at most N levels below the root (0 shows only the root)",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="descend into symlinked directories; cycles are detected",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="use ASCII tree characters instead of box-drawing characters",
    )
    parser.add_argument(
        "--no-dirs-first",
        action="store_true",
        help="sort all entries together instead of listing directories first",
    )

    return parser


def make_rules(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> IgnoreRules:
    normalize = str.casefold if args.ignore_case else (lambda value: value)

    try:
        extensions = tuple(
            normalize_extension(extension, args.ignore_case)
            for extension in args.ignore_ext
        )
    except ValueError as exc:
        parser.error(str(exc))

    directory_names = {normalize(name) for name in args.ignore_dir}
    file_names = {normalize(name) for name in args.ignore_file}
    patterns = tuple(
        normalize_pattern(pattern, args.ignore_case) for pattern in args.ignore_pattern
    )

    return IgnoreRules(
        directory_names=directory_names,
        file_names=file_names,
        extensions=extensions,
        patterns=patterns,
        ignore_hidden=args.ignore_hidden,
        ignore_case=args.ignore_case,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser()
    if not root.exists():
        parser.error(f"path does not exist: {root}")

    rules = make_rules(args, parser)
    style = (
        ASCII_STYLE
        if args.ascii or not stdout_supports_box_drawing()
        else UNICODE_STYLE
    )

    printer = TreePrinter(
        rules=rules,
        style=style,
        max_depth=args.max_depth,
        follow_symlinks=args.follow_symlinks,
        directories_first=not args.no_dirs_first,
    )

    try:
        printer.print(root)
    except BrokenPipeError:
        # Avoid a traceback when output is piped into tools such as "head".
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
