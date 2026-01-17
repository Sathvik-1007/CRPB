"""Deprecated module.

This file used to contain a deterministic, language-specific project crawler.

Per current CRPB requirements, project validation is **LLM-only** and lives in
`crpb.validation.project_validate`.

This stub remains only to fail loudly if something still imports it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def validate_project_outputs_deterministic(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    raise RuntimeError(
        "Deterministic project crawling has been removed. "
        "Use crpb.validation.project_validate.validate_project_outputs instead."
    )


def validate_project_outputs(
    *,
    outputs_dir: Path,
    constraints: Optional[Dict[str, Any]] = None,
    idea: str = "",
    plan: Optional[Dict[str, Any]] = None,
    file_specs: Optional[Dict[str, Any]] = None,
    validations_dir: Optional[Path] = None,
    label: str = "project",
) -> Dict[str, Any]:
    from .project_validate import validate_project_outputs as _validate

    return _validate(
        outputs_dir=outputs_dir,
        constraints=constraints,
        idea=idea,
        plan=plan,
        file_specs=file_specs,
        validations_dir=validations_dir,
        label=label,
    )


'''


@dataclass(frozen=True)
class JsImport:
    module: str
    named: Tuple[str, ...] = ()
    wants_default: bool = False


@dataclass(frozen=True)
class JsModuleInfo:
    has_module_syntax: bool
    exports: Set[str]
    has_default_export: bool
    imports: Tuple[JsImport, ...]


@dataclass(frozen=True)
class _Tok:
    t: str  # 'id' | 'str' | 'punc' | 'eof'
    v: str


class _JsLexer:
    def __init__(self, text: str) -> None:
        self._s = text or ""
        self._n = len(self._s)
        self._i = 0

    def _peek(self, k: int = 0) -> str:
        j = self._i + k
        return self._s[j] if 0 <= j < self._n else ""

    def _adv(self, k: int = 1) -> None:
        self._i = min(self._n, self._i + k)

    def _skip_ws_and_comments(self) -> None:
        while True:
            # whitespace
            while self._peek() and self._peek().isspace():
                self._adv(1)
            # line comment
            if self._peek() == "/" and self._peek(1) == "/":
                self._adv(2)
                while self._peek() and self._peek() not in "\r\n":
                    self._adv(1)
                continue
            # block comment
            if self._peek() == "/" and self._peek(1) == "*":
                self._adv(2)
                while self._peek():
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._adv(2)
                        break
                    self._adv(1)
                continue
            break

    def next(self) -> _Tok:
        self._skip_ws_and_comments()
        ch = self._peek()
        if not ch:
            return _Tok("eof", "")

        # string
        if ch in ("'", '"', "`"):
            q = ch
            self._adv(1)
            buf: List[str] = []
            while self._peek():
                c = self._peek()
                if c == "\\":
                    # escape
                    self._adv(1)
                    if self._peek():
                        buf.append(self._peek())
                        self._adv(1)
                    continue
                if c == q:
                    self._adv(1)
                    break
                buf.append(c)
                self._adv(1)
            return _Tok("str", "".join(buf))

        # identifier
        if ch.isalpha() or ch in ("_", "$"):
            buf = [ch]
            self._adv(1)
            while self._peek() and (self._peek().isalnum() or self._peek() in ("_", "$")):
                buf.append(self._peek())
                self._adv(1)
            return _Tok("id", "".join(buf))

        # punctuation (single char is enough for our purposes)
        self._adv(1)
        return _Tok("punc", ch)


def _parse_js_module(text: str) -> JsModuleInfo:
    lex = _JsLexer(text)
    exports: Set[str] = set()
    has_default = False
    imports: List[JsImport] = []
    has_module_syntax = False

    def _eat_until_semicolon_or_eof(tok: _Tok) -> _Tok:
        cur = tok
        while cur.t != "eof" and not (cur.t == "punc" and cur.v == ";"):
            cur = lex.next()
        return lex.next()

    tok = lex.next()
    while tok.t != "eof":
        if tok.t == "id" and tok.v in ("import", "export"):
            has_module_syntax = True

        # import ...
        if tok.t == "id" and tok.v == "import":
            named: List[str] = []
            wants_default = False

            t1 = lex.next()
            # import "./x.js";
            if t1.t == "str":
                imports.append(JsImport(module=t1.v, named=(), wants_default=False))
                tok = _eat_until_semicolon_or_eof(lex.next())
                continue

            # import defaultName ...
            if t1.t == "id":
                wants_default = True
                t2 = lex.next()
                # import defaultName, { a, b as c } from "...";
                if t2.t == "punc" and t2.v == ",":
                    t3 = lex.next()
                    t1 = t3
                else:
                    t1 = t2

            # import { a as b, c } from "...";
            if t1.t == "punc" and t1.v == "{":
                while True:
                    tname = lex.next()
                    if tname.t == "punc" and tname.v == "}":
                        break
                    if tname.t == "id":
                        orig = tname.v
                        ta = lex.next()
                        if ta.t == "id" and ta.v == "as":
                            talias = lex.next()  # alias name
                            _ = talias
                            tnext = lex.next()
                            if tnext.t == "punc" and tnext.v == ",":
                                continue
                            if tnext.t == "punc" and tnext.v == "}":
                                break
                            # otherwise keep scanning
                            continue
                        # no alias: keep this name
                        named.append(orig)
                        if ta.t == "punc" and ta.v == ",":
                            continue
                        if ta.t == "punc" and ta.v == "}":
                            break
                        # something else, keep scanning until }
                # expect from "..."
                tf = lex.next()
                while tf.t != "eof" and not (tf.t == "id" and tf.v == "from"):
                    tf = lex.next()
                tm = lex.next()
                if tm.t == "str":
                    imports.append(JsImport(module=tm.v, named=tuple(named), wants_default=wants_default))
                tok = _eat_until_semicolon_or_eof(lex.next())
                continue

            # import * as ns from "...";
            if t1.t == "punc" and t1.v == "*":
                tf = lex.next()
                while tf.t != "eof" and not (tf.t == "id" and tf.v == "from"):
                    tf = lex.next()
                tm = lex.next()
                if tm.t == "str":
                    imports.append(JsImport(module=tm.v, named=(), wants_default=wants_default))
                tok = _eat_until_semicolon_or_eof(lex.next())
                continue

            # fallback: consume statement
            tok = _eat_until_semicolon_or_eof(t1)
            continue

        # export ...
        if tok.t == "id" and tok.v == "export":
            t1 = lex.next()
            if t1.t == "id" and t1.v == "default":
                has_default = True
                tok = _eat_until_semicolon_or_eof(lex.next())
                continue

            # export { a as b, c } (from "...")?
            if t1.t == "punc" and t1.v == "{":
                while True:
                    tname = lex.next()
                    if tname.t == "punc" and tname.v == "}":
                        break
                    if tname.t == "id":
                        orig = tname.v
                        exports.add(orig)
                        ta = lex.next()
                        if ta.t == "id" and ta.v == "as":
                            talias = lex.next()
                            if talias.t == "id":
                                exports.add(talias.v)
                            tnext = lex.next()
                            if tnext.t == "punc" and tnext.v == ",":
                                continue
                            if tnext.t == "punc" and tnext.v == "}":
                                break
                            continue
                        if ta.t == "punc" and ta.v == ",":
                            continue
                        if ta.t == "punc" and ta.v == "}":
                            break
                tok = _eat_until_semicolon_or_eof(lex.next())
                continue

            # export async function name
            if t1.t == "id" and t1.v == "async":
                t1 = lex.next()

            if t1.t == "id" and t1.v in ("function", "class"):
                tname = lex.next()
                if tname.t == "id" and tname.v:
                    exports.add(tname.v)
                tok = _eat_until_semicolon_or_eof(lex.next())
                continue

            if t1.t == "id" and t1.v in ("const", "let", "var"):
                tname = lex.next()
                if tname.t == "id" and tname.v:
                    exports.add(tname.v)
                tok = _eat_until_semicolon_or_eof(lex.next())
                continue

            tok = _eat_until_semicolon_or_eof(t1)
            continue

        tok = lex.next()

    return JsModuleInfo(
        has_module_syntax=has_module_syntax,
        exports=exports,
        has_default_export=has_default,
        imports=tuple(imports),
    )


def _is_remote_ref(ref: str) -> bool:
    s = (ref or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("//")


def _is_non_file_like_ref(ref: str) -> bool:
    """Return True for refs that are not expected to resolve to project files.

    Examples: fragments, mailto/tel/javascript/data URLs.
    """
    s = (ref or "").strip()
    if not s:
        return True
    if s.startswith("#"):
        return True
    low = s.lower()
    if low.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return True
    return False


def _strip_query_and_fragment(ref: str) -> str:
    s = (ref or "").strip()
    if not s:
        return ""
    # Remove fragments and queries for file existence checks.
    s = s.split("#", 1)[0]
    s = s.split("?", 1)[0]
    return s.strip()


def _is_local_module_ref(ref: str) -> bool:
    """Return True if the import specifier is a local path we can resolve.

    Bare specifiers like 'react' or '@playwright/test' are treated as external
    dependencies and are not validated as local files.
    """
    s = str(ref or "").strip()
    if not s:
        return False
    return s.startswith("./") or s.startswith("../") or s.startswith("/")


def _norm_rel(p: str) -> str:
    s = str(p or "").replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def _collapse_rel_posix(p: str) -> Optional[str]:
    """Normalize a posix-ish relative path by collapsing '.' and '..'.

    Returns None if the path would escape above the root.
    """

    parts = [seg for seg in _norm_rel(p).split("/") if seg not in ("", ".")]
    stack: List[str] = []
    for seg in parts:
        if seg == "..":
            if not stack:
                return None
            stack.pop()
            continue
        stack.append(seg)
    return "/".join(stack)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            # skip staging/hidden run internals
            # If validating the main outputs dir, ignore nested outputs/_staging.
            # If validating the staging dir directly, do NOT exclude everything.
            if "_staging" in p.parts and Path(root).name != "_staging":
                continue
            yield p


def _build_index(root: Path) -> Set[str]:
    out: Set[str] = set()
    for p in _iter_files(root):
        rel = p.relative_to(root)
        out.add(_norm_rel(str(rel.as_posix())))
    return out


def _resolve_ref(*, from_rel: str, ref: str, index: Set[str]) -> Optional[str]:
    ref_s = _strip_query_and_fragment(ref)
    if not ref_s or _is_remote_ref(ref_s):
        return None

    # Treat leading '/' as project-root relative.
    if ref_s.startswith("/"):
        ref_s = ref_s[1:]

    from_dir = PurePosixPath(_norm_rel(from_rel)).parent
    raw_target = (from_dir / PurePosixPath(_norm_rel(ref_s))).as_posix()
    target = _collapse_rel_posix(raw_target)
    if target is None:
        return None
    if target in index:
        return target

    # IMPORTANT (language/tool neutrality): do NOT guess extensions.
    # If a ref omits an extension, treat it as unresolved here and let LLM-based
    # validation reason about intended targets (if needed).

    return None


def validate_project_outputs_deterministic(
    *,
    outputs_dir: Path,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic, language-agnostic project validation.

    This validator crawls the produced file tree and checks cross-file integrity
    without relying on an LLM context window.

    It is intentionally conservative: it validates *declared* references.
    It does not attempt to infer "connectivity" via regex heuristics.
    """
    logger.info("[project_crawl] Starting deterministic validation of: %s", outputs_dir)

    issues: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    root = Path(outputs_dir)
    if not root.exists() or not root.is_dir():
        logger.error("[project_crawl] Output directory missing: %s", outputs_dir)
        return {
            "ok": False,
            "issues": ["outputs_dir_missing"],
            "warnings": [],
            "suggestions": [],
        }

    index = _build_index(root)
    logger.debug("[project_crawl] File index built with %d files", len(index))

    # Precompute directory presence for cheap module/package existence checks.
    dir_index: Set[str] = set()
    try:
        for p in index:
            pp = PurePosixPath(p)
            for parent in pp.parents:
                s = str(parent.as_posix())
                if s and s != ".":
                    dir_index.add(s)
    except Exception:
        dir_index = set()

    # --- HTML: assets + module script correctness ---
    html_parsed: Dict[str, _HtmlRefParser] = {}
    for rel in sorted([p for p in index if p.lower().endswith(".html")]):
        logger.debug("[project_crawl] Validating HTML file: %s", rel)
        try:
            txt = (root / rel).read_text(encoding="utf-8")
        except Exception:
            issues.append(f"file_unreadable:{rel}")
            logger.warning("[project_crawl] HTML file unreadable: %s", rel)
            continue
        parser = _HtmlRefParser()
        try:
            parser.feed(txt)
        except Exception:
            warnings.append(f"html_parse_error:{rel}")
            logger.warning("[project_crawl] HTML parse error: %s", rel)
        html_parsed[rel] = parser

        for ss in parser.stylesheets:
            if _is_remote_ref(ss.href):
                issues.append(f"remote_resource_not_allowed:{rel}:{ss.href}")
                logger.warning("[project_crawl] Remote stylesheet not allowed: %s -> %s", rel, ss.href)
                continue
            resolved = _resolve_ref(from_rel=rel, ref=ss.href, index=index)
            if resolved is None:
                issues.append(f"missing_stylesheet_ref:{rel}:{ss.href}")
                logger.warning("[project_crawl] Missing stylesheet: %s -> %s", rel, ss.href)

        for sc in parser.scripts:
            if _is_remote_ref(sc.src):
                issues.append(f"remote_resource_not_allowed:{rel}:{sc.src}")
                logger.warning("[project_crawl] Remote script not allowed: %s -> %s", rel, sc.src)
                continue
            resolved = _resolve_ref(from_rel=rel, ref=sc.src, index=index)
            if resolved is None:
                issues.append(f"missing_script_ref:{rel}:{sc.src}")
                logger.warning("[project_crawl] Missing script: %s -> %s", rel, sc.src)
                continue
            # Module correctness: if referenced JS uses module syntax, script must be type=module
            if resolved.lower().endswith((".js", ".mjs", ".cjs", ".ts")):
                try:
                    jstxt = (root / resolved).read_text(encoding="utf-8")
                    minfo = _parse_js_module(jstxt)
                    if minfo.has_module_syntax and sc.type_attr != "module":
                        issues.append(f"html_script_requires_module:{rel}:{sc.src}")
                except Exception:
                    warnings.append(f"script_unreadable_or_unparseable:{resolved}")

        # Other HTML refs: validate conservatively (avoid treating routes as files).
        for rr in parser.refs:
            if _is_non_file_like_ref(rr):
                continue
            if _is_remote_ref(rr):
                # Deterministic policy: remote refs are disallowed (offline integrity).
                issues.append(f"remote_resource_not_allowed:{rel}:{rr}")
                continue
            rr2 = _strip_query_and_fragment(rr)
            if not rr2:
                continue
            # Only validate as a file if it looks like a file ref (has extension) or is a relative path.
            looks_file_like = ("." in PurePosixPath(rr2).name) or rr2.startswith(("./", "../", "/"))
            if not looks_file_like:
                continue
            resolved = _resolve_ref(from_rel=rel, ref=rr2, index=index)
            if resolved is None:
                issues.append(f"missing_html_ref:{rel}:{rr}")

    # --- JS/TS: import resolution + export presence (when parseable) ---
    js_infos: Dict[str, JsModuleInfo] = {}
    for rel in sorted([p for p in index if p.lower().endswith((".js", ".mjs", ".cjs", ".ts"))]):
        try:
            txt = (root / rel).read_text(encoding="utf-8")
        except Exception:
            issues.append(f"file_unreadable:{rel}")
            continue
        try:
            js_infos[rel] = _parse_js_module(txt)
        except Exception:
            warnings.append(f"js_parse_error:{rel}")

    for rel, info in js_infos.items():
        for imp in info.imports:
            if _is_remote_ref(imp.module):
                issues.append(f"remote_resource_not_allowed:{rel}:{imp.module}")
                continue
            # Only resolve local file imports. Bare package imports are allowed.
            if not _is_local_module_ref(imp.module):
                continue
            resolved = _resolve_ref(from_rel=rel, ref=imp.module, index=index)
            if resolved is None:
                issues.append(f"js_import_missing_file:{rel}:{imp.module}")
                continue
            target = js_infos.get(resolved)
            if target is None:
                # Imported something non-JS/TS; allowed, but we can't verify named exports.
                continue

            if imp.wants_default and not target.has_default_export:
                warnings.append(f"js_import_default_without_default_export:{rel}:{imp.module}")

            for name in imp.named:
                if name and name not in target.exports:
                    issues.append(f"js_import_missing_export:{rel}:{imp.module}:{name}")

    # --- Python: basic import validation (relative + best-effort local absolute modules) ---
    py_files = sorted([p for p in index if p.lower().endswith(".py")])
    for rel in py_files:
        logger.debug("[project_crawl] Validating Python file: %s", rel)
        try:
            txt = (root / rel).read_text(encoding="utf-8")
        except Exception:
            issues.append(f"file_unreadable:{rel}")
            continue

        def _dir_of(path_rel: str) -> str:
            d = str(PurePosixPath(path_rel).parent.as_posix())
            return "" if d == "." else d

        def _py_mod_exists(mod_path: str) -> bool:
            mp = str(mod_path or "").strip().replace("\\", "/")
            if not mp:
                return False
            # Accept either as a concrete file, a package dir (namespace ok), or a package __init__.
            if f"{mp}.py" in index:
                return True
            if mp in dir_index:
                return True
            if f"{mp}/__init__.py" in index:
                return True
            return False

        def _validate_absolute_module(mod: str) -> None:
            s = str(mod or "").strip()
            if not s:
                return
            # Only validate absolute imports that *look* local: first segment exists in project.
            parts = [p for p in s.split(".") if p]
            if not parts:
                return
            first = parts[0]
            if not (_py_mod_exists(first)):
                return  # likely external dependency; do not flag
            # Validate progressively: engine.engine => engine and engine/engine
            cur = ""
            for seg in parts:
                cur = f"{cur}.{seg}" if cur else seg
                cur_path = cur.replace(".", "/")
                if not _py_mod_exists(cur_path):
                    issues.append(f"py_import_missing_module:{rel}:{cur}")
                    logger.warning("[project_crawl] Python import not found: %s -> %s", rel, cur)
                    return

        def _validate_relative_module(level: int, mod: Optional[str]) -> None:
            # level=1 means "from . ..." relative to current package.
            rel_dir = _dir_of(rel)
            # Walk up (level-1) directories.
            target_dir = rel_dir
            for _ in range(max(0, int(level) - 1)):
                if "/" in target_dir:
                    target_dir = target_dir.rsplit("/", 1)[0]
                else:
                    target_dir = ""

            if not mod:
                return
            mod_path = mod.replace(".", "/")
            full = f"{target_dir}/{mod_path}" if target_dir else mod_path
            if not _py_mod_exists(full):
                issues.append(f"py_import_missing_module:{rel}:{'.' * int(level)}{mod}")
                logger.warning("[project_crawl] Python relative import not found: %s -> %s", rel, mod)

        # Parse with AST (more robust than regex) and validate imports.
        try:
            tree = ast.parse(txt)
        except Exception:
            warnings.append(f"py_parse_error:{rel}")
            continue

        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for alias in n.names or []:
                    if getattr(alias, "name", None):
                        _validate_absolute_module(str(alias.name))
            elif isinstance(n, ast.ImportFrom):
                level = int(getattr(n, "level", 0) or 0)
                mod = getattr(n, "module", None)
                if level and level > 0:
                    _validate_relative_module(level, str(mod) if mod else None)
                else:
                    if mod:
                        _validate_absolute_module(str(mod))

    # --- Markdown: local link/file reference integrity ---
    # Many generated projects include Markdown docs; broken local links are a common failure mode.
    # We validate conservatively:
    # - Only refs that look file-like (have an extension) or are explicit relative/absolute paths.
    # - Skip refs inside fenced code blocks.
    # - Ignore pure anchors (#...), and remote refs (http/https/mailto/etc).
    def _iter_md_link_refs(text: str) -> Iterable[str]:
        in_fence = False
        for raw in (text or "").splitlines():
            line = raw.rstrip("\n")
            s = line.lstrip()
            if s.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            i = 0
            while i < len(line):
                # Find the start of a markdown link: ]( or ![(
                j = line.find("](", i)
                if j < 0:
                    break
                k = j + 2
                # Parse until the matching ')', allowing balanced parentheses minimally.
                depth = 1
                buf: List[str] = []
                while k < len(line) and depth > 0:
                    ch = line[k]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    buf.append(ch)
                    k += 1
                ref = "".join(buf).strip()
                if ref:
                    yield ref
                i = k + 1

    for rel in sorted([p for p in index if p.lower().endswith(".md")]):
        try:
            txt = (root / rel).read_text(encoding="utf-8")
        except Exception:
            issues.append(f"file_unreadable:{rel}")
            continue

        for ref in _iter_md_link_refs(txt):
            if not isinstance(ref, str):
                continue
            r0 = ref.strip()
            if not r0:
                continue
            # Ignore pure anchors (in-doc).
            if r0.startswith("#"):
                continue
            if _is_remote_ref(r0):
                # Deterministic policy: remote refs are disallowed (offline integrity).
                issues.append(f"remote_resource_not_allowed:{rel}:{r0}")
                continue

            r1 = _strip_query_and_fragment(r0)
            if not r1:
                continue

            looks_file_like = ("." in PurePosixPath(r1).name) or r1.startswith(("./", "../", "/"))
            if not looks_file_like:
                continue

            resolved = _resolve_ref(from_rel=rel, ref=r1, index=index)
            if resolved is None:
                issues.append(f"missing_md_ref:{rel}:{r0}")

    # --- Basic constraint-sensitive rules (kept generic) ---
    c = constraints or {}
    offline_only = bool(c.get("offline_only")) if isinstance(c, dict) else False
    if offline_only:
        # If explicitly offline-only, remote refs are already issues; add suggestion.
        suggestions.append("Remove all remote resource references and bundle assets locally")

    ok = len(issues) == 0
    logger.info(
        "[project_crawl] Validation complete: ok=%s issues=%d warnings=%d suggestions=%d files=%d",
        ok, len(issues), len(warnings), len(suggestions), len(index)
    )
    if issues:
        logger.warning("[project_crawl] Issues found: %s", "; ".join(issues[:10]))
    return {
        "ok": ok,
        "issues": issues,
        "warnings": warnings,
        "suggestions": suggestions,
        "details": {
            "file_count": len(index),
            "html_files": sorted([p for p in index if p.lower().endswith('.html')]),
            "js_files": sorted([p for p in index if p.lower().endswith((".js", ".mjs", ".cjs", ".ts"))]),
            "py_files": py_files,
        },
    }


def validate_project_outputs(
    *,
    outputs_dir: Path,
    constraints: Optional[Dict[str, Any]] = None,
    idea: str = "",
    plan: Optional[Dict[str, Any]] = None,
    file_specs: Optional[Dict[str, Any]] = None,
    validations_dir: Optional[Path] = None,
    label: str = "project",
) -> Dict[str, Any]:
    """Validate outputs (LLM-only) and optionally repair until clean.

    This path is intentionally language/tool neutral:
    - No deterministic parsing/linting by language.
    - No extension-guessing candidates.
    - All validation is done via the LLM (DSPy) using file text + specs + plan.
    """

    c = constraints or {}
    plan_obj: Dict[str, Any] = dict(plan or {})

    # Hidden escape hatch for debugging; default is ON.
    repair_enabled = True
    try:
        if isinstance(c, dict) and c.get("repair_enabled") is False:
            repair_enabled = False
    except Exception:
        repair_enabled = True

    try:
        from crpb.agents.dspy_engine import DspyEngine
        from crpb.repairing.provider import DspyRepairProvider

        engine = DspyEngine()
        provider = DspyRepairProvider(engine=engine)

        try:
            max_pv_chars = int((c or {}).get("project_validate_max_chars") or os.environ.get("CRPB_PROJECT_VALIDATE_MAX_CHARS", "60000"))
        except Exception:
            max_pv_chars = 60000
        try:
            max_pv_files = int((c or {}).get("project_validate_max_files") or os.environ.get("CRPB_PROJECT_VALIDATE_MAX_FILES", "20"))
        except Exception:
            max_pv_files = 20

        def _read_all_files(root: Path) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for p in _iter_files(root):
                try:
                    rel = _norm_rel(str(p.relative_to(root).as_posix()))
                except Exception:
                    continue
                try:
                    out[rel] = p.read_text(encoding="utf-8")
                except Exception:
                    out[rel] = ""
            return out

        def _iter_chunks(files_map: Dict[str, str]) -> List[Dict[str, str]]:
            ordered = sorted([p for p in (files_map or {}).keys() if isinstance(p, str) and p.strip()])
            chunks: List[Dict[str, str]] = []
            cur: Dict[str, str] = {}
            total = 0
            for p in ordered:
                txt = str(files_map.get(p) or "")
                if cur and (len(cur) >= int(max_pv_files) or (total + len(txt)) > int(max_pv_chars)):
                    chunks.append(cur)
                    cur = {}
                    total = 0
                cur[p] = txt
                total += len(txt)
                if len(cur) >= int(max_pv_files) or total >= int(max_pv_chars):
                    chunks.append(cur)
                    cur = {}
                    total = 0
            if cur:
                chunks.append(cur)
            return chunks

        def _jury_profiles() -> List[str]:
            # Distinct validators (not rounds). Allow override via constraints.
            v = None
            try:
                v = (c or {}).get("project_jury_profiles")
            except Exception:
                v = None
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                out = [str(x).strip() for x in v if str(x).strip()]
                return out or ["default"]
            return [
                "default",
                "integration",
                "contracts",
                "completeness",
                "neutrality",
                "contrarian",
            ]

        def _apply_profile_rules(base_constraints: Dict[str, Any], profile: str) -> Dict[str, Any]:
            c2: Dict[str, Any] = dict(base_constraints or {})
            adds = list(c2.get("rule_blocks_add") or [])
            rb = {
                "integration": "project_validate_integration",
                "contracts": "project_validate_contracts",
                "completeness": "project_validate_completeness",
                "neutrality": "project_validate_neutrality",
                "contrarian": "project_validate_contrarian",
            }.get(profile)
            if rb:
                adds = list(dict.fromkeys(adds + [rb]))
            if adds:
                c2["rule_blocks_add"] = adds
            return c2

        def _compute_report(root: Path) -> Dict[str, Any]:
            files_local = _read_all_files(root)
            chunks = _iter_chunks(files_local)
            merged: Dict[str, Any] = {"ok": True, "issues": [], "warnings": [], "suggestions": []}
            jury_details: Dict[str, Any] = {"profiles": [], "summary": ""}
            prior_summary = ""
            key_issues: List[str] = []

            for ch in chunks:
                ch_specs = {p: (file_specs or {}).get(p, {}) for p in (ch or {}).keys()}
                for profile in _jury_profiles():
                    sc = dict((c or {}).get("side_context") or {}) if isinstance(c, dict) else {}
                    if prior_summary:
                        sc["prior_findings"] = prior_summary
                        sc["prior_key_issues"] = key_issues
                    cbase = dict(c or {})
                    cbase["side_context"] = sc
                    c_profile = _apply_profile_rules(cbase, profile)
                    rep = engine.project_validate(
                        idea=str(idea or ""),
                        constraints=c_profile,
                        plan=plan_obj,
                        files=dict(ch or {}),
                        file_specs=dict(ch_specs or {}),
                    )
                    for k in ("issues", "warnings", "suggestions"):
                        vals = rep.get(k) or []
                        if isinstance(vals, list):
                            merged[k].extend([str(x) for x in vals if str(x)])
                    jury_details["profiles"].append(
                        {
                            "profile": profile,
                            "issues": list(rep.get("issues") or []),
                            "warnings": list(rep.get("warnings") or []),
                            "suggestions": list(rep.get("suggestions") or []),
                        }
                    )

                    # Update memory capsule using LLM summarizer (bounded).
                    try:
                        findings_for_summary = (
                            [str(x) for x in (merged.get("issues") or [])]
                            + [str(x) for x in (merged.get("warnings") or [])]
                            + [str(x) for x in (merged.get("suggestions") or [])]
                        )
                        findings_for_summary = [x for x in findings_for_summary if x]
                        if len(findings_for_summary) > 250:
                            findings_for_summary = findings_for_summary[-250:]
                        srep = engine.summarize_findings(
                            findings=findings_for_summary,
                            constraints={"side_context": {"prior_summary": prior_summary}},
                        )
                        prior_summary = str(srep.get("summary") or "")
                        key_issues = list(srep.get("key_issues") or [])
                        jury_details["summary"] = prior_summary
                    except Exception:
                        pass

            # Deduplicate while preserving order.
            for k in ("issues", "warnings", "suggestions"):
                seen: Set[str] = set()
                out: List[str] = []
                for it in merged.get(k) or []:
                    s = str(it)
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
                merged[k] = out

            merged["ok"] = len(list(merged.get("issues") or [])) == 0
            merged["details"] = {"jury": jury_details, "label": str(label or "project")}
            return merged

        report0 = _compute_report(Path(outputs_dir))
        if bool(report0.get("ok", False)) or (not repair_enabled):
            return report0

        # Default lower to reduce token burn; callers can override via constraints or env.
        try:
            max_rounds = int(
                (c or {}).get("repair_max_rounds")
                or os.environ.get("CRPB_REPAIR_MAX_ROUNDS", "3")
            )
        except Exception:
            max_rounds = 3
        try:
            max_files = int(
                (c or {}).get("repair_max_files_per_round")
                or os.environ.get("CRPB_REPAIR_MAX_FILES_PER_ROUND", "8")
            )
        except Exception:
            max_files = 8

        # Import spider lazily to avoid import cycles when the spider itself uses
        # this function as a pure validation oracle.
        from crpb.repairing.spider import repair_outputs_until_ok_with_oracle

        rep = repair_outputs_until_ok_with_oracle(
            outputs_dir=Path(outputs_dir),
            oracle=lambda root: _compute_report(root),
            provider=provider,
            idea=str(idea or ""),
            constraints=dict(c) if isinstance(c, dict) else {},
            file_specs=dict(file_specs or {}),
            max_rounds=max_rounds,
            max_files_per_round=max_files,
            validations_dir=validations_dir,
            label=str(label or "project"),
        )

        report1 = _compute_report(Path(outputs_dir))
        try:
            report1.setdefault("details", {})
            if isinstance(report1.get("details"), dict):
                report1["details"].setdefault(
                    "repair",
                    {
                        "ok": bool(rep.ok),
                        "rounds": int(rep.rounds),
                        "applied_edits": int(rep.applied_edits),
                        "notes": list(rep.notes),
                    },
                )
        except Exception:
            pass
        return report1

    except Exception as e:
        return {
            "ok": False,
            "issues": [f"validator_error:{type(e).__name__}"],
            "warnings": [],
            "suggestions": [],
        }

'''
