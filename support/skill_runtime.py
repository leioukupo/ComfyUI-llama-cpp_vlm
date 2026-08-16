import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DIR = PLUGIN_ROOT / "skills"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _fallback_yaml_mapping(text: str) -> Dict[str, Any]:
    """Small YAML subset parser for metadata tests and installs without PyYAML."""
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        list_match = re.match(r"^\s*-\s+(.*)$", raw_line)
        if list_match and current_key:
            result.setdefault(current_key, []).append(_strip_quotes(list_match.group(1)))
            continue

        if ":" not in raw_line:
            current_key = None
            continue

        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            current_key = None
            continue

        if value == "":
            result[key] = []
            current_key = key
        else:
            result[key] = _strip_quotes(value)
            current_key = key
    return result


def _load_yaml_mapping(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {}

    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text) or {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return _fallback_yaml_mapping(text)


def split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        return {}, text

    meta = _load_yaml_mapping("\n".join(lines[1:end_index]))
    body = "\n".join(lines[end_index + 1 :]).lstrip()
    return meta, body


def resolve_skill_root(skill_dir: str) -> Path:
    raw = (skill_dir or "").strip()
    if not raw:
        return DEFAULT_SKILLS_DIR

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PLUGIN_ROOT / path
    return path.resolve()


def parse_selected_skills(selected_skills: str) -> List[str]:
    selected = []
    for part in re.split(r"[\n,;]+", selected_skills or ""):
        name = part.strip()
        if name:
            selected.append(name)
    return selected


def infer_language_from_text(text: str, default: str = "en") -> str:
    if re.search(r"[\u3400-\u9fff]", text or ""):
        return "zh"
    return default


@dataclass
class SkillEntry:
    name: str
    root: Path
    skill_file: Optional[Path] = None
    zh_skill_file: Optional[Path] = None
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    body_preview: str = ""

    @property
    def directory_name(self) -> str:
        return self.root.name

    def display_name(self, language: str = "en") -> str:
        if language == "zh":
            return str(
                self.meta.get("display-name-zh")
                or self.meta.get("title-cn")
                or self.frontmatter.get("title")
                or self.name
            )
        return str(
            self.meta.get("display-name-en")
            or self.meta.get("title-en")
            or self.frontmatter.get("title")
            or self.name
        )

    def description(self, language: str = "en") -> str:
        if language == "zh":
            value = (
                self.meta.get("summary-cn")
                or self.meta.get("desc-cn")
                or self.frontmatter.get("description")
                or self.meta.get("summary-en")
                or self.meta.get("desc-en")
            )
        else:
            value = (
                self.frontmatter.get("description")
                or self.meta.get("summary-en")
                or self.meta.get("desc-en")
                or self.meta.get("summary-cn")
                or self.meta.get("desc-cn")
            )

        if value:
            return str(value).strip()
        return self.body_preview

    def choose_file(self, language: str = "en") -> Path:
        if language == "zh" and self.zh_skill_file:
            return self.zh_skill_file
        if self.skill_file:
            return self.skill_file
        if self.zh_skill_file:
            return self.zh_skill_file
        raise FileNotFoundError(f'Skill "{self.name}" has no SKILL.md file.')

    def summary(self, language: str = "en") -> Dict[str, Any]:
        return {
            "name": self.name,
            "directory": self.directory_name,
            "display_name": self.display_name(language),
            "description": self.description(language),
            "has_zh": self.zh_skill_file is not None,
            "has_references": (self.root / "references").is_dir(),
        }


@dataclass
class SkillLibrary:
    root: Path
    entries: Dict[str, SkillEntry]
    selected_names: List[str] = field(default_factory=list)
    language: str = "auto"
    max_skill_chars: int = 24000
    max_file_chars: int = 20000

    def available_entries(self) -> List[SkillEntry]:
        if not self.selected_names:
            return list(self.entries.values())

        selected = []
        lookup = {name.lower(): entry for name, entry in self.entries.items()}
        by_dir = {entry.directory_name.lower(): entry for entry in self.entries.values()}
        for name in self.selected_names:
            entry = lookup.get(name.lower()) or by_dir.get(name.lower())
            if entry and entry not in selected:
                selected.append(entry)
        return selected

    def available_names(self) -> List[str]:
        return [entry.name for entry in self.available_entries()]

    def effective_language(self, hint_text: str = "") -> str:
        if self.language in {"zh", "en"}:
            return self.language
        return infer_language_from_text(hint_text, default="en")

    def list_summaries(self, language: str = "en") -> List[Dict[str, Any]]:
        return [entry.summary(language) for entry in self.available_entries()]

    def as_system_context(self, hint_text: str = "") -> str:
        language = self.effective_language(hint_text)
        summaries = self.list_summaries(language)
        if not summaries:
            return (
                "Local skill support is enabled, but no skills were found. "
                "You may still answer normally."
            )

        lines = [
            "Local skills are available. Use skill_list to inspect them and skill_read to load the exact instructions or reference files before relying on a skill.",
            "Available local skills:",
        ]
        for item in summaries:
            desc = item.get("description", "")
            if len(desc) > 260:
                desc = desc[:257].rstrip() + "..."
            lines.append(f'- {item["name"]}: {desc}')
        return "\n".join(lines)

    def _entry_for_name(self, skill_name: str) -> SkillEntry:
        wanted = (skill_name or "").strip().lower()
        if not wanted:
            raise ValueError("skill_name is required.")

        available = self.available_entries()
        for entry in available:
            if wanted in {entry.name.lower(), entry.directory_name.lower()}:
                return entry

        names = ", ".join(entry.name for entry in available) or "none"
        raise ValueError(f'Unknown or unselected skill "{skill_name}". Available skills: {names}')

    def _resolve_skill_path(self, entry: SkillEntry, relative_path: str, language: str) -> Path:
        if not relative_path or relative_path in {"SKILL.md", "SKILL.cn.md"}:
            return entry.choose_file(language)

        normalized = relative_path.replace("\\", "/").strip("/")
        candidate = Path(normalized)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Skill file paths must stay inside the selected skill directory.")

        resolved = (entry.root / candidate).resolve()
        root = entry.root.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Skill file paths must stay inside the selected skill directory.")
        if not resolved.is_file():
            raise FileNotFoundError(f'Skill file "{relative_path}" was not found.')
        return resolved

    def read_skill(self, skill_name: str, path: str = "", language: str = "auto", max_chars: Optional[int] = None) -> str:
        effective_language = self.language if language == "auto" else language
        if effective_language == "auto":
            effective_language = "en"

        entry = self._entry_for_name(skill_name)
        target = self._resolve_skill_path(entry, path, effective_language)
        limit = max(1, int(max_chars or (self.max_skill_chars if target.name.startswith("SKILL") else self.max_file_chars)))
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > limit:
            text = text[:limit].rstrip() + f"\n\n[truncated to {limit} characters]"

        return json.dumps(
            {
                "skill": entry.name,
                "path": str(target.relative_to(entry.root)),
                "content": text,
            },
            ensure_ascii=False,
        )

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "skill_list":
            language = arguments.get("language") or self.language
            if language == "auto":
                language = "en"
            query = str(arguments.get("query") or "").strip().lower()
            summaries = self.list_summaries(language)
            if query:
                summaries = [
                    item
                    for item in summaries
                    if query in item["name"].lower()
                    or query in item.get("display_name", "").lower()
                    or query in item.get("description", "").lower()
                ]
            return json.dumps({"skills": summaries}, ensure_ascii=False)

        if name == "skill_read":
            return self.read_skill(
                skill_name=str(arguments.get("skill_name") or arguments.get("name") or ""),
                path=str(arguments.get("path") or ""),
                language=str(arguments.get("language") or "auto"),
                max_chars=arguments.get("max_chars"),
            )

        raise ValueError(f'Unknown skill tool "{name}".')


def skill_tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "skill_list",
                "description": "List local agent skills available in this ComfyUI workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional keyword to filter skills by name or description.",
                        },
                        "language": {
                            "type": "string",
                            "enum": ["auto", "en", "zh"],
                            "description": "Preferred language for skill metadata.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "skill_read",
                "description": "Read a selected local skill's SKILL.md/SKILL.cn.md or a file under its references directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill name or directory name from skill_list.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional relative file path inside the skill directory, such as references/base-en.txt. Empty reads SKILL.md.",
                        },
                        "language": {
                            "type": "string",
                            "enum": ["auto", "en", "zh"],
                            "description": "Preferred SKILL file language when path is empty.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional maximum characters to return.",
                        },
                    },
                    "required": ["skill_name"],
                },
            },
        },
    ]


def _first_body_paragraph(body: str) -> str:
    plain = re.sub(r"^#+\s*", "", body.strip(), flags=re.MULTILINE)
    for paragraph in re.split(r"\n\s*\n", plain):
        value = " ".join(line.strip() for line in paragraph.splitlines()).strip()
        if value:
            return value[:500]
    return ""


def _read_skill_entry(directory: Path) -> Optional[SkillEntry]:
    skill_file = directory / "SKILL.md"
    zh_skill_file = directory / "SKILL.cn.md"
    if not skill_file.is_file() and not zh_skill_file.is_file():
        return None

    source_file = skill_file if skill_file.is_file() else zh_skill_file
    text = source_file.read_text(encoding="utf-8", errors="replace")
    frontmatter, body = split_frontmatter(text)

    meta_file = directory / "meta.yaml"
    meta = {}
    if meta_file.is_file():
        meta = _load_yaml_mapping(meta_file.read_text(encoding="utf-8", errors="replace"))

    name = str(frontmatter.get("name") or meta.get("name") or directory.name).strip()
    return SkillEntry(
        name=name,
        root=directory.resolve(),
        skill_file=skill_file.resolve() if skill_file.is_file() else None,
        zh_skill_file=zh_skill_file.resolve() if zh_skill_file.is_file() else None,
        frontmatter=frontmatter,
        meta=meta,
        body_preview=_first_body_paragraph(body),
    )


def scan_skill_directory(
    skill_dir: str = "",
    selected_skills: str = "",
    language: str = "auto",
    max_skill_chars: int = 24000,
    max_file_chars: int = 20000,
) -> SkillLibrary:
    root = resolve_skill_root(skill_dir)
    entries: Dict[str, SkillEntry] = {}

    if root.is_dir():
        candidates: Iterable[Path] = sorted(path for path in root.iterdir() if path.is_dir())
        for directory in candidates:
            entry = _read_skill_entry(directory)
            if entry is None:
                continue

            key = entry.name
            if key in entries:
                key = entry.directory_name
            if key in entries:
                suffix = 2
                while f"{key}-{suffix}" in entries:
                    suffix += 1
                key = f"{key}-{suffix}"
            entries[key] = entry

    return SkillLibrary(
        root=root,
        entries=entries,
        selected_names=parse_selected_skills(selected_skills),
        language=language if language in {"auto", "en", "zh"} else "auto",
        max_skill_chars=max(1, int(max_skill_chars)),
        max_file_chars=max(1, int(max_file_chars)),
    )
