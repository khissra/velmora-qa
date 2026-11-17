"""Load the Markdown articles and split front matter from content."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Article:
    id: str        # e.g. "KB-01"
    title: str
    category: str
    updated: str   # e.g. "2026-01-12"
    content: str   # original text, shown to the user


def parse_front_matter(raw):
    """Split '---\\nkey: value\\n---\\nbody' into (metadata dict, body)."""
    parts = raw.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError("Article must start with a --- front matter block")

    meta = {}
    for line in parts[1].strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    return meta, parts[2].strip()


def load_article(path):
    """Read one .md file and return an Article."""
    raw = Path(path).read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    return Article(
        id=meta["id"],
        title=meta["title"],
        category=meta["category"],
        updated=meta["updated"],
        content=body,
    )


def load_articles(kb_dir="kb"):
    """Load every .md file in the folder, sorted by file name."""
    paths = sorted(Path(kb_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    return [load_article(p) for p in paths]
