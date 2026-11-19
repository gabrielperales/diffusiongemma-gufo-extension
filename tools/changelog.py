#!/usr/bin/env python3
"""Read release versions from CHANGELOG.md.

  changelog.py latest          print the newest released version
  changelog.py notes VERSION   print that version's section, without heading
  changelog.py check           fail unless extension.json has the newest version

Released sections have the heading `## [X.Y.Z] - YYYY-MM-DD`; `[Unreleased]`
is ignored.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}\s*$")


def sections():
    """Yield (version, body) for each released section, newest first."""
    version, body = None, []
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") or re.match(r"^\[[^\]]+\]: ", line):
            if version:
                yield version, "\n".join(body).strip() + "\n"
            match = HEADING.match(line)
            version, body = (match.group(1) if match else None), []
        elif version:
            body.append(line)
    if version:
        yield version, "\n".join(body).strip() + "\n"


def latest():
    for version, _ in sections():
        return version
    sys.exit("CHANGELOG.md has no released version (## [X.Y.Z] - YYYY-MM-DD)")


def main(argv):
    command = argv[1] if len(argv) > 1 else ""
    if command == "latest" and len(argv) == 2:
        print(latest())
    elif command == "notes" and len(argv) == 3:
        for version, body in sections():
            if version == argv[2]:
                sys.stdout.write(body)
                return
        sys.exit(f"CHANGELOG.md has no section for {argv[2]}")
    elif command == "check" and len(argv) == 2:
        newest = latest()
        manifest = json.loads((ROOT / "extension.json").read_text(encoding="utf-8"))
        if manifest["version"] != newest:
            sys.exit(f"extension.json version {manifest['version']} does not match "
                     f"the newest CHANGELOG.md version {newest}")
        versions = [v for v, _ in sections()]
        if len(versions) != len(set(versions)):
            sys.exit("CHANGELOG.md repeats a version")
        print(f"version {newest}: CHANGELOG.md and extension.json agree")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
