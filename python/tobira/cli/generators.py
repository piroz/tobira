"""tobira.cli.generators - Generate MTA plugin configuration files."""

from __future__ import annotations

import re
from pathlib import Path

# Root directory of the integrations templates.
_INTEGRATIONS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "integrations"
)

# Mapping of MTA name to files to copy and their target config directories.
_MTA_FILES: dict[str, list[tuple[str, str]]] = {
    "rspamd": [
        ("rspamd/tobira.conf", "/etc/rspamd/local.d/tobira.conf"),
        ("rspamd/tobira.lua", "/etc/rspamd/plugins.d/tobira.lua"),
    ],
    "spamassassin": [
        ("spamassassin/tobira.cf", "/etc/spamassassin/tobira.cf"),
        ("spamassassin/Tobira.pm", "/etc/spamassassin/Tobira.pm"),
    ],
    "haraka": [
        ("haraka/config/tobira.ini", "/etc/haraka/config/tobira.ini"),
        ("haraka/plugins/tobira.js", "/etc/haraka/plugins/tobira.js"),
    ],
}

# Default API URL placeholder pattern used in templates.
_DEFAULT_URL_PATTERN = re.compile(r"http://127\.0\.0\.1:8000")


def generate_configs(
    mta: str,
    output_dir: Path,
    api_url: str = "http://127.0.0.1:8000",
) -> list[Path]:
    """Generate MTA plugin configuration files in the output directory.

    Reads template files from the integrations/ directory, replaces
    the default API URL if a custom one is provided, and writes the
    files to the output directory.

    Args:
        mta: MTA name (rspamd, spamassassin, haraka).
        output_dir: Directory to write generated files to.
        api_url: API server URL to use in configuration.

    Returns:
        List of paths to generated files.

    Raises:
        ValueError: If the MTA name is not supported.
        FileNotFoundError: If template files are missing.
    """
    if mta not in _MTA_FILES:
        msg = f"unsupported MTA: {mta}"
        raise ValueError(msg)

    generated: list[Path] = []

    for template_rel, _target_path in _MTA_FILES[mta]:
        template_path = _INTEGRATIONS_DIR / template_rel
        if not template_path.exists():
            msg = f"template not found: {template_path}"
            raise FileNotFoundError(msg)

        content = template_path.read_text(encoding="utf-8")

        # Replace default API URL with custom URL if different.
        if api_url != "http://127.0.0.1:8000":
            content = _DEFAULT_URL_PATTERN.sub(api_url, content)

        out_file = output_dir / template_path.name
        out_file.write_text(content, encoding="utf-8")
        generated.append(out_file)

    return generated


def get_install_instructions(mta: str, output_dir: Path) -> list[str]:
    """Get instructions for installing generated config files.

    Args:
        mta: MTA name.
        output_dir: Directory where files were generated.

    Returns:
        List of instruction strings.
    """
    if mta not in _MTA_FILES:
        return []

    instructions: list[str] = []

    for template_rel, target_path in _MTA_FILES[mta]:
        template_path = _INTEGRATIONS_DIR / template_rel
        src = output_dir / template_path.name
        instructions.append(f"  sudo cp {src} {target_path}")

    # Add reload instructions per MTA.
    reload_cmds: dict[str, str] = {
        "rspamd": "  sudo rspamadm configtest && sudo systemctl reload rspamd",
        "spamassassin": "  sudo systemctl restart spamassassin",
        "haraka": "  sudo systemctl restart haraka",
    }
    if mta in reload_cmds:
        instructions.append(reload_cmds[mta])

    return instructions
