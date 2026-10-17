# CODE_STYLE.md - llm-finetune

The repository does not enforce a formal code style via configuration files (e.g., .flake8, pyproject.toml). However, the existing Python scripts follow a consistent style that aligns with PEP 8 and common practices.

## Observed Conventions
- **Indentation**: 4 spaces per level (no tabs).
- **Line length**: Typically kept under 100 characters; no strict enforcement but lines are readable.
- **Naming**: 
  - Variables and functions: snake_case.
  - Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_BASE`, `DATASET`, `MAX_SEQ`).
  - Classes: PascalCase (e.g., `SFTConfig`, `SFTTrainer`, `FastLanguageModel`).
- **Imports**: 
  - Standard library imports first (`import argparse`, `import json`, `import sys`).
  - Third-party imports next (`from datasets import Dataset`, `import torch`, `from trl import SFTConfig, SFTTrainer`, `from unsloth import FastLanguageModel`).
  - Each import on its own line; grouped logically.
- **Docstrings**: 
  - Module-level docstrings at the top of each file (triple-quoted, describing purpose and usage).
  - Function docstrings used in some scripts (e.g., `format_row`, `load_dataset`, `main`).
- **Comments**: 
  - Used to explain non-obvious sections (e.g., workarounds for trl SFTTrainer bug).
  - Inline comments aligned with code.
- **Error handling**: 
  - Explicit checks for file existence (`if not DATASET.exists(): raise FileNotFoundError`).
  - Use of `try/except` in dataset generation for robustness.
- **String formatting**: 
  - f-strings for dynamic strings (e.g., `print(f\"Loading base: {args.base} (4-bit)\")`).
  - Double quotes for string literals unless single quotes simplify escaping.
- **Type hints**: 
  - Not used in the current scripts; the codebase does not enforce type annotations.
- **File operations**: 
  - Use of `pathlib.Path` for cross-platform paths.
  - Files opened with explicit UTF-8 encoding where applicable.

## Recommendations
To maintain consistency, new code should:
- Follow the observed 4-space indentation.
- Use snake_case for variables/functions, UPPER_SNAKE_CASE for constants, PascalCase for classes.
- Group imports: standard library, third-party, local (if any).
- Include module-level docstrings describing the script's purpose.
- Add comments for complex logic or workarounds.
- Handle errors gracefully (e.g., check file existence, catch JSON decode errors).
- Use f-strings for string interpolation.
- Keep line lengths reasonable for readability (aim for <100 chars).
- Explicitly specify encoding (`utf-8`) when reading/writing text files.

## Verified against
train.py
export_to_ollama.py
generate_stats_dataset.py
review_dataset.py
probe_sfttrainer.py
smoke_test.py