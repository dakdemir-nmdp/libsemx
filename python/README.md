# semx: Python Front-End for libsemx

Python package providing a user-friendly interface to the libsemx unified SEM/mixed-model engine.

## Installation

This package is part of the libsemx project. Install in development mode:

```bash
cd python
uv pip install -e .
```

For testing:

```bash
uv pip install -e ".[test]"
```

## Usage

```python
import semx

# Example usage (TBD)
```

## Testing

Run tests with:

```bash
uv run pytest
```

## Development

- Follow the main project conventions
- Tests mirror C++ fixtures for consistency
- Use `uv` for all Python operations