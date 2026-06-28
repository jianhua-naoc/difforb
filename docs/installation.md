# Installation

DiffOrb requires Python 3.11 or later.

```bash
python -m pip install difforb
```

Download the EOP parameter file, observatory list, and optical debiasing model:

```bash
python -m difforb.data install all
```

Verify the downloaded data:

```bash
python -m difforb.data status
```

## Development

From a source checkout, install DiffOrb in editable mode:

```bash
python -m pip install -e .
```

Install development and test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```
