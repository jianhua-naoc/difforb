# DiffOrb

![DiffOrb banner](docs/assets/brand/difforb-banner.png)

[![Documentation](https://img.shields.io/badge/docs-Read%20the%20Docs-2f855a)](https://difforb.readthedocs.io/en/latest/)
![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

DiffOrb is a differentiable, batchable Python framework for small-body orbit
propagation, orbit determination, and Horizons-like ephemeris products. It is
built on JAX and keeps the main numerical objects compatible with automatic
differentiation, JIT compilation, and vectorized execution.

**Documentation:** [Read the DiffOrb documentation](https://difforb.readthedocs.io/en/latest/)

## Features

- Native JAX with an object-oriented interface.
- Automatic differentiation support and usage.
- Batch computation and automatic broadcasting.
- Dense orbit propagation.
- Composable orbit-determination and ephemeris-generation workflow.
- Horizons-like ephemeris products.
- Reusable astrometric and astrodynamical infrastructure.

## Installation

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

## Quick Start

Propagate a small body and query its state from the dense trajectory:

```python
from difforb.body import SmallBody
from difforb.core import BCRS, State, Time
from difforb.dynamics import DynamicSystem
from difforb.integrator import NumericalIntegrator
from difforb.spk import set_default_ephemeris

set_default_ephemeris("/path/to/de441.bsp")

t0 = Time.from_tdb_date(2025, 1, 2)
t1 = Time.from_tdb_date(2025, 2, 15)
t_query = Time.from_tdb_date(2025, 1, 20)

state0 = State(
    tdb=t0.tdb(),
    pos=[1.685775738339898, -1.336388854313325, -0.2144927004440800],
    vel=[0.008995712853117517, 0.006985684417802803, 0.004020851173846060],
    frame=BCRS,
)

force_model = DynamicSystem.from_standard_system().build_force_model()
integrator = NumericalIntegrator(method="IAS15", tol=1e-12)

body = SmallBody.create(state0).propagate(t0.tdb(), t1.tdb(), force_model, integrator)
state = body.state(t_query.tdb(), frame=BCRS)

print(state.frame.name)
print(state.pos)
print(state.vel)
```

## License

DiffOrb source code, documentation, and DiffOrb-maintained bundled data files
are licensed under the Apache License, Version 2.0. Data files downloaded from
external sources are still governed by the original providers' terms.
