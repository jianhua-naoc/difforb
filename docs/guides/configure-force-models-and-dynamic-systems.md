# Configure Force Models And Dynamic Systems

This guide shows how to build `DynamicSystem` setups for three common propagation force-model paths:

- the standard major-body system
- the extended system with asteroid perturbers
- a custom system where you choose the force model body by body

For the model behind `ForceModel`, `DynamicSystem`, and the built-in force terms, read [Dynamical Models](../concepts/dynamical-models.md).

## Prerequisites

- Activate the project environment described in [Installation](../installation.md).
- You need local SPK kernels.
- Replace the placeholder paths in the snippets with local files such as `de441.bsp` and `sb441-n16.bsp`.
- The standard major-body system needs a planetary kernel.
- The extended system also needs a small-body kernel.

## 1. Build the standard major-body system

Use `DynamicSystem.from_standard_system()` when you want the built-in major-body model.

```python
from difforb.dynamics import DynamicSystem
from difforb.spk import set_default_ephemeris

planetary_kernel = "/path/to/your/de441.bsp"
set_default_ephemeris(planetary_kernel)

standard = DynamicSystem.from_standard_system()
force_model = standard.build_force_model()

print(force_model)
```

```text title="Output"
<ForceModel shape=() n_forces=1 forces=[PPNGravity] estimated_params=[]>
```

This built-in system builds one force term: `PPNGravity`.

The term includes these bodies:

- Sun
- Mercury barycenter
- Venus barycenter
- Earth
- Moon
- Mars barycenter
- Jupiter barycenter
- Saturn barycenter
- Uranus barycenter
- Neptune barycenter
- Pluto barycenter

This system includes:

- point-mass `PPN` gravity for the built-in major-body background
- no Newtonian asteroid perturbers
- no `J2`
- no non-gravitational terms

## 2. Add asteroid perturbers

Use `DynamicSystem.from_extended_system()` when you also want the supported asteroid perturbers.

```python
from difforb.dynamics import DynamicSystem
from difforb.spk import set_default_ephemeris

planetary_kernel = "/path/to/your/de441.bsp"
asteroid_kernel = "/path/to/your/sb441-n16.bsp"
set_default_ephemeris([planetary_kernel, asteroid_kernel])

extended = DynamicSystem.from_extended_system()
force_model = extended.build_force_model()

print(force_model)
```

```text title="Output"
<ForceModel shape=() n_forces=2 forces=[NewtonianGravity, PPNGravity] estimated_params=[]>
```

This built-in system builds two force terms:

- `PPNGravity` for the major-body background
- `NewtonianGravity` for the built-in asteroid perturbers

The `PPNGravity` term uses the same built-in major-body list as the standard major-body system. The `NewtonianGravity`
term adds these asteroid perturbers:

- Camilla
- Ceres
- Cybele
- Davida
- Eunomia
- Euphrosyne
- Europa
- Hygiea
- Interamnia
- Iris
- Juno
- Pallas
- Psyche
- Sylvia
- Thisbe
- Vesta

This system includes:

- point-mass `PPN` gravity for the built-in major-body background
- Newtonian point-mass gravity for the supported asteroid perturbers
- no `J2`
- no non-gravitational terms

## 3. Build a custom system

Use `DynamicSystem()` directly when you want full control.

The example below:

- uses `PPN` gravity for the Sun, Earth, and Moon
- uses Newtonian gravity for `Ceres`
- adds one comet outgassing term

```python
from difforb.body import EphemerisBody
from difforb.dynamics import DynamicSystem
from difforb.spk import set_default_ephemeris
from difforb.dynamics import CometOutgassingEffect

planetary_kernel = "/path/to/your/de441.bsp"
asteroid_kernel = "/path/to/your/sb441-n16.bsp"
set_default_ephemeris([planetary_kernel, asteroid_kernel])

system = DynamicSystem()

system.add_body(EphemerisBody("sun"), use_ppn=True)
system.add_body(EphemerisBody("earth"), use_ppn=True)
system.add_body(EphemerisBody("moon"), use_ppn=True)
system.add_body(EphemerisBody("ceres"))

system.add_non_grav_force(
    CometOutgassingEffect(
        EphemerisBody("sun"),
        A1=1e-12,
        A2=2e-13,
        A3=0.0,
    )
)

force_model = system.build_force_model()

print(force_model)
print(force_model.get_all_estimated_param_names())
print(force_model.get_all_estimated_params())
```

```text title="Output"
<ForceModel shape=() n_forces=3 forces=[CometOutgassingEffect, NewtonianGravity, PPNGravity] estimated_params=[Outgassing_A1, Outgassing_A2, Outgassing_A3]>
['Outgassing_A1', 'Outgassing_A2', 'Outgassing_A3']
[1.e-12 2.e-13 0.e+00]
```

The output shows that you can:

- choose Newtonian or `PPN` gravity body by body
- add non-gravitational terms to the same `ForceModel`
- see estimated non-gravitational parameters in one combined list

`build_force_model()` groups the terms this way:

- all bodies added with `use_ppn=True` become one `PPNGravity` term
- all bodies added without `use_ppn=True` become one `NewtonianGravity` term
- each non-gravitational effect stays as its own force term

## 4. Choose the right path

Use the standard major-body system when:

- you want the built-in major-body background
- you do not need the built-in asteroid perturbers

Use the extended system when:

- you want the built-in major-body background
- you also want the built-in asteroid perturbers

Use a custom system when:

- you want to choose Newtonian or `PPN` gravity body by body
- you want to add `J2` or non-gravitational terms
- you want full control over the active perturbers

## Common Mistakes

- `DynamicSystem.from_extended_system()` needs both the planetary kernel and the asteroid kernel.
- `EphemerisBody(...)` uses the current default ephemeris unless you pass an explicit `Ephemeris`.
- `use_ppn=True` is chosen per body.
- A non-gravitational term that uses the Sun still needs a valid Sun ephemeris body.

## Next Steps

- Continue to [Propagate A SmallBody And Evaluate Dense Trajectories](propagate-a-smallbody-and-evaluate-dense-trajectories.md).
- Read [Dynamical Models](../concepts/dynamical-models.md) for the model-level meaning of `Newtonian`, `PPN`, `J2`, and the built-in non-gravitational terms.
- Use the [Dynamics API](../api/dynamics.md) and [Integrator API](../api/integrator.md) for details on `DynamicSystem`,
  `ForceModel`, and integrator objects.
