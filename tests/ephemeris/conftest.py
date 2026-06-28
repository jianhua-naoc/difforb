import jax

jax.config.update("jax_enable_x64", True)

import pytest

import difforb.spk as spk
from difforb.body.ephbody import EphemerisBody
from difforb.body.site import Site
from difforb.body.smallbody import SmallBody
from difforb.core.element import KepElement
from difforb.core.time.timescale import Time
from difforb.dynamics.dynamic_system import DynamicSystem
from difforb.ephemeris.generator import EphemerisGenerator
from difforb.integrator.integrator import NumericalIntegrator
from difforb.spk.spk import Ephemeris
from tests.ephemeris.generator_reference import DE441_SPK_PATH, EPOCH_TDB_JD, SB441_SPK_PATH


@pytest.fixture
def ephemeris():
    if not DE441_SPK_PATH.exists() or not SB441_SPK_PATH.exists():
        pytest.skip("local DE441/SB441 SPK excerpts are not installed")
    return Ephemeris([str(DE441_SPK_PATH), str(SB441_SPK_PATH)])


@pytest.fixture
def default_ephemeris(ephemeris):
    spk.set_default_ephemeris(ephemeris)
    yield ephemeris
    spk.clear_default_ephemeris()


@pytest.fixture
def target(default_ephemeris):
    sun = EphemerisBody("sun", eph=default_ephemeris)
    tdb0 = Time.from_tdb_jd(EPOCH_TDB_JD, 0.0).tdb()
    element = KepElement.from_classical(
        tdb=tdb0,
        a=2.15,
        e=0.22,
        inc=10.0,
        node=75.0,
        peri=140.0,
        m=35.0,
    )
    body = SmallBody.create(element, sun=sun)
    force_model = DynamicSystem.from_extended_system(default_ephemeris).build_force_model()
    integrator = NumericalIntegrator(method="IAS15", tol=1.0e-12, initial_step=0.05, max_steps=4096)
    return body.propagate(
        Time.from_tdb_jd(EPOCH_TDB_JD, -5.0).tdb(),
        Time.from_tdb_jd(EPOCH_TDB_JD, 5.0).tdb(),
        force_model,
        integrator,
    )


@pytest.fixture
def generator(target):
    return EphemerisGenerator(target)


@pytest.fixture
def ground_site():
    return Site.from_code("568").require_ground()
