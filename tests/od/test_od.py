import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from difforb.astrometry.data import (
    ObsMode,
    ObsType,
    ObservationData,
    OpticalObservationData,
    RadarObservationData,
)
from difforb.astrometry.weight import WeightResult
from difforb.core.state.frame import BCRS
from difforb.core.state.state import State
from difforb.core.time.timescale import Time
from difforb.od.dc.result import DCResult, DCEstimate, LSQDiagnostics, OpticalResult, RadarResult
from difforb.od.iod.result import IODResult
from difforb.od.od import DCStrategy, IODStrategy, ODSolver
from tests.assertions import assert_allclose, assert_array_equal


EPOCH_TDB_JD = 2460000.5


class FakeIODSolver:
    def __init__(self, initial_orbit):
        self.initial_orbit = initial_orbit
        self.max_iter = None
        self.calls = []

    def solve(self, obs, arc_days, max_candidates, init_rho):
        self.calls.append(
            {
                "obs": obs,
                "arc_days": arc_days,
                "max_candidates": max_candidates,
                "init_rho": init_rho,
            }
        )
        return IODResult(
            initial_orbit=self.initial_orbit,
            iter_num=2,
            err=1.0e-10,
            used_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        )


class FakeDCSolver:
    def __init__(self):
        self.calls = []

    def solve(
            self,
            data,
            initial_orbit,
            force_model,
            integrator,
            weight_policy,
            debias_policy,
            outlier_policy,
            *,
            photocenter_correction=None,
            event_logger=None,
            **kwargs,
    ):
        stage = None if event_logger is None else event_logger.context.get("stage")
        self.calls.append(
            {
                "data": data,
                "initial_orbit": initial_orbit,
                "force_model": force_model,
                "integrator": integrator,
                "weight_policy": weight_policy,
                "debias_policy": debias_policy,
                "outlier_policy": outlier_policy,
                "photocenter_correction": photocenter_correction,
                "stage": stage,
                "kwargs": kwargs,
            }
        )
        return make_dc_result(initial_orbit, data.num_optical, data.num_radar)


class FakeForceModel:
    def __init__(self):
        self.updated_params = []

    def get_all_estimated_params(self):
        return jnp.asarray([], dtype=float)

    def update_estimated_params(self, params):
        self.updated_params.append(jnp.asarray(params, dtype=float))
        return self


class FakeWeightPolicy:
    def weights(self, obs):
        raise AssertionError("Weight policy should not be used by these epoch strategies.")


class FakeIndexedWeightPolicy:
    def __init__(self, optical_epoch_weights):
        self.optical_epoch_weights = np.asarray(optical_epoch_weights, dtype=float)
        self.calls = []

    def weights(self, obs):
        optical_indices = np.asarray(obs.optical.input_indices, dtype=int)
        self.calls.append(optical_indices.copy())
        epoch_weights = self.optical_epoch_weights[optical_indices]
        optical_uncertainties = np.sqrt(2.0 / epoch_weights)[:, None] * np.ones((len(optical_indices), 2))
        return WeightResult(
            optical_uncertainties=optical_uncertainties,
            radar_uncertainties=np.empty((0,), dtype=float),
            optical_sources=np.asarray(["TEST"] * len(optical_indices), dtype=object),
            radar_sources=np.asarray([], dtype=object),
            optical_correlations=np.zeros(len(optical_indices), dtype=float),
            optical_time_uncertainties=np.full(len(optical_indices), np.nan, dtype=float),
        )


def empty_radar_data():
    return RadarObservationData(
        t=Time.from_tdb_jd(jnp.asarray([], dtype=float), jnp.asarray([], dtype=float)),
        obs_type_ids=np.asarray([], dtype=int),
        obs_mode_ids=np.asarray([], dtype=int),
        values=np.empty((0,), dtype=float),
        uncertainties=np.empty((0,), dtype=float),
        rx_codes=np.asarray([], dtype=object),
        tx_codes=np.asarray([], dtype=object),
        tx_freq=np.empty((0,), dtype=float),
        input_indices=np.asarray([], dtype=int),
    )


def optical_data(offsets):
    offsets = np.asarray(offsets, dtype=float)
    count = len(offsets)
    return OpticalObservationData(
        t=Time.from_tdb_jd(EPOCH_TDB_JD + offsets, jnp.zeros(count, dtype=float)),
        trk_ids=np.asarray([f"T{i}" for i in range(count)], dtype=object),
        obs_type_ids=np.full(count, ObsType.OPTICAL.id, dtype=int),
        obs_mode_ids=np.full(count, ObsMode.CCD.id, dtype=int),
        values=np.deg2rad(np.column_stack([30.0 + offsets, 5.0 + 0.1 * offsets])),
        uncertainties=np.full((count, 2), np.deg2rad(1.0 / 3600.0), dtype=float),
        correlations=np.zeros(count, dtype=float),
        time_uncertainties=np.full(count, np.nan, dtype=float),
        rx_codes=np.asarray(["568"] * count, dtype=str),
        program_codes=np.asarray([""] * count, dtype=object),
        catalog_codes=np.asarray([""] * count, dtype=object),
        note_codes=np.asarray([""] * count, dtype=object),
        magnitudes=np.full(count, np.nan, dtype=float),
        band_codes=np.asarray([""] * count, dtype=object),
        sub_frames=np.asarray(["ICRF"] * count, dtype=object),
        input_indices=np.arange(count, dtype=int),
    )


def observation_data(offsets):
    return ObservationData(
        name="synthetic-od",
        optical=optical_data(offsets),
        radar=empty_radar_data(),
    )


def initial_state(epoch_tdb_jd=EPOCH_TDB_JD):
    return State(
        Time.from_tdb_jd(epoch_tdb_jd, 0.0).tdb(),
        jnp.asarray([1.0, 0.2, -0.1], dtype=jnp.float64),
        jnp.asarray([-0.002, 0.011, 0.001], dtype=jnp.float64),
        BCRS,
    )


def make_dc_result(orbit, n_optical, n_radar):
    n_flat = 2 * n_optical + n_radar
    cov = jnp.eye(6, dtype=jnp.float64)
    optical = OpticalResult(
        residuals=jnp.zeros((n_optical, 2), dtype=jnp.float64),
        normalized_residuals=jnp.zeros((n_optical, 2), dtype=jnp.float64),
        weighted_rms=0.0,
        unweighted_rms=0.0,
        inlier_masks=jnp.ones(n_optical, dtype=bool),
        metrics=jnp.zeros(n_optical, dtype=jnp.float64),
    )
    radar = RadarResult(
        residuals=jnp.zeros(n_radar, dtype=jnp.float64),
        normalized_residuals=jnp.zeros(n_radar, dtype=jnp.float64),
        inlier_masks=jnp.ones(n_radar, dtype=bool),
        metrics=jnp.zeros(n_radar, dtype=jnp.float64),
        delay_weighted_rms=float("nan"),
        delay_unweighted_rms=float("nan"),
        doppler_weighted_rms=float("nan"),
        doppler_unweighted_rms=float("nan"),
    )
    diagnostics = LSQDiagnostics(
        flat_jacobian=jnp.zeros((n_flat, 6), dtype=jnp.float64),
        flat_weights=jnp.ones(n_flat, dtype=jnp.float64),
        optical_weight_matrices=jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (n_optical, 2, 2)),
        radar_weights=jnp.ones(n_radar, dtype=jnp.float64),
        cov_mat_prior=cov,
        cov_rank=jnp.asarray(6, dtype=jnp.int32),
        cov_condition=jnp.asarray(1.0, dtype=jnp.float64),
        cov_valid=jnp.asarray(True),
        converged=True,
        termination_reason="gradient_converged",
        lsq_iterations=1,
        outlier_iterations=0,
    )
    return DCResult(
        estimate=DCEstimate(
            orbit=orbit,
            model_params=jnp.asarray([], dtype=jnp.float64),
            model_param_names=[],
            cov_mat_post=cov,
        ),
        optical=optical,
        radar=radar,
        lsq_diagnostics=diagnostics,
        normalized_residual_rms=0.0,
    )


def solve_od(obs, iod_solver, dc_solver, dc_strategy, weight_policy=None):
    force_model = FakeForceModel()
    result = ODSolver(iod_solver, dc_solver).solve(
        obs,
        force_model,
        integrator=object(),
        weight_policy=FakeWeightPolicy() if weight_policy is None else weight_policy,
        debias_policy=object(),
        outlier_policy=object(),
        iod_strategy=IODStrategy(
            arc_days=4.0,
            max_candidates=5,
            max_iterations=7,
            init_rho=(1.5, 2.5),
        ),
        dc_strategy=dc_strategy,
        log_detail="quiet",
    )
    return result, force_model


def test_od_solver_runs_staged_workflow():
    obs = observation_data([-20.0, -5.0, 0.0, 5.0, 20.0])
    iod_solver = FakeIODSolver(initial_state())
    dc_solver = FakeDCSolver()

    result, force_model = solve_od(
        obs,
        iod_solver,
        dc_solver,
        DCStrategy(
            incremental_arc_days=[12.0, 50.0],
            min_observations=3,
            epoch_strategy="keep_initial",
        ),
    )

    print(
        "[od.workflow.staged] "
        f"stage_statuses={[record.status for record in result.dc_stage_records]} "
        f"dc_counts={[len(call['data']) for call in dc_solver.calls]}"
    )

    assert iod_solver.max_iter == 7
    assert len(iod_solver.calls) == 1
    assert iod_solver.calls[0]["arc_days"] == 4.0
    assert iod_solver.calls[0]["max_candidates"] == 5
    assert iod_solver.calls[0]["init_rho"] == (1.5, 2.5)
    assert [call["stage"] for call in dc_solver.calls] == [1, 2]
    assert [len(call["data"]) for call in dc_solver.calls] == [3, 5]
    assert [record.status for record in result.dc_stage_records] == ["completed", "completed"]
    assert [record.selected_observation_count for record in result.dc_stage_records] == [3, 5]
    assert result.iod_result.initial_orbit is iod_solver.initial_orbit
    assert result.dc_result is not None
    assert len(force_model.updated_params) == 2


def test_od_solver_skips_too_few_observations():
    obs = observation_data([-5.0, 0.0, 5.0])
    iod_solver = FakeIODSolver(initial_state())
    dc_solver = FakeDCSolver()

    result, _ = solve_od(
        obs,
        iod_solver,
        dc_solver,
        DCStrategy(
            incremental_arc_days=[1.0, 20.0],
            min_observations=3,
            epoch_strategy="keep_initial",
        ),
    )

    print(
        "[od.workflow.too_few] "
        f"stage_statuses={[record.status for record in result.dc_stage_records]} "
        f"dc_calls={len(dc_solver.calls)}"
    )

    assert [record.status for record in result.dc_stage_records] == [
        "skipped_too_few_observations",
        "completed",
    ]
    assert [record.selected_observation_count for record in result.dc_stage_records] == [1, 3]
    assert len(dc_solver.calls) == 1
    assert len(dc_solver.calls[0]["data"]) == 3
    assert result.dc_result is not None


def test_od_solver_skips_unchanged_mask():
    obs = observation_data([-2.0, 0.0, 2.0])
    iod_solver = FakeIODSolver(initial_state())
    dc_solver = FakeDCSolver()

    result, _ = solve_od(
        obs,
        iod_solver,
        dc_solver,
        DCStrategy(
            incremental_arc_days=[10.0, 10.0],
            min_observations=3,
            epoch_strategy="keep_initial",
        ),
    )

    print(
        "[od.workflow.unchanged] "
        f"stage_statuses={[record.status for record in result.dc_stage_records]} "
        f"dc_calls={len(dc_solver.calls)}"
    )

    assert [record.status for record in result.dc_stage_records] == [
        "completed",
        "skipped_unchanged_observation_mask",
    ]
    assert [record.selected_observation_count for record in result.dc_stage_records] == [3, 3]
    assert len(dc_solver.calls) == 1
    assert result.dc_result is not None


def test_od_solver_recenters_to_arc_midpoint(monkeypatch):
    import difforb.od.od as od_module

    obs = observation_data([2.0, 4.0, 6.0])
    iod_solver = FakeIODSolver(initial_state())
    dc_solver = FakeDCSolver()
    recenter_calls = []

    def fake_recenter(cur_orbit, stage_epoch, force_model, integrator):
        recenter_calls.append((cur_orbit, stage_epoch, force_model, integrator))
        target_tdb = stage_epoch.tdb()
        shifted_orbit = State(target_tdb, cur_orbit.pos, cur_orbit.vel, cur_orbit.frame)
        return shifted_orbit, abs(float(target_tdb.jd - cur_orbit.tdb.jd))

    monkeypatch.setattr(od_module, "recenter_orbit_to_dc_epoch", fake_recenter)

    result, _ = solve_od(
        obs,
        iod_solver,
        dc_solver,
        DCStrategy(
            incremental_arc_days=[20.0],
            min_observations=3,
            epoch_strategy="arc_midpoint",
        ),
    )

    print(
        "[od.workflow.recenter] "
        f"recenter_epoch={float(dc_solver.calls[0]['initial_orbit'].tdb.jd):.9f} "
        f"stage_statuses={[record.status for record in result.dc_stage_records]}"
    )

    assert len(recenter_calls) == 1
    assert len(dc_solver.calls) == 1
    assert result.dc_stage_records[0].status == "completed"
    assert result.dc_stage_records[0].selected_observation_count == 3
    assert_allclose(dc_solver.calls[0]["initial_orbit"].tdb.jd, EPOCH_TDB_JD + 4.0, atol=0.0, rtol=0.0)
    assert_array_equal(dc_solver.calls[0]["data"].optical.input_indices, np.asarray([0, 1, 2]))


def test_od_solver_recenters_to_weighted_mean_epoch(monkeypatch):
    import difforb.od.od as od_module

    offsets = np.asarray([-20.0, -5.0, 0.0, 5.0, 20.0])
    epoch_weights = np.asarray([1.0, 1.0, 1.0, 1.0, 20.0])
    obs = observation_data(offsets)
    weight_policy = FakeIndexedWeightPolicy(epoch_weights)
    iod_solver = FakeIODSolver(initial_state())
    dc_solver = FakeDCSolver()
    recenter_calls = []

    def fake_recenter(cur_orbit, stage_epoch, force_model, integrator):
        recenter_calls.append((cur_orbit, stage_epoch, force_model, integrator))
        target_tdb = stage_epoch.tdb()
        shifted_orbit = State(target_tdb, cur_orbit.pos, cur_orbit.vel, cur_orbit.frame)
        return shifted_orbit, abs(float(target_tdb.jd - cur_orbit.tdb.jd))

    monkeypatch.setattr(od_module, "recenter_orbit_to_dc_epoch", fake_recenter)

    result, _ = solve_od(
        obs,
        iod_solver,
        dc_solver,
        DCStrategy(
            incremental_arc_days=[100.0],
            min_observations=3,
            epoch_strategy="weighted_mean",
        ),
        weight_policy=weight_policy,
    )

    weighted_offset = np.average(offsets, weights=epoch_weights)
    expected_epoch_jd = np.floor(EPOCH_TDB_JD + weighted_offset) + 0.5
    print(
        "[od.workflow.weighted_epoch] "
        f"weighted_offset={weighted_offset:.9f} "
        f"recenter_epoch={float(dc_solver.calls[0]['initial_orbit'].tdb.jd):.9f} "
        f"stage_statuses={[record.status for record in result.dc_stage_records]}"
    )

    assert len(recenter_calls) == 1
    assert len(dc_solver.calls) == 1
    assert result.dc_stage_records[0].status == "completed"
    assert result.dc_stage_records[0].selected_observation_count == 5
    assert_allclose(dc_solver.calls[0]["initial_orbit"].tdb.jd, expected_epoch_jd, atol=0.0, rtol=0.0)
    assert_array_equal(dc_solver.calls[0]["data"].optical.input_indices, np.asarray([0, 1, 2, 3, 4]))
    assert_array_equal(weight_policy.calls[-1], np.asarray([0, 1, 2, 3, 4]))
