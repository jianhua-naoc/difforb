import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import AbstractTerm, AbstractSolver, RESULTS, AbstractStepSizeController, AbstractLocalInterpolation
from diffrax._custom_types import DenseInfo
from jax import Array
from typing import NamedTuple, Any, Callable, Tuple

from jaxtyping import Float, Bool, ArrayLike
from difforb.core.validate import coerce_scalar_int

jax.config.update("jax_enable_x64", True)

# =============================
# 1. Constants
# =============================
EPSILON_DELTA_B = 1e-16


# Gauss-Radau spacings: h1,h2,...,h7
H = jnp.array(
    [
        0.0, 0.0562625605369221464656521910318, 0.180240691736892364987579942780, 0.352624717113169637373907769648,
        0.547153626330555383001448554766, 0.734210177215410531523210605558, 0.885320946839095768090359771030,
        0.977520613561287501891174488626
    ], dtype=jnp.float64
)
# Differences between spacings
RR = jnp.array(
    [0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278, 0.3526247171131696373739078,
     0.2963621565762474909082556, 0.1723840253762772723863278, 0.5471536263305553830014486, 0.4908910657936332365357964,
     0.3669129345936630180138686, 0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584,
     0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621, 0.8853209468390957680903598,
     0.8290583863021736216247076, 0.7050802551022034031027798, 0.5326962297259261307164520, 0.3381673205085403850889112,
     0.1511107696236852365671492, 0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945,
     0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639,
     0.0921996667221917338008147, ], dtype=jnp.float64)

R_INV_MAT = jnp.array([
    [1 / RR[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Substep 1
    [1 / RR[1], 1 / RR[2], 0.0, 0.0, 0.0, 0.0, 0.0],  # Substep 2
    [1 / RR[3], 1 / RR[4], 1 / RR[5], 0.0, 0.0, 0.0, 0.0],  # Substep 3
    [1 / RR[6], 1 / RR[7], 1 / RR[8], 1 / RR[9], 0.0, 0.0, 0.0],  # Substep 4
    [1 / RR[10], 1 / RR[11], 1 / RR[12], 1 / RR[13], 1 / RR[14], 0.0, 0.0],  # Substep 5
    [1 / RR[15], 1 / RR[16], 1 / RR[17], 1 / RR[18], 1 / RR[19], 1 / RR[20], 0.0],  # Substep 6
    [1 / RR[21], 1 / RR[22], 1 / RR[23], 1 / RR[24], 1 / RR[25], 1 / RR[26], 1 / RR[27]]  # Substep 7
], dtype=jnp.float64)

# Coefficients: g->b
C = jnp.array([-0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321,
               -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399,
               0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219,
               -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973,
               -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991,
               0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398,
               -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588],
              dtype=jnp.float64)
B_UPDATE_MATRIX = jnp.array([
    [1.0, C[0], C[1], C[3], C[6], C[10], C[15]],  # Contribution to b0
    [0.0, 1.0, C[2], C[4], C[7], C[11], C[16]],  # Contribution to b1
    [0.0, 0.0, 1.0, C[5], C[8], C[12], C[17]],  # Contribution to b2
    [0.0, 0.0, 0.0, 1.0, C[9], C[13], C[18]],  # Contribution to b3
    [0.0, 0.0, 0.0, 0.0, 1.0, C[14], C[19]],  # Contribution to b4
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, C[20]],  # Contribution to b5
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Contribution to b6
], dtype=jnp.float64).T

# Coefficients: b->g
D = jnp.array(
    [0.0562625605369221464656522, 0.0031654757181708292499905, 0.2365032522738145114532321, 0.0001780977692217433881125,
     0.0457929855060279188954539, 0.5891279693869841488271399, 0.0000100202365223291272096, 0.0084318571535257015445000,
     0.2535340690545692665214616, 1.1362815957175395318285885, 0.0000005637641639318207610, 0.0015297840025004658189490,
     0.0978342365324440053653648, 0.8752546646840910912297246, 1.8704917729329500633517991, 0.0000000317188154017613665,
     0.0002762930909826476593130, 0.0360285539837364596003871, 0.5767330002770787313544596, 2.2485887607691597933926895,
     2.7558127197720458314421588], dtype=jnp.float64)
G_UPDATE_MAT = jnp.array(
    [
        [1.0, D[0], D[1], D[3], D[6], D[10], D[15]],
        [0.0, 1.0, D[2], D[4], D[7], D[11], D[16]],
        [0.0, 0.0, 1.0, D[5], D[8], D[12], D[17]],
        [0.0, 0.0, 0.0, 1.0, D[9], D[13], D[18]],
        [0.0, 0.0, 0.0, 0.0, 1.0, D[14], D[19]],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, D[20]],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=jnp.float64
)

BEZIER_MAT = jnp.array([
    [1., 2., 3., 4., 5., 6., 7.],
    [0., 1., 3., 6., 10., 15., 21.],
    [0., 0., 1., 4., 10., 20., 35.],
    [0., 0., 0., 1., 5., 15., 35.],
    [0., 0., 0., 0., 1., 6., 21.],
    [0., 0., 0., 0., 0., 1., 7.],
    [0., 0., 0., 0., 0., 0., 1.]
], dtype=jnp.float64)


@jax.jit
def add_cs(sum: jnp.ndarray, compensation: jnp.ndarray, num: jnp.ndarray) -> tuple:
    """Kahan Compensated summation"""
    y = num - compensation
    t = sum + y
    t_safe = jax.lax.optimization_barrier(t)
    compensation = (t_safe - sum) - y
    sum = t
    return sum, compensation


class IAS15Term(AbstractTerm):
    """Term for IAS15 2-order ODE solver"""
    acc_fn: Callable[[Float[ArrayLike, ""], Tuple[Float[Array, "3"], Float[Array, "3"]], Any], Float[Array, "3"]]

    def vf(self, t: Float[ArrayLike, ""], state: Tuple[Float[Array, "3"], Float[Array, "3"]], args: Any) -> Tuple[Float[
        Array, "3"], Float[Array, "3"]]:
        _, vel = state
        acc = self.acc_fn(t, state, args)
        return vel, acc

    def contr(self, t0: Float[ArrayLike, ""], t1: Float[ArrayLike, ""], **kwargs) -> Float[ArrayLike, ""]:
        return t1 - t0

    def prod(self, vf: Tuple[Float[ArrayLike, "3"], Float[ArrayLike, "3"]], control: Float[ArrayLike, ""]) -> Tuple[
        jnp.ndarray, jnp.ndarray]:
        vel, acc = vf
        return vel * control, acc * control

    @property
    def is_composite(self) -> bool:
        return False


# ========================================
# 2. Predictor & Corrector Loop
# ========================================
class IAS15PredCorrState(NamedTuple):
    """State of a Predictor-Corrector iteration"""
    b: Float[Array, "7 3"]
    g: Float[Array, "7 3"]
    compensation_b: Float[Array, "7 3"]
    error: Float[Array, ""]
    last_error: Float[Array, ""]
    iter_num: Float[Array, ""]


POS_DENOMS = jnp.array([6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0], dtype=jnp.float64).reshape(7, 1)


@jax.jit
def predict_pos(pos0: Float[Array, "3"], vel0: Float[Array, "3"], acc0: Float[Array, "3"], h: Float[Array, ""],
                dt: Float[Array, ""], b: Float[Array, "7 3"], compensation_pos: Float[Array, "3"]) -> Float[Array, "3"]:
    """
    Predict position according to formula (7)
    """
    coeffs = (b * dt * dt) / POS_DENOMS
    coeffs_rev = coeffs[::-1]
    poly_val, _ = jax.lax.scan(
        lambda carry, c: (carry * h + c, None),
        jnp.zeros_like(pos0),
        coeffs_rev
    )
    poly_val = poly_val * (h * h * h)
    t = h * dt
    base = pos0 + vel0 * t + 0.5 * (t * t) * acc0
    return base + poly_val - compensation_pos


VEL_DENOMS = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=jnp.float64).reshape(7, 1)


@jax.jit
def predict_vel(vel0: Float[Array, "3"], acc0: Float[Array, "3"], h: Float[Array, ""],
                dt: Float[Array, ""], b: Float[Array, "7 3"], compensation_vel: Float[Array, "3"]) -> Float[Array, "3"]:
    """
    Predict velocity according to formula (6)
    """
    coeffs = (b * dt) / VEL_DENOMS
    coeffs_rev = coeffs[::-1]
    poly_val, _ = jax.lax.scan(
        lambda carry, c: (carry * h + c, None),
        jnp.zeros_like(vel0),
        coeffs_rev
    )
    poly_val = poly_val * (h * h)
    t = h * dt
    base = vel0 + acc0 * t
    return base + poly_val - compensation_vel


@jax.jit
def compute_nth_substep_g(acc_nth: Float[Array, "3"], acc0: Float[Array, "3"], prev_g: Float[Array, "7 3"],
                          r_inv_nth: Float[Array, "7"]) -> Float[Array, "3"]:
    """
    Compute g at the n-th substep according to formula (5).
    """
    # 1. Initial value: (acc_nth - acc0) / (h_nth - h0)
    init_val = (acc_nth - acc0) * r_inv_nth[0]

    # 2. new_val = (carry - g_k) / (h_nth - h_{k+1})
    def body_func(carry, input):
        g_k, r_inv = input
        cur_val = carry
        new_val = jnp.where(r_inv == 0.0, cur_val, (cur_val - g_k) * r_inv)
        return new_val, None

    g, _ = jax.lax.scan(body_func, init_val, (prev_g[:6], r_inv_nth[1:]))
    return g


@jax.jit
def ias15_predictor_corrector_step(term: 'IAS15Term', t0: Float[Array, ""], dt: Float[Array, ""],
                                   direction: Float[Array, ""],
                                   pos0: Float[Array, "3"], vel0: Float[Array, "3"], acc0: Float[Array, "3"],
                                   b: Float[Array, "7 3"], compensation_b: Float[Array, "7 3"],
                                   g: Float[Array, "7 3"], compensation_pos: Float[Array, "3"],
                                   compensation_vel: Float[Array, "3"], args: Any) -> Tuple[
    Float[Array, "7 3"], Float[Array, "7 3"], Float[Array, "7 3"], Float[Array, ""]]:
    """One step of predictor-corrector loop"""

    def body_func(carry, input):
        cur_b, cur_compensation_b, cur_g = carry
        h, r_inv_row, b_update_col, n = input

        # 1. Predict position and velocity
        pos = predict_pos(pos0, vel0, acc0, h, dt, cur_b, compensation_pos)
        vel = predict_vel(vel0, acc0, h, dt, cur_b, compensation_vel)

        # 2. Evaluate acceleration
        t = t0 + h * dt
        _, acc = term.vf(direction * t, (pos, vel), args)

        # 3. Correct g
        g_new_n = compute_nth_substep_g(acc, acc0, cur_g, r_inv_row)
        delta_g = g_new_n - cur_g[n]
        new_g = cur_g.at[n].set(g_new_n)

        # 4. Correct b
        # b_update_col: (7,)
        # delta_g: (3,)
        # b_update_col[:, None] * delta_g[:, None]: (7, 1) * (1, 3)->(7, 3)
        delta_b = b_update_col[:, None] * delta_g[None, :]
        new_b, new_compensation_b = add_cs(cur_b, cur_compensation_b, delta_b)

        # 5. Get stats
        is_last = (n == 6)
        max_acc = jnp.max(jnp.abs(acc))
        max_delta_b6 = jnp.max(jnp.abs(delta_g))  # delta_g6 is equal to delta_b6
        ret_max_acc = jnp.where(is_last, max_acc, 0.)
        ret_max_delta_b6 = jnp.where(is_last, max_delta_b6, 0.)

        return (new_b, new_compensation_b, new_g), (ret_max_acc, ret_max_delta_b6)

    input = (H[1:], R_INV_MAT, B_UPDATE_MATRIX, jnp.arange(7))
    init_carry = (b, compensation_b, g)
    (new_b, new_compensation_b, new_g), (ret_max_acc, ret_max_delta_b6) = jax.lax.scan(body_func, init_carry, input)
    max_acc = jnp.max(ret_max_acc)
    max_delta_b6 = jnp.max(ret_max_delta_b6)
    error = max_delta_b6 / max_acc

    return new_b, new_compensation_b, new_g, error


@jax.jit
def ias15_predictor_corrector_loop(term: 'IAS15Term', t0: Float[Array, ""], dt: Float[Array, ""],
                                   direction: Float[Array, ""],
                                   state0: Tuple[Float[Array, "3"], Float[Array, "3"]], acc0: Float[Array, "3"],
                                   init_pred_corr_state: 'IAS15PredCorrState', compensation_pos: Float[Array, "3"],
                                   compensation_vel: Float[Array, "3"],
                                   args: Any) -> 'IAS15PredCorrState':
    pos0, vel0 = state0

    def cond_func(state: 'IAS15PredCorrState'):
        return (
                (state.iter_num < 12)
                & (state.error > EPSILON_DELTA_B)
                & ((state.iter_num <= 2) | (state.error < state.last_error))
        )

    def body_func(state: 'IAS15PredCorrState'):
        new_b, new_compensation_b, new_g, new_error = ias15_predictor_corrector_step(term, t0, dt, direction, pos0,
                                                                                     vel0, acc0,
                                                                                     state.b, state.compensation_b,
                                                                                     state.g, compensation_pos,
                                                                                     compensation_vel,
                                                                                     args)
        return IAS15PredCorrState(b=new_b, compensation_b=new_compensation_b, g=new_g,
                                  error=new_error, last_error=state.error, iter_num=state.iter_num + 1)

    final_state = jax.lax.while_loop(cond_func, body_func, init_pred_corr_state)
    return final_state


# ========================================
# 3. One step solver
# ========================================
@jax.jit
def guess_next_b_and_e(prev_b: Float[Array, "7 3"], prev_e: Float[Array, "7 3"], ratio: Float[Array, ""]) -> Tuple[
    Float[Array, "7 3"], Float[Array, "7 3"]]:
    """
    Guess the b and e for the current step according to the b and e from the previous step.
    Ref: predict_next_step of the rebound
    """
    is_reset = ratio > 20.0
    q_series = jnp.power(ratio, jnp.arange(1., 8.))

    # 1. Guess e
    # (7, 1) * (7, 3) -> (7, 3)
    new_e = q_series[:, None] * jnp.matmul(BEZIER_MAT, prev_b)

    # 2. Guess b
    new_b = new_e + (prev_b - prev_e)

    # 3. If the step size changes too much, reset the prediction
    zeros = jnp.zeros_like(prev_b)
    new_e = jnp.where(is_reset, zeros, new_e)
    new_b = jnp.where(is_reset, zeros, new_b)

    return new_b, new_e


@jax.jit
def integrate_final_pos(pos0: Float[Array, "3"], vel0: Float[Array, "3"], acc0: Float[Array, "3"],
                        b: Float[Array, "7 3"], compensation_pos: Float[Array, "3"], dt: Float[Array, ""]) -> Tuple[
    Float[Array, "3"], Float[Array, "3"]]:
    """Use the converged b to integrate and get the final position after one step"""

    dt2 = dt * dt

    delta_pos_b = (b * dt2 / POS_DENOMS)[::-1]  # (7, 3), delta from b coefficients
    delta_pos_a = (acc0 / 2. * dt2)[None, :]  # (1, 3), delta from acceleration
    delta_pos_v = (vel0 * dt)[None, :]  # (1, 3), delta from velocity
    delta_pos = jnp.concatenate([delta_pos_b, delta_pos_a, delta_pos_v], axis=0)  # (9,3)

    def body_func(carry, delta):
        pos, compensation = carry
        new_pos, new_compensation = add_cs(pos, compensation, delta)
        return (new_pos, new_compensation), None

    (final_pos, final_compensation), _ = jax.lax.scan(body_func, (pos0, compensation_pos), delta_pos)

    return final_pos, final_compensation


@jax.jit
def integrate_final_vel(vel0: Float[Array, "3"], acc0: Float[Array, "3"],
                        b: Float[Array, "7 3"],
                        compensation_vel: Float[Array, "3"], dt: Float[Array, ""]) -> Tuple[
    Float[Array, "3"], Float[Array, "3"]]:
    """Use the converged b to integrate and get the final velocity after one step"""
    delta_vel_b = (b * dt / VEL_DENOMS)[::-1]  # (7, 3)
    delta_vel_a = (acc0 * dt)[None, :]  # (1, 3)
    delta_vel = jnp.concatenate([delta_vel_b, delta_vel_a], axis=0)  # (8, 3)

    def body_func(carry, delta):
        vel, compensation = carry
        new_vel, new_compensation = add_cs(vel, compensation, delta)
        return (new_vel, new_compensation), None

    (final_vel, final_compensation), _ = jax.lax.scan(body_func, (vel0, compensation_vel), delta_vel)

    return final_vel, final_compensation


class IAS15Interpolation(AbstractLocalInterpolation):
    """15-order interpolator based on Gauss-Radu nodes"""
    t0: Float[Array, ""]
    t1: Float[Array, ""]
    dt: Float[Array, ""]
    y0: Tuple[Float[Array, "3"], Float[Array, "3"]]
    y1: Tuple[Float[Array, "3"], Float[Array, "3"]]
    acc0: Float[Array, "3"]
    b: Float[Array, "7 3"]

    def evaluate(
            self, t: Float[Array, ""], t1: Float[Array, ""] | None = None, left: bool = True
    ) -> Tuple[Float[Array, "3"], Float[Array, "3"]]:
        direction = jnp.sign(self.dt)
        t = t * direction
        t0 = self.t0 * direction
        h = (t - t0) / self.dt
        pos0, vel0 = self.y0
        compensation_pos = jnp.zeros_like(pos0)
        compensation_vel = jnp.zeros_like(pos0)
        pos_t = predict_pos(pos0, vel0, self.acc0, h, self.dt, self.b, compensation_pos)
        vel_t = predict_vel(vel0, self.acc0, h, self.dt, self.b, compensation_vel)
        return pos_t, vel_t


class IAS15SolverState(NamedTuple):
    """State of a single step"""
    prev_b: Float[Array, "7 3"]
    prev_e: Float[Array, "7 3"]  # extrapolated value of b
    prev_acc1: Float[Array, "3"]  # acceleration at the end of the previous step
    prev_compensation_pos: Float[Array, "3"]
    prev_compensation_vel: Float[Array, "3"]
    prev_dt: Float[Array, ""]


class IAS15Solver(AbstractSolver):
    term_structure = AbstractTerm

    def init(
            self,
            term: 'AbstractTerm',
            t0: Float[Array, ""],
            t1: Float[Array, ""],
            y0: Tuple[Float[Array, "3"], Float[Array, "3"]],
            args: Any,
    ) -> 'IAS15SolverState':
        pos0, vel0 = y0
        _, acc0 = term.vf(t0, y0, args)
        return IAS15SolverState(
            prev_b=jnp.zeros((7, 3)),
            prev_e=jnp.zeros((7, 3)),
            prev_acc1=acc0,
            prev_compensation_pos=jnp.zeros_like(pos0),
            prev_compensation_vel=jnp.zeros_like(vel0),
            prev_dt=jnp.array(0., dtype=jnp.float64),
        )

    def order(self, terms):
        return 15

    def error_order(self, terms):
        return 7

    def step(
            self,
            term: 'AbstractTerm',
            _t0: Float[Array, ""],
            _t1: Float[Array, ""],
            y0: Tuple[Float[Array, "3"], Float[Array, "3"]],
            args: Any,
            solver_state: 'IAS15SolverState',
            made_jump: bool,
    ) -> tuple[
        Tuple[Float[Array, "3"], Float[Array, "3"]], dict[str, Float[Array, ""]], DenseInfo, 'IAS15SolverState', str]:
        pos0, vel0 = y0
        acc0 = solver_state.prev_acc1
        dt = term.contr(_t0, _t1)
        direction = jnp.where(dt >= 0, 1., -1.)
        t0 = direction * _t0
        t1 = direction * _t1
        # 1. Guess b, e and g
        # If ratio > 20, predict_next_b_and_e will return zeros. For the first step (last dt=0), this reset logic will set b and e to zeros.
        # ratio = jnp.where(solver_state.prev_dt == 0., 30., dt / solver_state.prev_dt)
        safe_prev_dt = jnp.where(solver_state.prev_dt == 0., 1.0, solver_state.prev_dt)
        ratio = jnp.where(solver_state.prev_dt == 0., 30., dt / safe_prev_dt)
        guess_b, guess_e = guess_next_b_and_e(solver_state.prev_b, solver_state.prev_e, ratio)
        guess_g = jnp.matmul(G_UPDATE_MAT, guess_b)

        # 2. Predictor-Corrector Loop
        init_pred_corr_state = IAS15PredCorrState(
            b=guess_b,
            g=guess_g,
            compensation_b=jnp.zeros_like(guess_b),
            error=jnp.array(1e50),
            last_error=jnp.array(1e50),
            iter_num=jnp.array(0, dtype=jnp.int32),
        )
        final_pred_corr_state = ias15_predictor_corrector_loop(term, t0, dt, direction, y0, acc0,
                                                               init_pred_corr_state, solver_state.prev_compensation_pos,
                                                               solver_state.prev_compensation_vel, args)
        b, pred_corr_error = final_pred_corr_state.b, final_pred_corr_state.error

        # 3. Integrate to get the position and velocity at t1
        pos1, compensation_pos1 = integrate_final_pos(pos0, vel0, acc0,
                                                      b,
                                                      solver_state.prev_compensation_pos, dt)
        vel1, compensation_vel1 = integrate_final_vel(vel0, acc0,
                                                      b,
                                                      solver_state.prev_compensation_vel, dt)

        # 4. Compute the acceleration at t1
        y1 = (pos1, vel1)
        _, acc1 = term.vf(direction * t1, y1, args)

        # 5. Compute the value for step size controller
        # acc1_poly represents the acceleration at t1 obtained through a polynomial, acc1 represents the acceleration at t1 derived from the dynamic model. In scenarios where convergence is good, the two values are remarkably close. However, during step size control, the polynomial is evaluated, hence acc1_poly should be employed. This ensures consistency in the derivation source, leading to a smoother error estimation plane.
        acc1_poly = solver_state.prev_acc1 + jnp.sum(b, axis=0)

        # 5.1 RS2015
        max_b6 = jnp.max(jnp.abs(b[6]))
        max_acc = jnp.max(jnp.abs(acc1_poly))
        step_error = jax.lax.stop_gradient(max_b6 / max_acc)

        # 5.2 PRS23
        # y2^2: |a|^2
        y2_2 = jnp.sum(acc1_poly * acc1_poly)
        # y3^2: |Jerk * dt|^2
        k_vec = jnp.arange(1, 8, dtype=jnp.float64)
        jerk_dt_vec = jnp.sum(b * k_vec[:, None], axis=0)
        y3_2 = jnp.sum(jerk_dt_vec * jerk_dt_vec)
        # y4^2: |Snap * dt^2|^2
        k_k_minus_1_vec = k_vec * (k_vec - 1.0)
        snap_dt2_vec = jnp.sum(b * k_k_minus_1_vec[:, None], axis=0)
        y4_2 = jnp.sum(snap_dt2_vec * snap_dt2_vec)

        # 5.3 Is in linear motion
        v2 = jnp.sum(vel0 * vel0)
        x2 = jnp.sum(pos0 * pos0)
        straight_line_metric = jax.lax.stop_gradient((v2 * dt * dt) / x2)

        error_info = {
            "RS2015_step_error": step_error,
            "y2_2": y2_2,
            "y3_2": y3_2,
            "y4_2": y4_2,
            "straight_line_metric": straight_line_metric
        }

        # 6. Update solver state
        new_solver_state = IAS15SolverState(
            prev_b=b,
            prev_e=guess_e,
            prev_acc1=acc1,
            prev_compensation_pos=compensation_pos1,
            prev_compensation_vel=compensation_vel1,
            prev_dt=dt
        )

        dense_info = dict(dt=dt, y0=y0, y1=y1, b=b, acc0=acc0)

        return y1, error_info, dense_info, new_solver_state, RESULTS.successful

    def interpolation_cls(self, *, t0, t1, dt, y0, y1, b, acc0):
        return IAS15Interpolation(
            dt=dt,
            t0=t0,
            t1=t1,
            y0=y0,
            y1=y1,
            acc0=acc0,
            b=b
        )

    def func(
            self,
            term: 'AbstractTerm',
            t: Float[ArrayLike, ""],
            y: Tuple[Float[Array, "3"], Float[Array, "3"]],
            args: Any,
    ):
        return term.vf(t, y, args)


class IAS15StepSizeController(AbstractStepSizeController):
    rtol: float = 0.  # not used
    atol: float = 1e-12  # epsilon_b
    adaptive_mode: int = eqx.field(static=True, default=2)
    safety_factor: float = 0.25
    min_dt: float = 0.

    def __init__(
            self,
            rtol: float = 0.0,
            atol: float = 1e-12,
            adaptive_mode: int = 2,
            safety_factor: float = 0.25,
            min_dt: float = 0.0,
    ):
        self.rtol = rtol
        self.atol = atol
        self.adaptive_mode = coerce_scalar_int("adaptive_mode", adaptive_mode)
        self.safety_factor = safety_factor
        self.min_dt = min_dt

    def wrap(self, direction) -> 'IAS15StepSizeController':
        return self

    def init(self, terms, t0: Float[Array, ""], t1: Float[Array, ""],
             y0: Tuple[Float[Array, "3"], Float[Array, "3"]], dt0: Float[Array, ""], args: Any, func,
             error_order: Float[Array, ""]):
        tnext = t0 + dt0
        return tnext, None

    def adapt_step_size(
            self,
            t0: Float[Array, ""],
            t1: Float[Array, ""],
            y0: Tuple[Float[Array, "3"], Float[Array, "3"]],
            y1: Tuple[Float[Array, "3"], Float[Array, "3"]],
            args: Any,
            y_error: dict,
            error_order: Float[ArrayLike, ""],
            controller_state,
    ) -> tuple[Bool[Array, ""], Float[ArrayLike, ""], Float[ArrayLike, ""], Bool[Array, ""], Any, RESULTS]:
        y_error = jax.tree_util.tree_map(jax.lax.stop_gradient, y_error)
        dt = jax.lax.stop_gradient(t1 - t0)
        dtype = dt.dtype
        tiny = jnp.finfo(dtype).tiny
        fallback_ratio = 1.0 / self.safety_factor
        # 1. Compute the ratio
        if self.adaptive_mode == 1:
            # === Mode 1: Rein & Spiegel (2015) Formula (11) ===
            step_error = y_error['RS2015_step_error']
            raw_ratio_candidate = jnp.power(self.atol / step_error, 1. / 7.)
            raw_ratio_candidate = jnp.nan_to_num(raw_ratio_candidate, nan=fallback_ratio, posinf=fallback_ratio, neginf=0.0)
            valid = jnp.isfinite(step_error) & (step_error > tiny)
            raw_ratio = jnp.where(valid, raw_ratio_candidate, fallback_ratio)
        else:
            # === Mode 2: PRS23 (2024) Formula (16)-(17) ===
            y2_2 = y_error['y2_2']
            y3_2 = y_error['y3_2']
            y4_2 = y_error['y4_2']
            # timescale_ratio = 2. * y2_2 / (y3_2 + jnp.sqrt(y2_2 * y4_2))
            # tau_over_dt = jnp.sqrt(timescale_ratio)
            timescale_ratio = 2.0 * y2_2 / (y3_2 + jnp.sqrt(y2_2) * jnp.sqrt(y4_2))
            raw_ratio_candidate = jnp.power(5040.0 * self.atol, 1.0 / 7.0) * jnp.sqrt(timescale_ratio)
            valid = jnp.isfinite(timescale_ratio) & (timescale_ratio > tiny)
            raw_ratio = jnp.where(valid, raw_ratio_candidate, fallback_ratio)

        straight_line_metric = y_error["straight_line_metric"]
        is_free_fall = (
                jnp.isfinite(straight_line_metric)
                & (straight_line_metric < 1e-16)
        )
        raw_ratio = jnp.where(is_free_fall, fallback_ratio, raw_ratio)
        raw_ratio = jax.lax.stop_gradient(raw_ratio)

        # 2. Accept / Reject
        keep_step = raw_ratio >= self.safety_factor

        # 3. Limit too large step
        upper_ratio = 1.0 / self.safety_factor
        actual_ratio = jnp.minimum(raw_ratio, upper_ratio)

        # 4. Compute the next dt
        proposed_dt = dt * actual_ratio

        # 5. Limit too small step
        next_dt = jnp.maximum(self.min_dt, proposed_dt)

        next_t0 = jnp.where(keep_step, t1, t0)
        next_t1 = next_t0 + next_dt

        made_jump = jnp.array(False)
        # jax.debug.print("raw_ratio {i}", i=raw_ratio)

        return keep_step, next_t0, next_t1, made_jump, controller_state, RESULTS.successful
