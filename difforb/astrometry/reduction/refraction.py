import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, ArrayLike

from difforb.core.batch import BatchableObject


class WeatherParams(BatchableObject):
    """Site weather parameters for atmospheric refraction."""
    temperature: Float[Array, "..."]  # T0: Kelvin
    pressure: Float[Array, "..."]  # P0: mb (hPa)
    humidity: Float[Array, "..."]  # Rh: 0.0 ~ 1.0
    wavelength: Float[Array, "..."]  # lambda: um

    def __init__(self, temperature: Float[ArrayLike, "..."] = 280.15, pressure: Float[ArrayLike, "..."] = 1005,
                 humidity: Float[ArrayLike, "..."] = 0.8, wavelength: Float[ArrayLike, "..."] = 0.574):
        temperature, pressure, humidity, wavelength = jnp.asarray(temperature), jnp.asarray(pressure), jnp.asarray(
            humidity), jnp.asarray(wavelength)
        self.temperature, self.pressure, self.humidity, self.wavelength = jnp.broadcast_arrays(temperature, pressure,
                                                                                               humidity, wavelength)

    @property
    def shape(self):
        return self.temperature.shape


@jax.jit
def auer_standish_refraction_single(zenith_obs: Float[Array, ""], geodetic_lat: Float[Array, ""],
                                    geodetic_alt: Float[Array, ""],
                                    weather: WeatherParams) -> float:
    # === Step 1 & 2: Constants and initial conditions ===
    R_gas, Md, Mw, delta_exp = 8314.36, 28.966, 18.016, 18.36
    re, ht, hs, alpha = 6378120.0, 11000.0, 80000.0, 0.0065
    r0 = re + geodetic_alt
    T0, P0, Rh, wl = weather.temperature, weather.pressure, weather.humidity, weather.wavelength

    # === Step 3: Core coefficients ===
    Pw0 = Rh * (T0 / 247.1) ** delta_exp
    g_bar = 9.784 * (1.0 - 0.0026 * jnp.cos(2 * geodetic_lat) - 0.00000028 * geodetic_alt)
    A = (287.604 + 1.6288 / (wl ** 2) + 0.0136 / (wl ** 4)) * (273.15 / 1013.25) * 1e-6
    C1 = alpha
    C2 = g_bar * Md / R_gas
    gamma = C2 / C1
    C5 = Pw0 * (1.0 - Mw / Md) * gamma / (delta_exp - gamma)
    C6 = A * (P0 + C5) / T0
    C7 = (A * C5 + 11.2684e-6 * Pw0) / T0
    C8 = alpha * (gamma - 1.0) * C6 / T0
    C9 = alpha * (delta_exp - 1.0) * C7 / T0

    # === Troposphere and stratosphere refractive-index functions ===
    def get_tropo(r):
        T = T0 - alpha * (r - r0)
        Tr = T / T0
        n = 1.0 + C6 * Tr ** (gamma - 1.0) - C7 * Tr ** (delta_exp - 1.0)
        dndr = -C8 * Tr ** (gamma - 2.0) + C9 * Tr ** (delta_exp - 2.0)
        return n, dndr

    rt = re + ht
    nt, _, = get_tropo(rt)
    Tt = T0 - alpha * (rt - r0)

    def get_strato(r):
        exp_term = jnp.exp(-C2 * (r - rt) / Tt)
        n = 1.0 + (nt - 1.0) * exp_term
        dndr = -(C2 / Tt) * (nt - 1.0) * exp_term
        return n, dndr

    # === Step 5: Integration limits ===
    n0, _ = get_tropo(r0)

    # Clamp the zenith angle to 89.9 degrees to avoid near-horizon singularities.
    z_safe = jnp.clip(zenith_obs, 0.0, 1.569)
    C_snell = n0 * r0 * jnp.sin(z_safe)

    zt = jnp.arcsin(jnp.clip(C_snell / (nt * rt), -1.0, 1.0))
    rs = re + hs
    ns, _ = get_strato(rs)
    zs = jnp.arcsin(jnp.clip(C_snell / (ns * rs), -1.0, 1.0))

    # === Step 4 & 6: Numerical integration with 65 fixed nodes for JIT compatibility ===
    N_pts = 65
    z_trop = jnp.linspace(z_safe, zt, N_pts)
    z_strat = jnp.linspace(zt, zs, N_pts)

    def integrand(z, is_tropo):
        # Initial guess; the Python branch is static under ``vmap``.
        if is_tropo:
            r = r0 + (rt - r0) * (z - z_safe) / (zt - z_safe + 1e-12)
        else:
            r = rt + (rs - rt) * (z - zt) / (zs - zt + 1e-12)

        sin_z = jnp.sin(z)

        # Solve for radius with four fixed Newton iterations.
        for _ in range(4):
            if is_tropo:
                n, dndr = get_tropo(r)
            else:
                n, dndr = get_strato(r)

            F = n * r - C_snell / sin_z
            F_prime = n + r * dndr
            r = r - F / F_prime

        if is_tropo:
            n, dndr = get_tropo(r)
        else:
            n, dndr = get_strato(r)

        return (r * dndr) / (n + r * dndr)

    f_trop = jax.vmap(lambda z: integrand(z, True))(z_trop)
    f_strat = jax.vmap(lambda z: integrand(z, False))(z_strat)

    # Simpson weights.
    w = jnp.ones(N_pts)
    w = w.at[1:-1:2].set(4.0)
    w = w.at[2:-1:2].set(2.0)

    xi_trop = jnp.sum(w * f_trop) * ((zt - z_safe) / (N_pts - 1)) / 3.0
    xi_strat = jnp.sum(w * f_strat) * ((zs - zt) / (N_pts - 1)) / 3.0

    return xi_trop + xi_strat
