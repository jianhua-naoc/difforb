from typing import List, Tuple, Union

import jax
import numpy as np
from jax.typing import ArrayLike
from jax import Array, numpy as jnp
from jaxtyping import Float
import requests
from pathlib import Path

jax.config.update("jax_enable_x64", True)


def R1_single(theta: ArrayLike) -> Array:
    """
    Rotation tensor around the x-axis.
    :param theta: rotation angle in radians.
    :return: 3*3 rotation matrix or n*3*3 rotation tensor.
    """
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    return jnp.array([[1.0, 0.0, 0.0],
                      [0., costheta, sintheta],
                      [0.0, -sintheta, costheta]])


def R2_single(theta: ArrayLike) -> Array:
    """
    Rotation tensor around the y-axis.
    :param theta: rotation angle in radians.
    :return: 3*3 rotation matrix or n*3*3 rotation tensor.
    """
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    return jnp.array([[costheta, 0., -sintheta],
                      [0., 1., 0.],
                      [sintheta, 0., costheta]])


def R3_single(theta: ArrayLike) -> Array:
    """
    Rotation tensor around the z-axis.
    :param theta: rotation angle in radians.
    :return: 3*3 rotation matrix or n*3*3 rotation tensor.
    """
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    return jnp.array([[costheta, sintheta, 0.],
                      [-sintheta, costheta, 0.],
                      [0., 0., 1.]])


class Radian:
    def __init__(self, rad: ArrayLike):
        self.rad = jnp.asarray(rad)

    @classmethod
    def from_hms(cls, hour, min, sec) -> 'Radian':
        deg = (hour + min / 60. + sec / 3600.) * 15.
        rad = jnp.deg2rad(deg)
        return cls(rad)

    @classmethod
    def from_dms(cls, deg, min, sec, sign=None) -> 'Radian':
        if sign is None:
            sign = jnp.copysign(1., jnp.asarray(deg))
        else:
            sign = jnp.sign(sign)
            sign = jnp.where(sign == 0, 1., sign)
        deg_abs = jnp.abs(deg)
        min_abs = jnp.abs(min)
        sec_abs = jnp.abs(sec)
        total_deg = sign * (deg_abs + min_abs / 60. + sec_abs / 3600.)
        rad = jnp.deg2rad(total_deg)
        return cls(rad)

    @property
    def deg(self):
        return jnp.rad2deg(self.rad)

    @property
    def hms(self) -> Tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, "N"]]:
        hour = self.deg / 15.
        hour_frac, hour_int = jnp.modf(hour)

        min = hour_frac * 60.
        min_frac, min_int = jnp.modf(min)

        sec = min_frac * 60.

        return hour_int, jnp.abs(min_int), jnp.abs(sec)

    @property
    def dms(self) -> Tuple[Float[Array, "N"], Float[Array, "N"], Float[Array, "N"]]:
        deg_frac, deg_int = jnp.modf(self.deg)

        min = deg_frac * 60.
        min_frac, min_int = jnp.modf(min)

        sec = (min_frac * 60.)

        return deg_int, jnp.abs(min_int), jnp.abs(sec)


def car2sph(car: Float[Array, "... 3"]) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Convert cartesian coordinates to spherical coordinates.
    :param car: cartesian coordinates [m*3 matrix].
    :return: spherical coordinates [m row vector, m row vector].
    """
    x, y, z = car[..., 0], car[..., 1], car[..., 2]
    rho = jnp.sqrt(x ** 2 + y ** 2)
    alpha = jnp.arctan2(y, x) % (2 * jnp.pi)  # [0, 2pi]
    delta = jnp.arctan2(z, rho)  # [-pi/2, pi/2]
    return alpha, delta


def sph2car(alpha: Float[Array, "N"], delta: Float[Array, "N"]) -> Float[Array, "N 3"]:
    cos_s2 = jnp.cos(delta)
    x = cos_s2 * jnp.cos(alpha)
    y = cos_s2 * jnp.sin(alpha)
    z = jnp.sin(delta)
    return jnp.stack([x, y, z], axis=-1)


def as_str_array(s: Union[str, List[str], np.ndarray]):
    # Convert string, list of string, np.ndarray to np.ndarray.
    s = np.asarray(s).reshape(-1)
    return s


def as_upper_str_array(s: "Union[str,List[str],np.ndarray]"):
    s = as_str_array(s)
    # Uppercase
    s = np.char.upper(s)
    return s


def str2int(s: str) -> int:
    return int(s.strip())


def str2float(s: str) -> float:
    return float(s.strip())


def download(down_path: str, url: str):
    res = requests.get(url, timeout=60)
    res.raise_for_status()
    path = Path(down_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(res.content)


@jax.jit
def arcsec_to_rad(x: ArrayLike) -> Array:
    return x * (jnp.pi / 648000.0)


def rad_to_arcsec(value):
    return jnp.rad2deg(value) * 3600.0
