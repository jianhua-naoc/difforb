"""Photometric magnitude models for ephemeris products.

This module defines apparent magnitude models for ephemeris generation. Distances are in ``au`` and phase angles are in radians.
"""

from abc import abstractmethod

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, ArrayLike

from difforb.core.batch import BatchableObject


class MagModel(BatchableObject):
    """Base class for apparent magnitude models."""

    @abstractmethod
    def compute_mag(self, r: Float[Array, "..."], delta: Float[Array, "..."], phase_angle: Float[Array, "..."]) -> \
            Float[Array, "..."]:
        """Return the apparent magnitude.

        Parameters
        ----------
        r : Float[Array, "..."]
            Heliocentric distance in ``au``.
        delta : Float[Array, "..."]
            Observer-to-target distance in ``au``.
        phase_angle : Float[Array, "..."]
            Sun-target-observer phase angle in radians.

        Returns
        -------
        Float[Array, "..."]
            Apparent magnitude.
        """
        pass


class HGModel(MagModel):
    """IAU ``H-G`` asteroid magnitude model."""
    H: Float[Array, "..."]
    G: Float[Array, "..."]

    def __init__(self, H: Float[ArrayLike, "..."], G: Float[ArrayLike, "..."] = 0.15):
        """Initialize an ``H-G`` magnitude model.

        Parameters
        ----------
        H : Float[ArrayLike, "..."]
            Absolute magnitude.
        G : Float[ArrayLike, "..."], default=0.15
            Slope parameter.
        """
        H, G = jnp.asarray(H), jnp.asarray(G)
        self.H, self.G = jnp.broadcast_arrays(H, G)

    def compute_mag(self, r: Float[Array, "..."], delta: Float[Array, "..."], phase_angle: Float[Array, "..."]) -> \
            Float[Array, "..."]:
        """Return the ``H-G`` apparent magnitude.

        Parameters
        ----------
        r : Float[Array, "..."]
            Heliocentric distance in ``au``.
        delta : Float[Array, "..."]
            Observer-to-target distance in ``au``.
        phase_angle : Float[Array, "..."]
            Sun-target-observer phase angle in radians.

        Returns
        -------
        Float[Array, "..."]
            Apparent magnitude.
        """
        ta_half = jnp.tan(phase_angle / 2.0)
        phi_1 = jnp.exp(-3.33 * jnp.power(ta_half, 0.63))
        phi_2 = jnp.exp(-1.87 * jnp.power(ta_half, 1.22))

        phase_term = (1.0 - self.G) * phi_1 + self.G * phi_2
        phase_term = jnp.clip(phase_term, a_min=1e-10)

        mag = self.H + 5.0 * jnp.log10(r * delta) - 2.5 * jnp.log10(phase_term)
        return mag

    @property
    def shape(self):
        """Return the broadcast batch shape of the model parameters.

        Returns
        -------
        tuple[int, ...]
            Batch shape.
        """
        return self.H.shape


class CometTotalMagModel(MagModel):
    """Comet's apparent visual total magnitude model."""
    M1: Float[Array, "..."]
    k1: Float[Array, "..."]

    def __init__(self, M1: Float[Array, "..."], k1: Float[Array, "..."]):
        """Initialize a comet total magnitude model.

        Parameters
        ----------
        M1 : Float[Array, "..."]
            Total absolute magnitude parameter.
        k1 : Float[Array, "..."]
            Heliocentric distance slope.
        """
        M1, k1 = jnp.asarray(M1), jnp.asarray(k1)
        self.M1, self.k1 = jnp.broadcast_arrays(M1, k1)

    def compute_mag(self, r: Float[Array, "..."], delta: Float[Array, "..."], phase_angle: Float[Array, "..."]) -> \
            Float[Array, "..."]:
        """Return the comet total apparent magnitude.

        Parameters
        ----------
        r : Float[Array, "..."]
            Heliocentric distance in ``au``.
        delta : Float[Array, "..."]
            Observer-to-target distance in ``au``.
        phase_angle : Float[Array, "..."]
            Sun-target-observer phase angle in radians. This model does not use it.

        Returns
        -------
        Float[Array, "..."]
            Apparent total magnitude.
        """
        mag = self.M1 + 5.0 * jnp.log10(delta) + self.k1 * jnp.log10(r)
        return mag

    @property
    def shape(self):
        """Return the broadcast batch shape of the model parameters.

        Returns
        -------
        tuple[int, ...]
            Batch shape.
        """
        return self.M1.shape


class CometNuclearMagModel(MagModel):
    """Comet nuclear apparent magnitude model."""
    M2: Float[Array, "..."]
    k2: Float[Array, "..."]
    phcof: Float[Array, "..."]

    def __init__(self, M2: Float[Array, "..."], k2: Float[Array, "..."], phcof: Float[Array, "..."] = 0.04):
        """Initialize a comet nuclear magnitude model.

        Parameters
        ----------
        M2 : Float[Array, "..."]
            Nuclear absolute magnitude parameter.
        k2 : Float[Array, "..."]
            Heliocentric distance slope.
        phcof : Float[Array, "..."], default=0.04
            Linear phase coefficient in magnitudes per degree.
        """
        M2, k2, phcof = jnp.asarray(M2), jnp.asarray(k2), jnp.asarray(phcof)
        self.M2, self.k2, self.phcof = jnp.broadcast_arrays(M2, k2, phcof)

    def compute_mag(self, r: Float[Array, "..."], delta: Float[Array, "..."], phase_angle: Float[Array, "..."]) -> \
            Float[Array, "..."]:
        """Return the comet nuclear apparent magnitude.

        Parameters
        ----------
        r : Float[Array, "..."]
            Heliocentric distance in ``au``.
        delta : Float[Array, "..."]
            Observer-to-target distance in ``au``.
        phase_angle : Float[Array, "..."]
            Sun-target-observer phase angle in radians.

        Returns
        -------
        Float[Array, "..."]
            Apparent nuclear magnitude.

        Notes
        -----
        The phase coefficient ``phcof`` is in magnitudes per degree.
        """
        beta_deg = jnp.rad2deg(phase_angle)
        return self.M2 + 5.0 * jnp.log10(delta) + self.k2 * jnp.log10(r) + self.phcof * beta_deg

    @property
    def shape(self):
        """Return the broadcast batch shape of the model parameters.

        Returns
        -------
        tuple[int, ...]
            Batch shape.
        """
        return self.M2.shape
