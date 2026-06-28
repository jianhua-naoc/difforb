"""Base force interfaces used by the propagation and estimation stack."""

from abc import abstractmethod
from typing import Any, List, Tuple

from jax import Array
from jaxtyping import Float

from difforb.core.batch import BatchableObject


class Force(BatchableObject):
    """Base class for force terms used by :class:`ForceModel`."""

    @abstractmethod
    def __call__(self, tdb_jd1: Float[Array, ""], tdb_jd2: Float[Array, ""], state: Tuple[Float[Array, "3"], Float[Array,
    "3"]], args: Any) -> Float[Array,
    "3"]:
        """Evaluate the acceleration.

        Parameters
        ----------
        tdb_jd1, tdb_jd2 : Float[Array, ""]
            Split Julian date of the ``TDB`` epoch.
        state : tuple[Float[Array, "3"], Float[Array, "3"]]
            Cartesian state ``(pos, vel)`` in ``BCRS``.
        args : Any
            Extra runtime data passed by the propagator.

        Returns
        -------
        Float[Array, "3"]
            Acceleration in ``au / day^2``.
        """
        pass


class ParametrizedForce(Force):
    """Base class for force terms with estimable parameters used by :class:`ForceModel`."""

    @property
    @abstractmethod
    def n_estimated_params(self) -> int:
        """Number of estimable parameters."""
        pass

    @abstractmethod
    def get_estimated_params(self) -> Float[Array, "N_estimated"]:
        """Return the current estimated parameters."""
        pass

    @abstractmethod
    def get_estimated_param_scales(self) -> Float[Array, "N_estimated"]:
        """
        Return characteristic scales for the estimated parameters.

        The scales are multiplicative parameter increments in the same units as
        :meth:`get_estimated_params`. Solvers can use them to normalize model
        parameters before forming least-squares or trust-region steps. Each
        concrete parametrized force must provide scales that match its
        estimated-parameter ordering.

        Returns
        -------
        Float[Array, "N_estimated"]
            Characteristic scales for each estimated parameter.
        """
        pass

    @abstractmethod
    def update_estimated_params(self, new_params: Float[Array, "N_estimated"]) -> 'ParametrizedForce':
        """Return a new ``ParametrizedForce`` instance with updated estimated parameters."""
        pass

    @abstractmethod
    def get_estimated_param_names(self) -> List[str]:
        """Return the names of the estimated parameters."""
        pass
