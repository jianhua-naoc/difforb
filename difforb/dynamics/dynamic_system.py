"""Dynamic-system builders for orbit propagation.

This module groups perturbing bodies and non-gravitational forces, then assembles them into a :class:`difforb.dynamics.force_model.ForceModel`. It provides two preset systems: a standard planetary system and an extended system with major asteroids.
"""

from typing import TYPE_CHECKING
from difforb.dynamics.force_model import ForceModel, NewtonianGravity, PPNGravity
from difforb.body.ephbody import EphemerisBody

if TYPE_CHECKING:
    from difforb.spk.spk import Ephemeris
    from difforb.dynamics.force_model import Force


class DynamicSystem:
    """Collection of perturbing bodies and extra forces used to build a force model."""

    def __init__(self):
        """Initialize an empty dynamic system."""
        self._newton_bodies = []
        self._ppn_bodies = []
        self._non_grav_forces = []

    @classmethod
    def from_standard_system(cls, eph: 'Ephemeris' = None):
        """Build a dynamic system from the standard planetary set.

        Parameters
        ----------
        eph : Ephemeris, optional
            Ephemeris source used to build the perturbing bodies.

        Returns
        -------
        DynamicSystem
            System with the Sun, major planets, and the Moon. All listed bodies use the ``PPN`` gravity model.
        """
        system = cls()
        body_names = ['sun', 'mercury barycenter', 'venus barycenter', 'earth', 'moon', 'mars barycenter',
                      'jupiter barycenter', 'saturn barycenter', 'uranus barycenter', 'neptune barycenter',
                      'pluto barycenter']
        for body_name in body_names:
            body = EphemerisBody(body_name, eph=eph)
            system.add_body(body, use_ppn=True)
        return system

    @classmethod
    def from_extended_system(cls, eph: 'Ephemeris' = None):
        """Build a dynamic system from the extended body set.

        Parameters
        ----------
        eph : Ephemeris, optional
            Ephemeris source used to build the perturbing bodies.

        Returns
        -------
        DynamicSystem
            Standard system plus the supported large asteroids. The extra asteroids use the Newtonian gravity model.
        """
        system = cls.from_standard_system(eph)
        ast_names = ['camilla', 'ceres', 'cybele', 'davida', 'eunomia', 'euphrosyne', 'europa',
                     'hygiea', 'interamnia', 'iris', 'juno', 'pallas', 'psyche', 'sylvia', 'thisbe',
                     'vesta']
        for ast_name in ast_names:
            body = EphemerisBody(ast_name, eph=eph)
            system.add_body(body)
        return system

    def add_body(self, body: EphemerisBody, use_ppn: bool = False):
        """Add one perturbing body.

        Parameters
        ----------
        body : EphemerisBody
            Perturbing body.
        use_ppn : bool, default=False
            If ``True``, add the body to the ``PPN`` gravity list. Otherwise, add it to the Newtonian gravity list.
        """
        if use_ppn:
            self._ppn_bodies.append(body)
        else:
            self._newton_bodies.append(body)

    def add_non_grav_force(self, force: 'Force'):
        """Add one non-gravitational force model.

        Parameters
        ----------
        force : Force
            Force model to append.
        """
        self._non_grav_forces.append(force)

    def build_force_model(self) -> ForceModel:
        """Build the combined force model.

        Returns
        -------
        ForceModel
            Force model assembled from the stored non-gravitational forces, Newtonian gravity bodies, and ``PPN`` gravity bodies.
        """
        forces = []
        forces.extend(self._non_grav_forces)
        if self._newton_bodies:
            forces.append(NewtonianGravity(self._newton_bodies))
        if self._ppn_bodies:
            forces.append(PPNGravity(self._ppn_bodies))
        return ForceModel(forces)
