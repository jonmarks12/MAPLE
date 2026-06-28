# -*- coding: utf-8 -*-
"""
FSM: Freezing String Method for double-ended transition-state guess generation.

Thin MAPLE adapter over the upstream ``mlfsm`` package
(https://github.com/thegomeslab/ML-FSM). FSM grows two strings inward from the
reactant and product, optimizing each new frontier node perpendicular to the
local tangent and then freezing it. The highest-energy node at convergence is
returned as the transition-state guess. With ``refine_ts=True`` that guess is
handed to MAPLE's P-RFO to converge a true first-order saddle.

Units
-----
MAPLE calculators return energy in Hartree and forces in Hartree/Angstrom, while
``mlfsm`` is written against ASE-native eV / eV-per-Angstrom. ``_ASEUnitsCalculator``
wraps the active MAPLE calculator so the FSM internals (step sizes, trust radii,
convergence) behave exactly as upstream, and reported energies are converted
back to Hartree for the MAPLE output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .neb import write_xyz
from ...jobABC import JobABC
from maple.function.utility import Molecules
from maple.function.timer import timer


# Hartree (MAPLE native) <-> eV (ASE / mlfsm native).
HARTREE2EV = 27.211386245988
EV2HARTREE = 1.0 / HARTREE2EV


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class FSMParams:
    """Parameters for the Freezing String Method.

    Names mirror the upstream ``fsm_example.py`` so documentation transfers
    directly. All distances are in Angstrom; energies are reported in Hartree.
    """

    interp: str = "ric"          # frontier interpolation: "ric" | "lst" | "cart"
    optcoords: str = "cart"      # node optimization space: "cart" | "ric"
    nnodes_min: int = 18         # nominal node count (sets the step size)
    stepsize: float = 0.0        # explicit Cartesian step (A); > 0 overrides nnodes_min
    ninterp: int = 50            # dense interpolation points between frontier nodes
    method: str = "L-BFGS-B"     # scipy minimizer for node relaxation: "L-BFGS-B" | "CG"
    maxiter: int = 2             # micro-iterations per node relaxation
    maxls: int = 3               # line-search steps per micro-iteration
    dmax: float = 0.05           # max displacement per node step (A)
    refine_ts: bool = True       # refine the TS guess with P-RFO
    write_path: bool = True      # write the full frozen string to <stem>_fsm_path.xyz


# =============================================================================
# Unit-reconciling calculator shim
# =============================================================================
class _ASEUnitsCalculator(Calculator):
    """Present a MAPLE (Hartree, Hartree/A) calculator to ``mlfsm`` as eV / eV/A.

    The wrapper delegates every evaluation to the active MAPLE calculator and
    rescales energy and forces by :data:`HARTREE2EV`. The MAPLE calculators read
    charge and multiplicity from ``atoms.info``; ``mlfsm`` builds new frontier
    nodes that may not carry ``info``, so the endpoint's charge/mult is re-stamped
    on every evaluation to keep charged and open-shell systems correct.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, inner: Calculator, info: Optional[dict] = None) -> None:
        super().__init__()
        self._inner = inner
        self._info = dict(info) if info else {}

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        for key in ("charge", "mult"):
            if key in self._info:
                atoms.info.setdefault(key, self._info[key])
        energy = float(self._inner.get_potential_energy(atoms)) * HARTREE2EV
        forces = np.asarray(self._inner.get_forces(atoms), dtype=float) * HARTREE2EV
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


# =============================================================================
# FSM driver
# =============================================================================
class FSM(JobABC):
    """MAPLE transition-state method wrapping the upstream ``mlfsm`` driver."""

    def __init__(self, output: str, atoms_or_molecules, paras: Optional[dict] = None):
        super().__init__(output)

        if isinstance(atoms_or_molecules, Molecules):
            images = atoms_or_molecules.multiatoms
        elif isinstance(atoms_or_molecules, list):
            images = atoms_or_molecules
        else:
            raise ValueError("FSM requires a Molecules object or a list of Atoms (reactant + product).")

        if len(images) < 2:
            raise ValueError("FSM requires at least two structures (reactant and product).")

        # First and last structures are the endpoints; intermediates are ignored
        # (FSM constructs its own path).
        self.reactant: Atoms = images[0]
        self.product: Atoms = images[-1]

        if self.reactant.constraints or self.product.constraints:
            self.log_info(["[FSM] WARNING: ASE constraints are not honored by mlfsm "
                           "and will be ignored during the string search.\n"])

        self._paras = paras
        self.params = self._init_params(FSMParams, paras, ("fsm", "FSM", "ts"))
        self._log_params()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log_params(self) -> None:
        p = self.params
        self.log_info(
            [
                "\n" + "=" * 70 + "\n",
                "Freezing String Method (FSM) Parameters\n",
                "=" * 70 + "\n",
                f"interp (frontier):   {p.interp}\n",
                f"optcoords (relax):   {p.optcoords}\n",
                f"nnodes_min:          {p.nnodes_min}\n",
                f"stepsize:            {p.stepsize} A {'(active)' if p.stepsize > 0 else '(from nnodes_min)'}\n",
                f"ninterp:             {p.ninterp}\n",
                f"method:              {p.method}\n",
                f"maxiter:             {p.maxiter}\n",
                f"maxls:               {p.maxls}\n",
                f"dmax:                {p.dmax} A\n",
                f"refine_ts (P-RFO):   {p.refine_ts}\n",
                "=" * 70 + "\n",
            ]
        )

    def _log_iteration(self, iteration: int, string) -> None:
        energies = [e for e in (string.r_energy + string.p_energy[::-1]) if e is not None]
        if energies:
            rel = (np.array(energies) - min(energies)) * EV2HARTREE
            emax = float(rel.max())
        else:
            emax = 0.0
        self.log_info(
            [
                f"[FSM] iter {iteration:3d}  nodes {len(string.r_string) + len(string.p_string):3d}  "
                f"gap {string.dist:6.3f} A  max_rel_E {emax:.6f} Eh  ngrad {string.ngrad}\n"
            ]
        )

    # ------------------------------------------------------------------ #
    # Endpoint preparation
    # ------------------------------------------------------------------ #
    def _prepare_endpoint(self, atoms: Atoms) -> Atoms:
        """Copy an endpoint and attach the unit-reconciling calculator.

        ``Atoms.copy`` preserves ``info`` (charge/multiplicity) but drops the
        calculator, so we re-attach the eV/A shim. The shim wraps the live MAPLE
        calculator that the engine already set on the endpoint.
        """
        if atoms.calc is None:
            raise ValueError("FSM endpoint has no calculator attached.")
        prepared = atoms.copy()
        prepared.calc = _ASEUnitsCalculator(atoms.calc, info=atoms.info)
        return prepared

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        with timer("Transition State Search (FSM)"):
            try:
                from mlfsm.cos import FreezingString
                from mlfsm.opt import CartesianOptimizer, InternalsOptimizer
            except ImportError as exc:
                message = (
                    "FSM requires the 'mlfsm' package (and its geomeTRIC/NetworkX deps). "
                    "Install it with \"pip install 'mlfsm==1.0.1'\"."
                )
                self.log_error(message)
                raise ImportError(message) from exc

            p = self.params
            reactant = self._prepare_endpoint(self.reactant)
            product = self._prepare_endpoint(self.product)

            string = FreezingString(
                reactant,
                product,
                nnodes_min=p.nnodes_min,
                interp_method=p.interp,
                ninterp=p.ninterp,
                stepsize=p.stepsize,
            )

            if p.optcoords == "cart":
                optimizer = CartesianOptimizer(reactant.calc, p.method, p.maxiter, p.maxls, p.dmax)
            elif p.optcoords == "ric":
                optimizer = InternalsOptimizer(reactant.calc, p.method, p.maxiter, p.maxls, p.dmax)
            else:
                raise ValueError(f"Unknown FSM optcoords '{p.optcoords}'. Use 'cart' or 'ric'.")

            self.log_info([f"\n[FSM] growing string (dist {string.dist:.3f} A, "
                           f"step {string.stepsize:.3f} A, ~{string.nnodes_min} nodes)\n"])

            iteration = 0
            while string.growing:
                string.grow()
                string.optimize(optimizer)
                iteration += 1
                self._log_iteration(iteration, string)

            # Assemble the full reactant->product path and node energies (eV).
            path: List[Atoms] = list(string.r_string) + list(string.p_string[::-1])
            energies_ev = list(string.r_energy) + list(string.p_energy[::-1])
            energies = [float(e) * EV2HARTREE if e is not None else float("nan") for e in energies_ev]

            ts_idx = int(np.nanargmax(energies))
            ts_atoms = path[ts_idx].copy()

            self._write_outputs(path, energies, ts_idx)
            self._log_summary(energies, ts_idx, string.ngrad)

            if p.refine_ts:
                self._refine_ts(ts_atoms)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def _write_outputs(self, path: List[Atoms], energies: List[float], ts_idx: int) -> None:
        base, _ = os.path.splitext(self.output)

        ts_file = base + "_fsm_ts.xyz"
        write_xyz(ts_file, [path[ts_idx]], energies=[energies[ts_idx]])
        self.log_info([f"\nWrote FSM transition-state guess to: {ts_file}\n"])

        if self.params.write_path:
            path_file = base + "_fsm_path.xyz"
            write_xyz(path_file, path, energies=energies)
            self.log_info([f"Wrote FSM string path to: {path_file}\n"])

    def _log_summary(self, energies: List[float], ts_idx: int, ngrad: int) -> None:
        finite = [e for e in energies if not np.isnan(e)]
        e_min = min(finite) if finite else 0.0
        barrier = (energies[ts_idx] - e_min) if finite else 0.0
        self.log_info(
            [
                "\n" + "=" * 70 + "\n",
                "FSM Summary\n",
                "=" * 70 + "\n",
                f"Nodes:               {len(energies)}\n",
                f"TS guess node:       {ts_idx}\n",
                f"Forward barrier:     {barrier:.6f} Eh ({barrier * 627.5094740631:.2f} kcal/mol)\n",
                f"Gradient calls:      {ngrad}\n",
                "=" * 70 + "\n",
            ]
        )

    # ------------------------------------------------------------------ #
    # Optional P-RFO refinement
    # ------------------------------------------------------------------ #
    def _refine_ts(self, ts_atoms: Atoms) -> None:
        """Refine the FSM TS guess to a true saddle with MAPLE's P-RFO.

        P-RFO runs in MAPLE's native Hartree units and needs ``get_hessian``, so
        it is given the *original* MAPLE calculator rather than the eV/A shim.
        """
        from .PRFO import PRFO

        refined = ts_atoms.copy()
        refined.calc = self.reactant.calc  # native MAPLE calculator (Hartree, has get_hessian)

        # Carry over convergence thresholds the dispatcher set on the endpoints.
        for attr in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
            if hasattr(self.reactant, attr):
                setattr(refined, attr, getattr(self.reactant, attr))

        self.log_info(["\n[FSM] refining TS guess with P-RFO ...\n"])
        prfo = PRFO(output=self.output, atoms=refined, paras=self._paras)
        prfo.run()
