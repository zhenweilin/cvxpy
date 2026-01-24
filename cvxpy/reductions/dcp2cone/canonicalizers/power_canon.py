"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

import cvxpy as cp
from cvxpy.constraints import PowCone3D
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs, powcone_constrs
from cvxpy.utilities.solver_context import SolverInfo


def _get_expr_bounds(expr):
    """Get bounds from expression, returning None if unbounded or invalid.

    Also returns None if bounds match sign information (avoid redundant constraints).
    """
    try:
        lb, ub = expr.get_bounds()
        # Check if bounds are finite and worth using
        if np.all(np.isinf(lb)) and np.all(np.isinf(ub)):
            return None
        # Check for NaN values which are not valid bounds
        if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
            return None
        # Check if bounds only match sign info (avoid redundant constraints)
        # If lb is all 0 or -inf, and ub is all inf, and expr is nonneg, skip
        lb_trivial = np.all((lb == 0) | np.isinf(lb))
        ub_trivial = np.all(np.isinf(ub))
        if lb_trivial and ub_trivial and expr.is_nonneg():
            return None
        # If ub is all 0 or inf, and lb is all -inf, and expr is nonpos, skip
        ub_trivial_nonpos = np.all((ub == 0) | np.isinf(ub))
        lb_trivial_nonpos = np.all(np.isinf(lb))
        if lb_trivial_nonpos and ub_trivial_nonpos and expr.is_nonpos():
            return None
        return [lb, ub]
    except (NotImplementedError, AttributeError):
        return None


def power_canon(expr, args, solver_context: SolverInfo | None = None):
    # Compute bounds for the auxiliary variable if solver supports bounds
    bounds = None
    if solver_context is not None and solver_context.solver_supports_bounds:
        bounds = _get_expr_bounds(expr)

    # If user requested approximation (default), use SOC
    if expr._approx:
        return power_canon_approx(expr, args, bounds=bounds)

    # User requested power cones (_approx=False)
    # Check if solver supports them
    if solver_context is not None \
            and PowCone3D in solver_context.solver_supported_constraints:
        return power_canon_cone(expr, args, bounds=bounds)

    # Fallback to SOC if pow3d not supported
    # Need to recreate expr with _approx=True for correct rational approx
    expr = cp.power(args[0], expr._p_orig, max_denom=expr.max_denom, _approx=True)
    return power_canon_approx(expr, [args[0]], bounds=bounds)


def power_canon_approx(expr, args, bounds=None):
    x = args[0]
    p = expr.p_rational
    w = expr.w

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []
    else:
        t = Variable(shape, bounds=bounds)
        # TODO(akshayka): gm_constrs requires each of its inputs to be a Variable;
        # is this something that we want to change?
        if 0 < p < 1:
            return t, gm_constrs(t, [x, ones], w)
        elif p > 1:
            return t, gm_constrs(x, [t, ones], w)
        elif p < 0:
            return t, gm_constrs(ones, [x, t], w)
        else:
            raise NotImplementedError('This power is not yet supported.')


def power_canon_cone(expr, args, bounds=None):
    x = args[0]
    p = expr.p_rational

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []

    w = expr.w[0]
    t = Variable(shape, bounds=bounds)

    if 0 < p < 1:
        return t, powcone_constrs(t, [x, ones], w)
    elif p > 1:
        constrs = powcone_constrs(x, [t, ones], w)
        if p % 2 != 0:
            # noneven numerator: add x >= 0 constraint.
            constrs += [x >= 0]
        return t, constrs
    elif p < 0:
        return t, powcone_constrs(ones, [x, t], w)
    else:
        raise NotImplementedError('This power is not yet supported.')

