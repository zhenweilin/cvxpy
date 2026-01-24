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

from fractions import Fraction

import numpy as np

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
from cvxpy.utilities.power_tools import gm_constrs
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


def pnorm_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    p = expr.p
    axis = expr.axis
    shape = expr.shape

    # Compute bounds for the auxiliary variable if solver supports bounds
    bounds = None
    if solver_context is not None and solver_context.solver_supports_bounds:
        bounds = _get_expr_bounds(expr)

    t = Variable(shape, bounds=bounds)

    if p == 2:
        if axis is None:
            assert shape == tuple()
            return t, [SOC(t, vec(x, order='F'))]
        else:
            return t, [SOC(vec(t, order='F'), x, axis)]

    # we need an absolute value constraint for the symmetric convex branches
    # (p > 1)
    constraints = []
    if p > 1:
        # TODO(akshayka): Express this more naturally (recursively), in terms
        # of the other atoms
        abs_expr = abs(x)
        abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
        x = abs_x
        constraints += abs_constraints

    # now, we take care of the remaining convex and concave branches
    # to create the rational powers, we need a new variable, r, and
    # the constraint sum(r) == t
    r = Variable(x.shape)
    constraints += [sum(r) == t]

    # todo: no need to run gm_constr to form the tree each time.
    # we only need to form the tree once
    promoted_t = Constant(np.ones(x.shape)) * t
    p = Fraction(p)
    if p < 0:
        constraints += gm_constrs(promoted_t, [x, r],  (-p/(1-p), 1/(1-p)))
    if 0 < p < 1:
        constraints += gm_constrs(r,  [x, promoted_t], (p, 1-p))
    if p > 1:
        constraints += gm_constrs(x,  [r, promoted_t], (1/p, 1-1/p))

    return t, constraints
