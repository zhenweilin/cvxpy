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

from cvxpy.atoms import promote
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def exp_canon(expr, args, solver_context: SolverInfo | None = None):
    x = promote(args[0], expr.shape)

    # Compute bounds for the auxiliary variable if solver supports bounds
    bounds = None
    if solver_context is not None and solver_context.solver_supports_bounds:
        bounds = _get_expr_bounds(expr)

    t = Variable(expr.shape, bounds=bounds)
    ones = Constant(np.ones(expr.shape))
    constraints = [ExpCone(x, ones, t)]
    return t, constraints


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
