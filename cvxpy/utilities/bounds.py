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
from typing import List, Optional, Tuple, Union

import numpy as np

# Type alias for bounds: (lower_bound, upper_bound)
Bounds = Tuple[np.ndarray, np.ndarray]


def unbounded(shape: Tuple[int, ...]) -> Bounds:
    """Return unbounded interval (-inf, inf) for given shape.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the bounds arrays.

    Returns
    -------
    Bounds
        A tuple of (lower, upper) arrays filled with -inf and inf.
    """
    lower = np.full(shape, -np.inf)
    upper = np.full(shape, np.inf)
    return (lower, upper)


def scalar_bounds(lb: float, ub: float) -> Bounds:
    """Return bounds for a scalar.

    Parameters
    ----------
    lb : float
        Lower bound.
    ub : float
        Upper bound.

    Returns
    -------
    Bounds
        A tuple of scalar arrays.
    """
    return (np.array(lb), np.array(ub))


def add_bounds(lb1: np.ndarray, ub1: np.ndarray,
               lb2: np.ndarray, ub2: np.ndarray) -> Bounds:
    """Bounds for elementwise addition: x + y.

    Parameters
    ----------
    lb1, ub1 : np.ndarray
        Bounds for the first operand.
    lb2, ub2 : np.ndarray
        Bounds for the second operand.

    Returns
    -------
    Bounds
        Bounds for the sum.
    """
    return (lb1 + lb2, ub1 + ub2)


def sum_bounds(lb: np.ndarray, ub: np.ndarray,
               axis: Optional[Union[int, Tuple[int, ...]]] = None,
               keepdims: bool = False) -> Bounds:
    """Bounds for sum reduction.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression being summed.
    axis : None or int or tuple of ints
        Axis or axes along which to sum.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the sum.
    """
    new_lb = np.sum(lb, axis=axis, keepdims=keepdims)
    new_ub = np.sum(ub, axis=axis, keepdims=keepdims)
    return (new_lb, new_ub)


def neg_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for negation: -x.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.

    Returns
    -------
    Bounds
        Bounds for the negation.
    """
    return (-ub, -lb)


def mul_bounds(lb1: np.ndarray, ub1: np.ndarray,
               lb2: np.ndarray, ub2: np.ndarray) -> Bounds:
    """Bounds for elementwise multiplication: x * y.

    Uses interval arithmetic: the result is [min(products), max(products)]
    where products are all combinations of endpoints.

    Parameters
    ----------
    lb1, ub1 : np.ndarray
        Bounds for the first operand.
    lb2, ub2 : np.ndarray
        Bounds for the second operand.

    Returns
    -------
    Bounds
        Bounds for the product.
    """
    # All four products of interval endpoints
    p1 = lb1 * lb2
    p2 = lb1 * ub2
    p3 = ub1 * lb2
    p4 = ub1 * ub2

    # Stack along a new axis and take min/max
    products = np.stack([p1, p2, p3, p4], axis=0)
    new_lb = np.min(products, axis=0)
    new_ub = np.max(products, axis=0)
    return (new_lb, new_ub)


def div_bounds(lb1: np.ndarray, ub1: np.ndarray,
               lb2: np.ndarray, ub2: np.ndarray) -> Bounds:
    """Bounds for elementwise division: x / y.

    Note: If the divisor interval contains zero, returns unbounded.

    Parameters
    ----------
    lb1, ub1 : np.ndarray
        Bounds for the numerator.
    lb2, ub2 : np.ndarray
        Bounds for the divisor.

    Returns
    -------
    Bounds
        Bounds for the quotient.
    """
    # Check for division by interval containing zero
    contains_zero = (lb2 <= 0) & (ub2 >= 0)

    # Compute 1/[lb2, ub2]
    # When lb2 and ub2 have the same sign, reciprocal reverses the interval
    inv_lb = np.where(contains_zero, -np.inf, 1.0 / ub2)
    inv_ub = np.where(contains_zero, np.inf, 1.0 / lb2)

    # Now multiply by numerator bounds
    return mul_bounds(lb1, ub1, inv_lb, inv_ub)


def scale_bounds(lb: np.ndarray, ub: np.ndarray, c: float) -> Bounds:
    """Bounds for scalar multiplication: c * x.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    c : float
        The scalar multiplier.

    Returns
    -------
    Bounds
        Bounds for the scaled expression.
    """
    if c >= 0:
        return (c * lb, c * ub)
    else:
        return (c * ub, c * lb)


def abs_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise absolute value: |x|.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.

    Returns
    -------
    Bounds
        Bounds for the absolute value.
    """
    # When interval spans zero, lower bound is 0
    # When interval is entirely positive, |x| = x
    # When interval is entirely negative, |x| = -x
    spans_zero = (lb <= 0) & (ub >= 0)
    entirely_positive = lb >= 0
    entirely_negative = ub <= 0

    new_lb = np.where(spans_zero, 0.0,
                      np.where(entirely_positive, lb, -ub))
    new_ub = np.where(entirely_positive, ub,
                      np.where(entirely_negative, -lb,
                               np.maximum(-lb, ub)))
    return (new_lb, new_ub)


def maximum_bounds(bounds_list: List[Bounds]) -> Bounds:
    """Bounds for elementwise maximum: max(x1, x2, ...).

    Parameters
    ----------
    bounds_list : list of Bounds
        List of (lb, ub) tuples for each argument.

    Returns
    -------
    Bounds
        Bounds for the maximum.
    """
    # Use reduce with np.maximum to handle broadcasting between different shapes
    lb_result = bounds_list[0][0]
    ub_result = bounds_list[0][1]
    for lb, ub in bounds_list[1:]:
        lb_result = np.maximum(lb_result, lb)
        ub_result = np.maximum(ub_result, ub)
    return (lb_result, ub_result)


def minimum_bounds(bounds_list: List[Bounds]) -> Bounds:
    """Bounds for elementwise minimum: min(x1, x2, ...).

    Parameters
    ----------
    bounds_list : list of Bounds
        List of (lb, ub) tuples for each argument.

    Returns
    -------
    Bounds
        Bounds for the minimum.
    """
    # Use reduce with np.minimum to handle broadcasting between different shapes
    lb_result = bounds_list[0][0]
    ub_result = bounds_list[0][1]
    for lb, ub in bounds_list[1:]:
        lb_result = np.minimum(lb_result, lb)
        ub_result = np.minimum(ub_result, ub)
    return (lb_result, ub_result)


def max_reduction_bounds(lb: np.ndarray, ub: np.ndarray,
                         axis: Optional[Union[int, Tuple[int, ...]]] = None,
                         keepdims: bool = False) -> Bounds:
    """Bounds for max reduction: max(x, axis=axis).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis or axes along which to take max.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the max.
    """
    new_lb = np.max(lb, axis=axis, keepdims=keepdims)
    new_ub = np.max(ub, axis=axis, keepdims=keepdims)
    return (new_lb, new_ub)


def min_reduction_bounds(lb: np.ndarray, ub: np.ndarray,
                         axis: Optional[Union[int, Tuple[int, ...]]] = None,
                         keepdims: bool = False) -> Bounds:
    """Bounds for min reduction: min(x, axis=axis).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis or axes along which to take min.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the min.
    """
    new_lb = np.min(lb, axis=axis, keepdims=keepdims)
    new_ub = np.min(ub, axis=axis, keepdims=keepdims)
    return (new_lb, new_ub)


def power_bounds(lb: np.ndarray, ub: np.ndarray, p: float) -> Bounds:
    """Bounds for elementwise power: x^p.

    Handles different cases based on p:
    - p > 0: behavior depends on whether p is even/odd and sign of interval
    - p < 0: requires positive interval
    - p = 0: constant 1

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the base expression.
    p : float
        The power.

    Returns
    -------
    Bounds
        Bounds for the power.
    """
    if p == 0:
        return (np.ones_like(lb), np.ones_like(ub))

    if p > 0:
        if p == int(p) and int(p) % 2 == 0:
            # Even integer power: x^(2k)
            # If interval entirely positive: [lb^p, ub^p]
            # If interval entirely negative: [ub^p, lb^p]
            # If interval spans zero: [0, max(lb^p, ub^p)]
            spans_zero = (lb <= 0) & (ub >= 0)
            entirely_positive = lb >= 0

            lb_power = np.abs(lb) ** p
            ub_power = np.abs(ub) ** p

            new_lb = np.where(spans_zero, 0.0,
                              np.where(entirely_positive, lb_power, ub_power))
            new_ub = np.where(spans_zero, np.maximum(lb_power, ub_power),
                              np.where(entirely_positive, ub_power, lb_power))
            return (new_lb, new_ub)
        else:
            # Odd integer power or non-integer positive power
            # For odd powers, x^p is monotonic: [lb^p, ub^p]
            # For non-integer powers, typically only defined for x >= 0
            # We'll handle the general case conservatively
            if p == int(p):
                # Odd integer power: monotonic
                return (lb ** p, ub ** p)
            else:
                # Non-integer power: requires x >= 0 for real result
                # If lb < 0, we have undefined behavior, return unbounded for those
                valid = lb >= 0
                new_lb = np.where(valid, lb ** p, -np.inf)
                new_ub = np.where(valid, ub ** p, np.inf)
                return (new_lb, new_ub)
    else:
        # Negative power: x^(-|p|) = 1/x^|p|
        # Only defined for x != 0
        # For positive intervals: [ub^p, lb^p] (reverses order)
        # For negative intervals with odd |p|: [ub^p, lb^p]
        # Intervals spanning zero: unbounded
        spans_zero = (lb <= 0) & (ub >= 0)
        entirely_positive = lb > 0

        if p == int(p) and int(-p) % 2 == 1:
            # Negative odd power: monotonically decreasing
            new_lb = np.where(spans_zero, -np.inf, np.minimum(lb ** p, ub ** p))
            new_ub = np.where(spans_zero, np.inf, np.maximum(lb ** p, ub ** p))
        else:
            # Negative even power or non-integer: only for positive x
            new_lb = np.where(entirely_positive, ub ** p, -np.inf)
            new_ub = np.where(entirely_positive, lb ** p, np.inf)

        return (new_lb, new_ub)


def exp_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise exponential: exp(x).

    exp is monotonically increasing, so bounds are [exp(lb), exp(ub)].

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for the exponential.
    """
    return (np.exp(lb), np.exp(ub))


def log_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise natural log: log(x).

    log is monotonically increasing on (0, inf), undefined for x <= 0.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for the logarithm.
    """
    # log is only defined for positive arguments
    valid = lb > 0
    new_lb = np.where(valid, np.log(lb), -np.inf)
    new_ub = np.where(ub > 0, np.log(ub), np.inf)
    return (new_lb, new_ub)


def sqrt_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise square root: sqrt(x).

    sqrt is monotonically increasing on [0, inf).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument (must be >= 0).

    Returns
    -------
    Bounds
        Bounds for the square root.
    """
    # sqrt is only defined for non-negative arguments
    new_lb = np.where(lb >= 0, np.sqrt(lb), -np.inf)
    new_ub = np.where(ub >= 0, np.sqrt(ub), np.inf)
    return (new_lb, new_ub)


def norm1_bounds(lb: np.ndarray, ub: np.ndarray,
                 axis: Optional[Union[int, Tuple[int, ...]]] = None,
                 keepdims: bool = False) -> Bounds:
    """Bounds for 1-norm: sum(|x|).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis along which to compute the norm.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the 1-norm.
    """
    abs_lb, abs_ub = abs_bounds(lb, ub)
    return sum_bounds(abs_lb, abs_ub, axis=axis, keepdims=keepdims)


def norm_inf_bounds(lb: np.ndarray, ub: np.ndarray,
                    axis: Optional[Union[int, Tuple[int, ...]]] = None,
                    keepdims: bool = False) -> Bounds:
    """Bounds for infinity-norm: max(|x|).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis along which to compute the norm.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the infinity norm.
    """
    abs_lb, abs_ub = abs_bounds(lb, ub)
    return max_reduction_bounds(abs_lb, abs_ub, axis=axis, keepdims=keepdims)


def broadcast_bounds(lb: np.ndarray, ub: np.ndarray,
                     target_shape: Tuple[int, ...]) -> Bounds:
    """Broadcast bounds to a target shape.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds to broadcast.
    target_shape : tuple of ints
        Target shape.

    Returns
    -------
    Bounds
        Broadcasted bounds.
    """
    return (np.broadcast_to(lb, target_shape), np.broadcast_to(ub, target_shape))


def reshape_bounds(lb: np.ndarray, ub: np.ndarray,
                   new_shape: Tuple[int, ...]) -> Bounds:
    """Reshape bounds to a new shape.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds to reshape.
    new_shape : tuple of ints
        New shape.

    Returns
    -------
    Bounds
        Reshaped bounds.
    """
    return (lb.reshape(new_shape), ub.reshape(new_shape))


def transpose_bounds(lb: np.ndarray, ub: np.ndarray,
                     axes: Optional[Tuple[int, ...]] = None) -> Bounds:
    """Transpose bounds.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds to transpose.
    axes : tuple of ints or None
        Permutation of axes.

    Returns
    -------
    Bounds
        Transposed bounds.
    """
    return (np.transpose(lb, axes), np.transpose(ub, axes))


def index_bounds(lb: np.ndarray, ub: np.ndarray, key) -> Bounds:
    """Index into bounds.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds to index.
    key : index
        Index/slice to apply.

    Returns
    -------
    Bounds
        Indexed bounds.
    """
    return (lb[key], ub[key])


def matmul_bounds(lb1: np.ndarray, ub1: np.ndarray,
                  lb2: np.ndarray, ub2: np.ndarray) -> Bounds:
    """Bounds for matrix multiplication: x @ y.

    This uses interval arithmetic for matrix multiplication.

    Parameters
    ----------
    lb1, ub1 : np.ndarray
        Bounds for the first matrix.
    lb2, ub2 : np.ndarray
        Bounds for the second matrix.

    Returns
    -------
    Bounds
        Bounds for the matrix product.
    """
    # For matrix multiplication C = A @ B where A in [lb1, ub1] and B in [lb2, ub2]:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Each term A[i,k] * B[k,j] has bounds from interval multiplication
    # The sum of intervals gives the final bounds

    # Compute all four products
    p1 = lb1 @ lb2
    p2 = lb1 @ ub2
    p3 = ub1 @ lb2
    p4 = ub1 @ ub2

    # For a more accurate bound, we need to consider positive and negative parts
    # separately, but this is a conservative approximation
    lb1_pos = np.maximum(lb1, 0)
    lb1_neg = np.minimum(lb1, 0)
    ub1_pos = np.maximum(ub1, 0)
    ub1_neg = np.minimum(ub1, 0)

    lb2_pos = np.maximum(lb2, 0)
    lb2_neg = np.minimum(lb2, 0)
    ub2_pos = np.maximum(ub2, 0)
    ub2_neg = np.minimum(ub2, 0)

    # Lower bound: min contributions from positive-positive, negative-negative parts
    # Upper bound: max contributions
    new_lb = (lb1_pos @ lb2_neg + lb1_neg @ ub2_pos +
              ub1_neg @ ub2_neg + ub1_pos @ lb2_pos)
    new_ub = (ub1_pos @ ub2_pos + ub1_neg @ lb2_neg +
              lb1_neg @ lb2_pos + lb1_pos @ ub2_neg)

    # Conservative fallback using element-wise min/max
    products = np.stack([p1, p2, p3, p4], axis=0)
    conservative_lb = np.min(products, axis=0)
    conservative_ub = np.max(products, axis=0)

    # Use the tighter of the two approaches
    new_lb = np.maximum(new_lb, conservative_lb)
    new_ub = np.minimum(new_ub, conservative_ub)

    return (new_lb, new_ub)


def refine_bounds_from_sign(lb: np.ndarray, ub: np.ndarray,
                            is_nonneg: bool, is_nonpos: bool) -> Bounds:
    """Refine bounds based on sign information.

    Parameters
    ----------
    lb, ub : np.ndarray
        Current bounds.
    is_nonneg : bool
        Whether the expression is known to be non-negative.
    is_nonpos : bool
        Whether the expression is known to be non-positive.

    Returns
    -------
    Bounds
        Refined bounds.
    """
    if is_nonneg:
        lb = np.maximum(lb, 0)
    if is_nonpos:
        ub = np.minimum(ub, 0)
    return (lb, ub)
