"""Hagan-West (2006) monotone-convex forward-rate interpolation.

Reference: P. Hagan & G. West, "Interpolation Methods for Curve Construction",
Applied Mathematical Finance 13(2), 2006.

Given pillars ``0 = t_0 < t_1 < ... < t_N`` and discount factors ``P(0, t_i)``:

1.  Discrete forward for interval ``i`` (covering ``[t_{i-1}, t_i]``)::

        F_i = (ln P(0, t_{i-1}) - ln P(0, t_i)) / (t_i - t_{i-1})

2.  Knot instantaneous forwards ``f_i`` by a length-weighted blend of the two
    adjacent discrete forwards, with the standard boundary rules, then
    (when every ``F_i >= 0``) collared into ``[0, 2 * min(adjacent F)]`` so the
    interpolant stays non-negative.

3.  On each interval, with ``x = (t - t_{i-1}) / (t_i - t_{i-1})`` and
    ``g_0 = f_{i-1} - F_i``, ``g_1 = f_i - F_i``, the deviation ``g(x) = f - F_i``
    is the quadratic (or a flat + quadratic spliced form) that satisfies
    ``g(0) = g_0``, ``g(1) = g_1`` and ``INT_0^1 g dx = 0`` -- so the interval
    reprices ``F_i`` exactly. Hagan-West branch on ``(g_0, g_1)`` to keep the
    forward monotone/convex where the data is, without overshoot.

4.  ``g`` is piecewise polynomial, so ``INT_0^t f`` is closed form and
    ``P(0, t) = exp(-INT_0^t f)``, ``z(t) = -ln P(0, t) / t`` (``z(0) = f_0``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from finmodels.curves.base import BaseYieldCurve

_ZERO, _QUAD, _CASE_II, _CASE_III, _CASE_IV = 0, 1, 2, 3, 4


def _classify(g0: float, g1: float) -> tuple[int, float, float]:
    """Hagan-West interval case for deviation endpoints ``(g0, g1)``.

    Returns ``(case, eta, A)``; ``eta`` / ``A`` are dummy for cases that ignore
    them (kept in a safe open interval so vectorized formulas never divide by 0).
    """
    if g0 == 0.0 and g1 == 0.0:
        return _ZERO, 0.5, 0.0
    # (i) the plain repricing quadratic is already monotone on [0, 1]
    if (g0 <= 0.0 and -0.5 * g0 <= g1 <= -2.0 * g0) or (
        g0 >= 0.0 and -2.0 * g0 <= g1 <= -0.5 * g0
    ):
        return _QUAD, 0.5, 0.0
    # (ii) flat at g0, then a monotone parabola up to g1
    if (g0 < 0.0 and g1 > -2.0 * g0) or (g0 > 0.0 and g1 < -2.0 * g0):
        eta = (g1 + 2.0 * g0) / (g1 - g0)
        return _CASE_II, eta, 0.0
    # (iii) a monotone parabola from g0, then flat at g1
    if (g0 > 0.0 and -0.5 * g0 < g1 < 0.0) or (g0 < 0.0 and 0.0 < g1 < -0.5 * g0):
        eta = 3.0 * g1 / (g1 - g0)
        return _CASE_III, eta, 0.0
    # (iv) g0, g1 same sign: two parabolas meeting at level A
    if g0 * g1 > 0.0:
        eta = g1 / (g0 + g1)
        A = -g0 * g1 / (g0 + g1)
        return _CASE_IV, eta, A
    # degenerate (exactly one of g0, g1 is zero): plain quadratic still reprices
    return _QUAD, 0.5, 0.0


def _g_of_x(x, g0, g1, case, eta, A):
    """Deviation ``g(x)`` (broadcast; ``x`` and the params are aligned arrays)."""
    xx = x * x
    quad = g0 * (1.0 - 4.0 * x + 3.0 * xx) + g1 * (-2.0 * x + 3.0 * xx)
    with np.errstate(divide="ignore", invalid="ignore"):
        c2 = np.where(
            x <= eta, g0, g0 + (g1 - g0) * np.square((x - eta) / (1.0 - eta))
        )
        c3 = np.where(
            x < eta, g1 + (g0 - g1) * np.square((eta - x) / eta), g1
        )
        c4 = np.where(
            x <= eta,
            A + (g0 - A) * np.square((eta - x) / eta),
            A + (g1 - A) * np.square((x - eta) / (1.0 - eta)),
        )
    return np.select(
        [case == _ZERO, case == _QUAD, case == _CASE_II, case == _CASE_III, case == _CASE_IV],
        [np.zeros_like(x), quad, c2, c3, c4],
        default=quad,
    )


def _gint_of_x(x, g0, g1, case, eta, A):
    """Closed-form ``INT_0^x g(s) ds`` for the same branching."""
    xx = x * x
    quad = g0 * (x - 2.0 * xx + x * xx) + g1 * (-xx + x * xx)
    with np.errstate(divide="ignore", invalid="ignore"):
        c2 = np.where(
            x <= eta,
            g0 * x,
            g0 * x + (g1 - g0) * (x - eta) ** 3 / (3.0 * np.square(1.0 - eta)),
        )
        c3 = np.where(
            x < eta,
            g1 * x + (g0 - g1) * (eta**3 - (eta - x) ** 3) / (3.0 * np.square(eta)),
            g1 * x + (g0 - g1) * eta / 3.0,
        )
        c4 = np.where(
            x <= eta,
            A * x + (g0 - A) * (eta**3 - (eta - x) ** 3) / (3.0 * np.square(eta)),
            A * x
            + (g0 - A) * eta / 3.0
            + (g1 - A) * (x - eta) ** 3 / (3.0 * np.square(1.0 - eta)),
        )
    return np.select(
        [case == _ZERO, case == _QUAD, case == _CASE_II, case == _CASE_III, case == _CASE_IV],
        [np.zeros_like(x), quad, c2, c3, c4],
        default=quad,
    )


class HaganWestInterpolator(BaseYieldCurve):
    """Monotone-convex forward-rate curve (Hagan-West 2006)."""

    def __init__(
        self,
        pillars: Sequence[float] | np.ndarray,
        zero_rates: Sequence[float] | np.ndarray | None = None,
        *,
        discount_factors: Sequence[float] | np.ndarray | None = None,
        enforce_positivity: bool = True,
    ) -> None:
        pillars = np.asarray(pillars, dtype=float)
        if pillars.ndim != 1 or pillars.size < 2:
            raise ValueError("need at least two pillars")
        if np.any(pillars <= 0.0) or np.any(np.diff(pillars) <= 0.0):
            raise ValueError("pillars must be strictly ascending and positive")

        if (zero_rates is None) == (discount_factors is None):
            raise ValueError("provide exactly one of zero_rates or discount_factors")

        if discount_factors is not None:
            dfs = np.asarray(discount_factors, dtype=float)
            if dfs.shape != pillars.shape or np.any(dfs <= 0.0):
                raise ValueError("discount_factors must be positive and aligned")
            self.zero_rates = -np.log(dfs) / pillars
        else:
            self.zero_rates = np.asarray(zero_rates, dtype=float)
            if self.zero_rates.shape != pillars.shape:
                raise ValueError("zero_rates must align with pillars")
            dfs = np.exp(-self.zero_rates * pillars)

        self.pillars = pillars

        # Internal grid with t_0 = 0, P(0,0) = 1 prepended.
        t = np.concatenate(([0.0], pillars))
        p = np.concatenate(([1.0], dfs))
        n = t.size - 1  # number of intervals / pillars
        dt = np.diff(t)  # length n
        f_disc = (np.log(p[:-1]) - np.log(p[1:])) / dt  # F_i, length n

        # Knot instantaneous forwards f_0 .. f_n.
        f = np.empty(n + 1)
        for i in range(1, n):
            span = t[i + 1] - t[i - 1]
            f[i] = dt[i - 1] / span * f_disc[i] + dt[i] / span * f_disc[i - 1]
        f[0] = f_disc[0] - 0.5 * (f[1] - f_disc[0])
        f[n] = f_disc[n - 1] - 0.5 * (f[n - 1] - f_disc[n - 1])

        if enforce_positivity and np.all(f_disc >= 0.0):
            f[0] = min(max(f[0], 0.0), 2.0 * f_disc[0])
            for i in range(1, n):
                f[i] = min(max(f[i], 0.0), 2.0 * min(f_disc[i - 1], f_disc[i]))
            f[n] = min(max(f[n], 0.0), 2.0 * f_disc[n - 1])

        # Per-interval coefficients (index 1..n; slot 0 is padding).
        g0 = np.zeros(n + 1)
        g1 = np.zeros(n + 1)
        case = np.zeros(n + 1, dtype=int)
        eta = np.full(n + 1, 0.5)
        amp = np.zeros(n + 1)
        for i in range(1, n + 1):
            g0[i] = f[i - 1] - f_disc[i - 1]
            g1[i] = f[i] - f_disc[i - 1]
            case[i], eta[i], amp[i] = _classify(g0[i], g1[i])

        self._t = t
        self._n = n
        self._dt = np.concatenate(([1.0], dt))  # 1-indexed like the rest
        self._f_disc = np.concatenate(([0.0], f_disc))
        self._f = f
        self._g0, self._g1, self._case, self._eta, self._A = g0, g1, case, eta, amp
        self._cumint = -np.log(p)  # INT_0^{t_i} f = -ln P(0, t_i)

    # ------------------------------------------------------------------ internals
    def _locate(self, tt: np.ndarray):
        i = np.clip(np.searchsorted(self._t, tt, side="left"), 1, self._n)
        beyond = tt > self._t[self._n]
        i = np.where(beyond, self._n, i)
        x = (tt - self._t[i - 1]) / self._dt[i]
        return i, x, beyond

    def _forward(self, tt: np.ndarray) -> np.ndarray:
        i, x, beyond = self._locate(tt)
        g = _g_of_x(
            x, self._g0[i], self._g1[i], self._case[i], self._eta[i], self._A[i]
        )
        f = self._f_disc[i] + g
        return np.where(beyond, self._f[self._n], f)

    def _integral(self, tt: np.ndarray) -> np.ndarray:
        i, x, beyond = self._locate(tt)
        gint = _gint_of_x(
            x, self._g0[i], self._g1[i], self._case[i], self._eta[i], self._A[i]
        )
        partial = self._dt[i] * (self._f_disc[i] * x + gint)
        total = self._cumint[i - 1] + partial
        tail = self._cumint[self._n] + self._f[self._n] * (tt - self._t[self._n])
        return np.where(beyond, tail, total)

    def _wrap(self, values: np.ndarray, t) -> float | np.ndarray:
        arr = np.asarray(t, dtype=float)
        if np.isscalar(t) or arr.ndim == 0:
            return float(values[0])
        return values

    # ------------------------------------------------------------------- public
    def instantaneous_forward_rate(
        self, t: float | np.ndarray
    ) -> float | np.ndarray:
        arr = np.asarray(t, dtype=float)
        if np.any(arr < 0.0):
            raise ValueError("t must be non-negative")
        return self._wrap(self._forward(np.atleast_1d(arr).astype(float)), t)

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(t, dtype=float)
        if np.any(arr < 0.0):
            raise ValueError("t must be non-negative")
        df = np.exp(-self._integral(np.atleast_1d(arr).astype(float)))
        return self._wrap(df, t)

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(t, dtype=float)
        if np.any(arr < 0.0):
            raise ValueError("t must be non-negative")
        tt = np.atleast_1d(arr).astype(float)
        integral = self._integral(tt)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(tt > 0.0, integral / tt, self._f[0])
        return self._wrap(z, t)
