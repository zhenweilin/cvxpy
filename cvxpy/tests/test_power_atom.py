"""
Copyright 2025 CVXPY developers

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

import warnings

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestPowerAtom(BaseTest):
    """Unit tests for power atom."""

    def _get_cone_counts(self, prob, solver):
        """Helper to get SOC and power cone counts from problem data."""
        data, _, _ = prob.get_problem_data(solver)
        dims = data['dims']
        return len(dims.soc), len(dims.p3d)

    def test_explicitapprox_true_forces_soc(self) -> None:
        """Test that approx=True forces SOC even with power-cone-capable solver."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        )
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        # Should use SOC because user explicitly requested approximation
        self.assertGreater(soc_count, 0, "approx=True should force SOC cones")
        self.assertEqual(p3d_count, 0, "approx=True should not use power cones")

    def test_explicitapprox_false_forces_power_cones(self) -> None:
        """Test that approx=False forces power cones."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        # Should use power cones
        self.assertGreater(p3d_count, 0, "approx=False should use power cones")
        self.assertEqual(soc_count, 0, "approx=False should not use SOC cones")

    def test_powerapprox(self) -> None:
        """Test power atom with approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_noapprox(self) -> None:
        """Test power atom without approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_with_and_withoutapprox_low(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, -1.5, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, -1.5, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_withoutapprox_mid(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 0.8, approx=True) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 0.8, approx=False) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_withoutapprox_high(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 4.5, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 4.5, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_withoutapprox_even(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 8, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value

        constr = [
            cp.power(x, 8, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_noapprox_unsupported_solver(self) -> None:
        """
        Test fallback behavior: approx=False with a solver that doesn't
        support power cones should fall back to SOC.
        This test is skipped if ECOS is not installed.
        """
        if cp.ECOS not in cp.installed_solvers():
            self.skipTest("ECOS not installed.")
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        # ECOS doesn't support power cones, so should fall back to SOC
        soc_count, p3d_count = self._get_cone_counts(prob, cp.ECOS)
        self.assertGreater(soc_count, 0, "Should fall back to SOC cones")
        self.assertEqual(p3d_count, 0, "Should not use power cones with ECOS")

    def test_approx_warning_triggered_many_soc(self) -> None:
        """Warning should be triggered when many SOC constraints are needed."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            # Should have at least one warning about power approximation
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertGreater(len(power_warnings), 0,
                               "Should warn about SOC approximation")

    def test_approx_warning_not_triggered_with_approx_false(self) -> None:
        """Warning should NOT be triggered when using approx=False."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(power_warnings), 0,
                             "Should not warn when using power cones")

    def test_approx_warning_not_triggered_unsupported_solver(self) -> None:
        """Warning should NOT be triggered when solver doesn't support power cones."""
        if cp.ECOS not in cp.installed_solvers():
            self.skipTest("ECOS not installed.")
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.ECOS)
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(power_warnings), 0,
                             "Should not warn when solver doesn't support power cones")

    def test_approx_warning_not_triggered_few_soc(self) -> None:
        """Warning should NOT be triggered when few SOC constraints are needed."""
        x = cp.Variable(3)
        # x^2 uses only 3 SOC constraints, which is <= 4
        prob = cp.Problem(
            cp.Minimize(x[0] + x[1] - x[2]),
            [cp.power(x, 2, approx=True) <= np.ones(3)]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            power_warnings = [
                warning for warning in w
                if "Power atom" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(power_warnings), 0,
                             "Should not warn when few SOC constraints are needed")


class TestGeoMeanApprox(BaseTest):
    """Unit tests for geo_mean approx parameter."""

    def _get_cone_counts(self, prob, solver):
        """Helper to get SOC and power cone counts from problem data."""
        data, _, _ = prob.get_problem_data(solver)
        dims = data['dims']
        # pnd is n-dimensional power cones
        pnd_count = len(dims.pnd) if hasattr(dims, 'pnd') else 0
        return len(dims.soc), pnd_count

    def test_geo_mean_approx_true_uses_soc(self) -> None:
        """Test that geo_mean with approx=True uses SOC constraints."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), [cp.sum(x) <= 3])
        soc_count, pnd_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc_count, 0, "approx=True should use SOC cones")
        self.assertEqual(pnd_count, 0, "approx=True should not use power cones")

    def test_geo_mean_approx_false_uses_power_cones(self) -> None:
        """Test that geo_mean with approx=False uses power cones."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), [cp.sum(x) <= 3])
        soc_count, pnd_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(pnd_count, 0, "approx=False should use power cones")
        self.assertEqual(soc_count, 0, "approx=False should not use SOC cones")

    def test_geo_mean_approx_true_solves(self) -> None:
        """Test that geo_mean with approx=True solves correctly."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), [cp.sum(x) <= 3])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], geo_mean = 1
        self.assertAlmostEqual(prob.value, 1.0, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_geo_mean_approx_false_solves(self) -> None:
        """Test that geo_mean with approx=False solves correctly."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), [cp.sum(x) <= 3])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], geo_mean = 1
        self.assertAlmostEqual(prob.value, 1.0, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_geo_mean_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation."""
        x = cp.Variable(4, pos=True)
        constr = [cp.sum(x) <= 4, x[0] >= 0.5]
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_geo_mean_weighted_approx(self) -> None:
        """Test weighted geo_mean with approx parameter."""
        x = cp.Variable(3, pos=True)
        weights = [1, 2, 1]
        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, weights, approx=True)),
            [cp.sum(x) <= 4]
        )
        prob.solve(solver=cp.CLARABEL)
        val_approx = prob.value
        x_approx = x.value.copy()

        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, weights, approx=False)),
            [cp.sum(x) <= 4]
        )
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_geo_mean_approx_warning_triggered(self) -> None:
        """Warning should be triggered for geo_mean with many SOC constraints."""
        x = cp.Variable(5, pos=True)
        # 5 elements will require more SOC constraints
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=True)), [cp.sum(x) <= 5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            geo_mean_warnings = [
                warning for warning in w
                if "geo_mean" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertGreater(len(geo_mean_warnings), 0,
                               "Should warn about SOC approximation")

    def test_geo_mean_approx_warning_not_triggered_with_approx_false(self) -> None:
        """Warning should NOT be triggered when using approx=False."""
        x = cp.Variable(5, pos=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x, approx=False)), [cp.sum(x) <= 5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            geo_mean_warnings = [
                warning for warning in w
                if "geo_mean" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(geo_mean_warnings), 0,
                             "Should not warn when using power cones")


class TestPnormApprox(BaseTest):
    """Unit tests for pnorm approx parameter."""

    def _get_cone_counts(self, prob, solver):
        """Helper to get SOC and power cone counts from problem data."""
        data, _, _ = prob.get_problem_data(solver)
        dims = data['dims']
        return len(dims.soc), len(dims.p3d)

    def test_pnorm_approx_true_uses_soc(self) -> None:
        """Test that pnorm with approx=True uses SOC constraints."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), [cp.sum(x) >= 3, x >= 0])
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(soc_count, 0, "approx=True should use SOC cones")
        self.assertEqual(p3d_count, 0, "approx=True should not use power cones")

    def test_pnorm_approx_false_uses_power_cones(self) -> None:
        """Test that pnorm with approx=False uses power cones."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=False)), [cp.sum(x) >= 3, x >= 0])
        soc_count, p3d_count = self._get_cone_counts(prob, cp.CLARABEL)
        self.assertGreater(p3d_count, 0, "approx=False should use power cones")

    def test_pnorm_approx_true_solves(self) -> None:
        """Test that pnorm with approx=True solves correctly."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), [cp.sum(x) >= 3, x >= 0])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], pnorm_3 = 3^(1/3)
        expected_norm = 3 ** (1/3)
        self.assertAlmostEqual(prob.value, expected_norm, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_pnorm_approx_false_solves(self) -> None:
        """Test that pnorm with approx=False solves correctly."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=False)), [cp.sum(x) >= 3, x >= 0])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is x = [1, 1, 1], pnorm_3 = 3^(1/3)
        expected_norm = 3 ** (1/3)
        self.assertAlmostEqual(prob.value, expected_norm, places=3)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=3)

    def test_pnorm_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation."""
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0, x[0] <= 2]
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_pnorm_fractional_p_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation for fractional p."""
        x = cp.Variable(3)
        constr = [cp.sum(x) >= 3, x >= 0, x[0] <= 2]
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 2.5, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 2.5, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_pnorm_concave_with_and_without_approx(self) -> None:
        """Compare answers with and without approximation for concave pnorm (p < 1)."""
        x = cp.Variable(3, pos=True)
        constr = [cp.sum(x) <= 3, x >= 0.1]
        prob = cp.Problem(cp.Maximize(cp.pnorm(x, 0.5, approx=True)), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value.copy()
        val_approx = prob.value

        prob = cp.Problem(cp.Maximize(cp.pnorm(x, 0.5, approx=False)), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, val_approx, places=3)
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_pnorm_approx_warning_triggered(self) -> None:
        """Warning should be triggered for pnorm with many SOC constraints."""
        x = cp.Variable(3)
        # p=3.3 will require more SOC constraints
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=True)), [cp.sum(x) >= 3, x >= 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            pnorm_warnings = [
                warning for warning in w
                if "pnorm" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertGreater(len(pnorm_warnings), 0,
                               "Should warn about SOC approximation")

    def test_pnorm_approx_warning_not_triggered_with_approx_false(self) -> None:
        """Warning should NOT be triggered when using approx=False."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.pnorm(x, 3.3, approx=False)), [cp.sum(x) >= 3, x >= 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve(solver=cp.CLARABEL)
            pnorm_warnings = [
                warning for warning in w
                if "pnorm" in str(warning.message)
                and "SOC constraints" in str(warning.message)
            ]
            self.assertEqual(len(pnorm_warnings), 0,
                             "Should not warn when using power cones")


class TestExotic2CommonPowerCones(BaseTest):
    """Tests for power cone conversions in Exotic2Common."""

    def test_pow3d_to_soc_conversion(self) -> None:
        """Test that pow_3d_canon correctly converts PowCone3D to SOC."""
        from cvxpy.constraints.power import PowCone3D
        from cvxpy.constraints.second_order import SOC
        from cvxpy.reductions.cone2cone.exotic2common import pow_3d_canon

        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()

        # Create PowCone3D constraint: x^0.5 * y^0.5 >= |z|
        con = PowCone3D(x, y, z, 0.5)
        canon_constr, aux_constrs = pow_3d_canon(con, [x, y, z])

        # Result should be SOC constraint (first one), and aux_constrs are the rest
        self.assertIsInstance(canon_constr, SOC)
        # All constraints (canon + aux) should be SOC
        all_constrs = [canon_constr] + aux_constrs
        for c in all_constrs:
            self.assertIsInstance(c, SOC)

    def test_pow3d_to_soc_vector_case(self) -> None:
        """Test PowCone3D to SOC conversion with vector variables."""
        from cvxpy.constraints.power import PowCone3D
        from cvxpy.constraints.second_order import SOC
        from cvxpy.reductions.cone2cone.exotic2common import pow_3d_canon

        x = cp.Variable(3, pos=True)
        y = cp.Variable(3, pos=True)
        z = cp.Variable(3)

        con = PowCone3D(x, y, z, 0.5)
        canon_constr, aux_constrs = pow_3d_canon(con, [x, y, z])

        # Result should be SOC constraint (first one), and aux_constrs are the rest
        self.assertIsInstance(canon_constr, SOC)
        # All constraints (canon + aux) should be SOC
        all_constrs = [canon_constr] + aux_constrs
        self.assertGreater(len(all_constrs), 0)
        for c in all_constrs:
            self.assertIsInstance(c, SOC)

    def test_mixed_pownd_pow3d_with_scs(self) -> None:
        """Test problem with both PowConeND and PowCone3D works with SCS.

        SCS supports PowCone3D but not PowConeND. This test verifies that:
        1. geo_mean(approx=False) produces PowConeND
        2. power(approx=False) produces PowCone3D
        3. Exotic2Common converts PowConeND -> PowCone3D
        4. The resulting problem with only PowCone3D can be solved by SCS
        """
        # geo_mean produces PowConeND
        x = cp.Variable(3, pos=True)
        gm = cp.geo_mean(x, approx=False)

        # power produces PowCone3D
        y = cp.Variable(pos=True)
        pw = cp.power(y, 0.5, approx=False)

        # Create a problem using both
        prob = cp.Problem(
            cp.Maximize(gm + pw),
            [cp.sum(x) <= 3, y <= 4]
        )

        # Solve with SCS (supports PowCone3D but not PowConeND)
        result = prob.solve(solver=cp.SCS, eps=1e-5)

        # Verify solution
        self.assertIsNotNone(result)
        np.testing.assert_allclose(x.value, [1, 1, 1], rtol=1e-2)
        np.testing.assert_allclose(y.value, 4.0, rtol=1e-2)

    def test_exotic2common_preserves_pow3d_for_capable_solver(self) -> None:
        """Test that Exotic2Common only converts what's needed.

        For SCS (PowCone3D supported, PowConeND not supported):
        - PowConeND should be converted to PowCone3D
        - Original PowCone3D should remain as PowCone3D (not converted to SOC)
        """

        # Create a problem with just PowCone3D (from power atom)
        x = cp.Variable(pos=True)
        prob = cp.Problem(
            cp.Maximize(cp.power(x, 0.5, approx=False)),
            [x <= 4]
        )

        # Get problem data for SCS
        data, _, _ = prob.get_problem_data(cp.SCS)

        # Should have power cones, not just SOC
        dims = data['dims']
        self.assertGreater(len(dims.p3d), 0,
                           "SCS should use PowCone3D, not convert to SOC")

    def test_geo_mean_approx_false_with_scs_uses_exotic2common(self) -> None:
        """Test that geo_mean(approx=False) works with SCS via Exotic2Common."""
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, approx=False)),
            [cp.sum(x) <= 3]
        )

        prob.solve(solver=cp.SCS, eps=1e-5)
        np.testing.assert_allclose(x.value, [1, 1, 1], rtol=1e-2)

    def test_pnorm_approx_false_with_scs(self) -> None:
        """Test that pnorm(approx=False) works with SCS."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(cp.pnorm(x, p=3, approx=False)),
            [cp.sum(x) == 3, x >= 0]
        )

        prob.solve(solver=cp.SCS, eps=1e-5)
        np.testing.assert_allclose(x.value, [1, 1, 1], rtol=1e-2)

    def test_pow3d_to_soc_dual_recovery(self) -> None:
        """Test that dual variables are properly recovered when pow_3d_canon is used.

        This test directly applies the Exotic2Common reduction to convert
        PowCone3D to SOC constraints, then solves and verifies that:
        1. The problem solves correctly
        2. Primal variables are properly recovered
        3. The dual value for the power cone constraint is in the inverted solution
        """
        from cvxpy.constraints.power import PowCone3D
        from cvxpy.reductions.cone2cone.exotic2common import Exotic2Common
        from cvxpy.reductions.solution import Solution

        # Create a simple problem with PowCone3D constraint
        # max t s.t. x^0.5 * y^0.5 >= t, x + y == 2, x, y >= 0
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        t = cp.Variable()

        # PowCone3D: x^alpha * y^(1-alpha) >= |t|
        # With alpha=0.5: sqrt(x*y) >= |t|
        pow_con = PowCone3D(x, y, t, 0.5)
        eq_con = x + y == 2

        prob = cp.Problem(cp.Maximize(t), [pow_con, eq_con, x >= 0.1, y >= 0.1])

        # Apply Exotic2Common reduction manually to convert PowCone3D to SOC
        reduction = Exotic2Common(prob)
        reduced_prob, inv_data = reduction.apply(prob)

        # Solve the reduced problem
        reduced_prob.solve(solver=cp.CLARABEL)

        self.assertIn(reduced_prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])

        # Create solution object and invert the reduction
        sol = Solution(
            reduced_prob.status,
            reduced_prob.value,
            {v.id: v.value for v in reduced_prob.variables()},
            {c.id: c.dual_value for c in reduced_prob.constraints},
            {}
        )
        inverted_sol = reduction.invert(sol, inv_data)

        # Verify primal variables are properly recovered
        self.assertIsNotNone(inverted_sol.primal_vars.get(x.id))
        self.assertIsNotNone(inverted_sol.primal_vars.get(y.id))
        self.assertIsNotNone(inverted_sol.primal_vars.get(t.id))

        # Verify primal solution: at optimum x=y=1, t=1 (since sqrt(1*1)=1)
        self.assertAlmostEqual(inverted_sol.primal_vars[x.id], 1.0, places=2)
        self.assertAlmostEqual(inverted_sol.primal_vars[y.id], 1.0, places=2)
        self.assertAlmostEqual(inverted_sol.primal_vars[t.id], 1.0, places=2)

        # Verify dual variables are in the inverted solution
        self.assertIn(pow_con.id, inverted_sol.dual_vars,
                      "Dual for PowCone3D should be in inverted solution")
        self.assertIsNotNone(inverted_sol.dual_vars[pow_con.id],
                             "Dual value for PowCone3D should not be None")

        # Check the dual value can be converted to expected format [dual_x, dual_y, dual_t]
        dual_val = inverted_sol.dual_vars[pow_con.id]
        # The dual from SOC is in [t, X] format, verify it exists
        self.assertIsNotNone(dual_val)

    def test_pow3d_to_soc_dual_recovery_vector(self) -> None:
        """Test dual recovery for vector PowCone3D constraints with SOC conversion."""
        from cvxpy.constraints.power import PowCone3D
        from cvxpy.reductions.cone2cone.exotic2common import Exotic2Common
        from cvxpy.reductions.solution import Solution

        # Vector problem with PowCone3D
        n = 3
        x = cp.Variable(n, pos=True)
        y = cp.Variable(n, pos=True)
        t = cp.Variable(n)

        # PowCone3D for each element: x_i^0.5 * y_i^0.5 >= |t_i|
        pow_con = PowCone3D(x, y, t, 0.5)
        sum_con = cp.sum(x) + cp.sum(y) == 6

        prob = cp.Problem(cp.Maximize(cp.sum(t)), [pow_con, sum_con, x >= 0.1, y >= 0.1])

        # Apply Exotic2Common reduction manually
        reduction = Exotic2Common(prob)
        reduced_prob, inv_data = reduction.apply(prob)

        # Solve the reduced problem
        reduced_prob.solve(solver=cp.CLARABEL)

        self.assertIn(reduced_prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])

        # Create solution object and invert
        sol = Solution(
            reduced_prob.status,
            reduced_prob.value,
            {v.id: v.value for v in reduced_prob.variables()},
            {c.id: c.dual_value for c in reduced_prob.constraints},
            {}
        )
        inverted_sol = reduction.invert(sol, inv_data)

        # At optimum, x_i = y_i = 1 for all i, t_i = 1 for all i
        np.testing.assert_allclose(inverted_sol.primal_vars[x.id], np.ones(n), rtol=0.1)
        np.testing.assert_allclose(inverted_sol.primal_vars[y.id], np.ones(n), rtol=0.1)
        np.testing.assert_allclose(inverted_sol.primal_vars[t.id], np.ones(n), rtol=0.1)

        # Verify dual variable is in the inverted solution
        self.assertIn(pow_con.id, inverted_sol.dual_vars,
                      "Dual for PowCone3D should be in inverted solution")
        self.assertIsNotNone(inverted_sol.dual_vars[pow_con.id],
                             "Dual value for vector PowCone3D should not be None")

    def test_pow3d_to_soc_non_half_alpha(self) -> None:
        """Test dual recovery for PowCone3D with non-0.5 alpha."""
        from cvxpy.constraints.power import PowCone3D
        from cvxpy.reductions.cone2cone.exotic2common import Exotic2Common
        from cvxpy.reductions.solution import Solution

        # Problem where power cone is tight at optimum
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        t = cp.Variable()

        pow_con = PowCone3D(x, y, t, 0.3)  # x^0.3 * y^0.7 >= |t|
        eq_con = x + y == 2

        prob = cp.Problem(cp.Maximize(t), [pow_con, eq_con, x >= 0.1, y >= 0.1])

        # Apply Exotic2Common reduction manually
        reduction = Exotic2Common(prob)
        reduced_prob, inv_data = reduction.apply(prob)

        # Solve the reduced problem
        reduced_prob.solve(solver=cp.CLARABEL)

        self.assertIn(reduced_prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])

        # Create solution object and invert
        sol = Solution(
            reduced_prob.status,
            reduced_prob.value,
            {v.id: v.value for v in reduced_prob.variables()},
            {c.id: c.dual_value for c in reduced_prob.constraints},
            {}
        )
        inverted_sol = reduction.invert(sol, inv_data)

        # Verify primal is recovered
        self.assertIsNotNone(inverted_sol.primal_vars.get(x.id))
        self.assertIsNotNone(inverted_sol.primal_vars.get(y.id))
        self.assertIsNotNone(inverted_sol.primal_vars.get(t.id))

        # Verify dual is in the inverted solution
        self.assertIn(pow_con.id, inverted_sol.dual_vars,
                      "Dual for PowCone3D should be in inverted solution")
        self.assertIsNotNone(inverted_sol.dual_vars[pow_con.id],
                             "Dual value for PowCone3D should not be None")


class TestPowerAtomDPP(BaseTest):
    """DPP tests for power-related atoms with approx=False."""

    def test_power_approx_false_dpp(self) -> None:
        """Test that power(approx=False) works with DPP (parameterized problems)."""
        x = cp.Variable(pos=True)
        b = cp.Parameter(pos=True)

        # Minimize x^2 subject to x >= b
        # With approx=False, this uses power cones
        prob = cp.Problem(
            cp.Minimize(cp.power(x, 2, approx=False)),
            [x >= b]
        )

        # First solve
        b.value = 2.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(x.value, 2.0, places=3)
        self.assertAlmostEqual(prob.value, 4.0, places=3)

        # Re-solve with different parameter - should reuse chain
        b.value = 3.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(x.value, 3.0, places=3)
        self.assertAlmostEqual(prob.value, 9.0, places=3)

    def test_power_approx_false_dpp_fractional_exponent(self) -> None:
        """Test power(approx=False) with fractional exponent and DPP."""
        x = cp.Variable(pos=True)
        c = cp.Parameter(pos=True)

        # Maximize x^0.5 subject to x <= c
        prob = cp.Problem(
            cp.Maximize(cp.power(x, 0.5, approx=False)),
            [x <= c]
        )

        # First solve
        c.value = 4.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(x.value, 4.0, places=3)
        self.assertAlmostEqual(prob.value, 2.0, places=3)

        # Re-solve with different parameter
        c.value = 9.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(x.value, 9.0, places=3)
        self.assertAlmostEqual(prob.value, 3.0, places=3)

    def test_geo_mean_approx_false_dpp(self) -> None:
        """Test that geo_mean(approx=False) works with DPP."""
        x = cp.Variable(3, pos=True)
        budget = cp.Parameter(pos=True)

        # Maximize geometric mean subject to sum(x) <= budget
        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, approx=False)),
            [cp.sum(x) <= budget]
        )

        # First solve - optimal is x = [budget/3, budget/3, budget/3]
        budget.value = 3.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        np.testing.assert_allclose(x.value, np.ones(3), rtol=0.01)
        self.assertAlmostEqual(prob.value, 1.0, places=3)

        # Re-solve with different budget
        budget.value = 6.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        np.testing.assert_allclose(x.value, 2.0 * np.ones(3), rtol=0.01)
        self.assertAlmostEqual(prob.value, 2.0, places=3)

    def test_geo_mean_approx_false_weighted_dpp(self) -> None:
        """Test weighted geo_mean(approx=False) with DPP."""
        x = cp.Variable(2, pos=True)
        budget = cp.Parameter(pos=True)

        # Weighted geometric mean: x[0]^(1/3) * x[1]^(2/3)
        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, [1, 2], approx=False)),
            [cp.sum(x) <= budget]
        )

        # First solve
        budget.value = 3.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal allocation: x[0] = budget/3, x[1] = 2*budget/3
        self.assertAlmostEqual(x.value[0], 1.0, places=2)
        self.assertAlmostEqual(x.value[1], 2.0, places=2)

        # Re-solve with different budget
        budget.value = 6.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(x.value[0], 2.0, places=2)
        self.assertAlmostEqual(x.value[1], 4.0, places=2)

    def test_pnorm_approx_false_dpp(self) -> None:
        """Test that pnorm(approx=False) works with DPP."""
        x = cp.Variable(3)
        target = cp.Parameter(3)

        # Minimize p-norm distance to target
        prob = cp.Problem(
            cp.Minimize(cp.pnorm(x - target, p=3, approx=False)),
            [cp.sum(x) == 3]
        )

        # First solve - target at origin
        target.value = np.zeros(3)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # With sum(x) = 3 and minimizing 3-norm, optimal is x = [1, 1, 1]
        np.testing.assert_allclose(x.value, np.ones(3), rtol=0.01)

        # Re-solve with different target
        target.value = np.array([1.0, 1.0, 1.0])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # Optimal is still x = [1, 1, 1] which equals target
        np.testing.assert_allclose(x.value, np.ones(3), rtol=0.01)
        self.assertAlmostEqual(prob.value, 0.0, places=3)

    def test_pnorm_approx_false_dpp_fractional_p(self) -> None:
        """Test pnorm(approx=False) with fractional p and DPP."""
        x = cp.Variable(2, pos=True)
        lower = cp.Parameter(2, pos=True)

        # Minimize 2.5-norm subject to lower bounds
        prob = cp.Problem(
            cp.Minimize(cp.pnorm(x, p=2.5, approx=False)),
            [x >= lower]
        )

        # First solve
        lower.value = np.array([1.0, 2.0])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        np.testing.assert_allclose(x.value, lower.value, rtol=0.01)

        # Re-solve with different lower bounds
        lower.value = np.array([2.0, 3.0])
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        np.testing.assert_allclose(x.value, lower.value, rtol=0.01)

    def test_mixed_power_geo_mean_dpp(self) -> None:
        """Test problem with both power and geo_mean (approx=False) with DPP."""
        x = cp.Variable(2, pos=True)
        y = cp.Variable(pos=True)
        budget = cp.Parameter(pos=True)

        # Maximize geo_mean(x) + sqrt(y) subject to sum(x) + y <= budget
        prob = cp.Problem(
            cp.Maximize(cp.geo_mean(x, approx=False) + cp.power(y, 0.5, approx=False)),
            [cp.sum(x) + y <= budget]
        )

        # First solve
        budget.value = 4.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        val1 = prob.value

        # Re-solve with different budget
        budget.value = 8.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        val2 = prob.value

        # Value should increase with larger budget
        self.assertGreater(val2, val1)

    def test_power_approx_false_objective_param(self) -> None:
        """Test power(approx=False) with parameter in objective coefficient."""
        x = cp.Variable(pos=True)
        coeff = cp.Parameter(pos=True)

        # Minimize coeff * x^2 subject to x >= 1
        prob = cp.Problem(
            cp.Minimize(coeff * cp.power(x, 2, approx=False)),
            [x >= 1]
        )

        # First solve
        coeff.value = 1.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(x.value, 1.0, places=3)
        self.assertAlmostEqual(prob.value, 1.0, places=3)

        # Re-solve with different coefficient
        coeff.value = 2.0
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(x.value, 1.0, places=3)
        self.assertAlmostEqual(prob.value, 2.0, places=3)
