import CausalAlgebraicGeometry.AntichainTiling
import CausalAlgebraicGeometry.GrowthRateIs16
import CausalAlgebraicGeometry.GaugeConnection
import CausalAlgebraicGeometry.DimensionLaw
import CausalAlgebraicGeometry.TightUpperBound
import CausalAlgebraicGeometry.RhoEquals16

-- === SUBSTANCE CHECK ===
-- For each key theorem, print its TYPE to see what it actually says.
-- A tautology would be something like "True" or "x = x".
-- A genuine theorem has real mathematical content.

#check @CausalAlgebraicGeometry.AntichainTiling.tiling_inequality
-- Expected: numConvexDim d m ^ ac.card ≤ numConvexDim d (k * m)

#check @CausalAlgebraicGeometry.DimensionLaw.numConvexDim_supermul
-- Expected: numConvexDim d m * numConvexDim d n ≤ numConvexDim d (m + n)

#check @CausalAlgebraicGeometry.DimensionLaw.numConvexDim_le_exp
-- Expected: numConvexDim d m ≤ downsetCountDim d m * upsetCountDim d m

#check @CausalAlgebraicGeometry.DimensionLaw.numConvexDim_exponential_lower
-- Expected: 2 ^ m ≤ numConvexDim d m

#check @CausalAlgebraicGeometry.DimensionLaw.dimension_law
-- Expected: 2^m ≤ numConvexDim d m ∧ upper bound ∧ monotonicity

#check @CausalAlgebraicGeometry.GrowthRateIs16.growth_constant_eq_neg_log_sixteen
-- Expected: neg_log_subadditive.lim = -Real.log 16

#check @CausalAlgebraicGeometry.GaugeConnection.cylinder_wilson_loop_trace
-- Expected: intervalSize / card = (t-1)/t

#check @CausalAlgebraicGeometry.TightUpperBound.numGridConvex_le_choose_sq
-- Expected: numGridConvex m m ≤ Nat.choose (2*m) m ^ 2

#check @CausalAlgebraicGeometry.TightUpperBound.card_downsets_eq_choose
-- Expected: downsets count = Nat.choose (2*m) m

#check @CausalAlgebraicGeometry.Supermultiplicativity.supermultiplicativity
-- Expected: numGridConvex m m * numGridConvex n n ≤ numGridConvex (m+n) (m+n)

#check @CausalAlgebraicGeometry.RhoEquals16.numGridConvex_ge_catalan_bound
-- Expected: C(2m,m)^2/(2(m+1)) ≤ numGridConvex m m

#check @CausalAlgebraicGeometry.AntichainTiling.antichain_union_convex
-- Expected: union of convex sets at antichain positions is convex

#check @CausalAlgebraicGeometry.AntichainTiling.block_incomparable
-- Expected: points from incomparable blocks are incomparable

#check @CausalAlgebraicGeometry.GrowthRateHelper.neg_log_div_le
-- Expected: upper bound on -log(a(n))/n

#check @CausalAlgebraicGeometry.GrowthRateHelper.correction_tendsto_zero
-- Expected: correction term → 0
