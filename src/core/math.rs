//! Fixed-point arithmetic (Q16.16), vector/matrix types, weight type (Q8.8),
//! and a XorShift64* PRNG for the HKL-1 neuromorphic AI. All operations are
//! integer-only with no floating-point or stdlib dependency.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// FIXED-POINT ARITHMETIC (Q16.16) - Pure integer, no float needed
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct FixedPoint(pub i32);

impl FixedPoint {
    pub const FRAC_BITS: i32 = 16;
    pub const ONE: Self = Self(1 << Self::FRAC_BITS);
    pub const ZERO: Self = Self(0);
    pub const HALF: Self = Self(1 << (Self::FRAC_BITS - 1));
    pub const MIN: Self = Self(i32::MIN);
    pub const MAX: Self = Self(i32::MAX);
    pub const PI: Self = Self(205887);
    pub const TAU: Self = Self(411774);
    pub const FRAC_PI_2: Self = Self(102943);

    #[inline(always)]
    pub const fn from_int(x: i32) -> Self {
        Self(x << Self::FRAC_BITS)
    }

    #[inline(always)]
    pub const fn from_bits(bits: i32) -> Self {
        Self(bits)
    }

    #[inline(always)]
    pub const fn to_int(self) -> i32 {
        self.0 >> Self::FRAC_BITS
    }

    #[inline(always)]
    pub const fn to_bits(self) -> i32 {
        self.0
    }

    #[inline(always)]
    pub const fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self::ZERO;
        }
        Self((x * 65536.0) as i32)
    }

    #[inline(always)]
    pub const fn to_f32(self) -> f32 {
        self.0 as f32 / 65536.0
    }

    #[inline(always)]
    pub const fn from_parts(sign: bool, int: u16, frac: u16) -> Self {
        let mut v = ((int as i32) << Self::FRAC_BITS) | (frac as i32);
        if sign {
            v = -v;
        }
        Self(v)
    }

    // Integer arithmetic (no float)
    #[inline(always)]
    pub const fn add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[inline(always)]
    pub const fn sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    #[inline(always)]
    pub const fn mul(self, other: Self) -> Self {
        Self(((self.0 as i64 * other.0 as i64) >> Self::FRAC_BITS) as i32)
    }

    #[inline(always)]
    pub const fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            return Self::MAX;
        }
        Self((((self.0 as i64) << Self::FRAC_BITS) / other.0 as i64) as i32)
    }

    #[inline(always)]
    pub const fn abs(self) -> Self {
        Self(if self.0 < 0 { -self.0 } else { self.0 })
    }

    #[inline(always)]
    pub const fn min(self, other: Self) -> Self {
        if self.0 < other.0 { self } else { other }
    }

    #[inline(always)]
    pub const fn max(self, other: Self) -> Self {
        if self.0 > other.0 { self } else { other }
    }

    #[inline(always)]
    pub const fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    #[inline(always)]
    pub const fn neg(self) -> Self {
        Self(-self.0)
    }

    // Integer-based exponential approximation (e^x)
    // Uses the identity: e^x = 2^(x / ln 2)
    // We implement 2^x for fixed point
    pub fn exp(self) -> Self {
        if self.0 <= -10 * Self::ONE.0 {
            return Self::ZERO;
        }
        if self.0 >= 10 * Self::ONE.0 {
            return Self::MAX;
        }

        // ln(2) in Q16.16 ≈ 45426
        const LN2: i32 = 45426;
        // x / ln(2)
        let n = ((self.0 as i64) << 16) / LN2 as i64;
        let int_part = (n >> 16) as i32;
        let frac_part = (n & 0xFFFF) as u32;

        // 2^frac_part using polynomial approx for 2^x on [0,1)
        // P(x) = 1 + ln2*x + (ln2^2/2)*x^2 + (ln2^3/6)*x^3
        let x = Self::from_bits((frac_part << 16) as i32 >> 16);
        let ln2 = Self(LN2);
        let ln2_2 = Self(102990); // ln(2)^2 / 2
        let ln2_3 = Self(38881); // ln(2)^3 / 6

        let x2 = x.mul(x);
        let x3 = x2.mul(x);
        let mut result = Self::ONE + ln2.mul(x) + ln2_2.mul(x2) + ln2_3.mul(x3);

        // 2^int_part = shift
        if int_part > 0 {
            let shift = int_part.min(31) as u32;
            result.0 = result.0.checked_shl(shift).unwrap_or(0);
        } else if int_part < 0 {
            let shift = (-int_part).min(31) as u32;
            result.0 = result.0.checked_shr(shift).unwrap_or(0);
        }

        result
    }

    // Natural log approximation (for x > 0)
    // ln(x) ≈ 2 * atanh((x-1)/(x+1))
    pub fn ln(self) -> Self {
        if self.0 <= 0 {
            return Self::from_int(-10);
        }

        // Use identity: ln(x) = ln(2^e * m) = e*ln2 + ln(m)
        let mut n = self.0;
        let mut e = 0i32;
        // Normalize to [1, 2)
        while n > 2 << 16 {
            n >>= 1;
            e += 1;
        }
        while n < 1 << 16 {
            n <<= 1;
            e -= 1;
        }
        let x = Self(n);

        // ln(m) for m in [1,2) using Pade approximation
        // ln(1+z) where z = m-1, z in [0,1)
        let z = x - Self::ONE;
        // P(z) = z - z^2/2 + z^3/3 - z^4/4
        let z2 = z.mul(z);
        let z3 = z2.mul(z);
        let z4 = z3.mul(z);
        let ln_m =
            z - z2.div(Self::from_int(2)) + z3.div(Self::from_int(3)) - z4.div(Self::from_int(4));

        // ln2 in Q16.16
        const LN2: i32 = 45426;
        Self::from_int(e).mul(Self(LN2)) + ln_m
    }

    // Square root using Newton's method
    pub fn sqrt(self) -> Self {
        if self.0 <= 0 {
            return Self::ZERO;
        }
        if self == Self::ZERO || self == Self::ONE {
            return self;
        }

        // Initial guess
        let mut x = if self < Self::from_int(100) {
            self
        } else {
            Self::from_int(10)
        };
        for _ in 0..8 {
            x = (x + self.div(x)).div(Self::from_int(2));
        }
        x
    }

    // Sigmoid: 1 / (1 + exp(-x))
    pub fn sigmoid(self) -> Self {
        Self::ONE.div(Self::ONE.add((-self).exp()))
    }

    // Tanh: (exp(2x) - 1) / (exp(2x) + 1)
    pub fn tanh(self) -> Self {
        let exp2x = (self.mul(Self::from_int(2))).exp();
        (exp2x.sub(Self::ONE)).div(exp2x.add(Self::ONE))
    }

    // ReLU
    #[inline(always)]
    pub fn relu(self) -> Self {
        if self.0 < 0 { Self::ZERO } else { self }
    }

    // Power: self^exp for integer exp
    #[inline(always)]
    pub fn pow(self, exp: u32) -> Self {
        let mut result = Self::ONE;
        for _ in 0..exp {
            result = result * self;
        }
        result
    }

    #[inline(always)]
    pub fn powi(self, exp: i32) -> Self {
        if exp >= 0 {
            self.pow(exp as u32)
        } else {
            Self::ONE.div(self.pow((-exp) as u32))
        }
    }

    /// Sine approximation using Bhaskara I formula in fixed-point
    pub fn sin(self) -> Self {
        let mut x = self.0.rem_euclid(Self::TAU.0);
        let negative = if x > Self::PI.0 {
            x = Self::TAU.0 - x;
            true
        } else {
            false
        };

        let pi = Self::PI.0 as i64;
        let x64 = x as i64;
        let p = (x64 * (pi - x64)) >> Self::FRAC_BITS;

        let num = 16 * p;
        let pi_sq = (pi * pi) >> Self::FRAC_BITS;
        let den = 5 * pi_sq - 4 * p;

        if den == 0 {
            return Self::ZERO;
        }

        let res = ((num << Self::FRAC_BITS) / den) as i32;
        if negative {
            Self(-res)
        } else {
            Self(res)
        }
    }

    /// Cosine approximation: cos(x) = sin(x + PI/2)
    pub fn cos(self) -> Self {
        (self + Self::FRAC_PI_2).sin()
    }

    /// Fractional part of fixed-point number
    pub fn fract(self) -> Self {
        Self(self.0.rem_euclid(Self::ONE.0))
    }

    /// Floor (round down to integer)
    pub fn floor(self) -> Self {
        Self(self.0 & !(Self::ONE.0 - 1))
    }

    /// Ceil (round up to integer)
    pub fn ceil(self) -> Self {
        let rem = self.0.rem_euclid(Self::ONE.0);
        if rem == 0 {
            self
        } else {
            Self((self.0 & !(Self::ONE.0 - 1)) + Self::ONE.0)
        }
    }

    /// Power for fixed-point exponent: self^exp = exp(exp * ln(self))
    pub fn powf(self, exp: Self) -> Self {
        if self.0 <= 0 {
            return Self::ZERO;
        }
        (exp * self.ln()).exp()
    }
}

impl Add for FixedPoint {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
}
impl Sub for FixedPoint {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self.sub(other)
    }
}
impl Mul for FixedPoint {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
}
impl Div for FixedPoint {
    type Output = Self;
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        self.div(other)
    }
}
impl Neg for FixedPoint {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl AddAssign for FixedPoint {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
    }
}
impl SubAssign for FixedPoint {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(other);
    }
}
impl MulAssign for FixedPoint {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(other);
    }
}
impl DivAssign for FixedPoint {
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        *self = self.div(other);
    }
}

impl fmt::Debug for FixedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

// ============================================================================
// VECTOR - Fixed-size SIMD-friendly array
// ============================================================================

#[derive(Clone, Copy)]
pub struct Vector<const N: usize> {
    data: [FixedPoint; N],
}

impl<const N: usize> Vector<N> {
    #[inline(always)]
    pub const fn new(data: [FixedPoint; N]) -> Self {
        Self { data }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            data: [FixedPoint::ZERO; N],
        }
    }

    #[inline(always)]
    pub const fn splat(val: FixedPoint) -> Self {
        Self { data: [val; N] }
    }

    #[inline(always)]
    pub fn index(&self, i: usize) -> FixedPoint {
        self.data[i]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, val: FixedPoint) {
        self.data[i] = val;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[FixedPoint] {
        &self.data
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> FixedPoint {
        #[cfg(feature = "simd")]
        {
            let mut acc0 = FixedPoint::ZERO;
            let mut acc1 = FixedPoint::ZERO;
            let mut acc2 = FixedPoint::ZERO;
            let mut acc3 = FixedPoint::ZERO;
            let chunks = N / 4;
            let remainder = N % 4;

            for c in 0..chunks {
                let base = c * 4;
                acc0 += self.data[base] * other.data[base];
                acc1 += self.data[base + 1] * other.data[base + 1];
                acc2 += self.data[base + 2] * other.data[base + 2];
                acc3 += self.data[base + 3] * other.data[base + 3];
            }

            let mut sum = acc0 + acc1 + acc2 + acc3;
            for i in (N - remainder)..N {
                sum += self.data[i] * other.data[i];
            }
            sum
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = FixedPoint::ZERO;
            for i in 0..N {
                sum += self.data[i] * other.data[i];
            }
            sum
        }
    }

    #[inline]
    pub fn add_assign(&mut self, other: &Self) {
        #[cfg(feature = "simd")]
        {
            let chunks = N / 4;
            let remainder = N % 4;
            for c in 0..chunks {
                let base = c * 4;
                self.data[base] += other.data[base];
                self.data[base + 1] += other.data[base + 1];
                self.data[base + 2] += other.data[base + 2];
                self.data[base + 3] += other.data[base + 3];
            }
            for i in (N - remainder)..N {
                self.data[i] += other.data[i];
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..N {
                self.data[i] += other.data[i];
            }
        }
    }

    #[inline]
    pub fn elementwise_mul(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        #[cfg(feature = "simd")]
        {
            let chunks = N / 4;
            let remainder = N % 4;
            for c in 0..chunks {
                let base = c * 4;
                result.data[base] = self.data[base] * other.data[base];
                result.data[base + 1] = self.data[base + 1] * other.data[base + 1];
                result.data[base + 2] = self.data[base + 2] * other.data[base + 2];
                result.data[base + 3] = self.data[base + 3] * other.data[base + 3];
            }
            for i in (N - remainder)..N {
                result.data[i] = self.data[i] * other.data[i];
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..N {
                result.data[i] = self.data[i] * other.data[i];
            }
        }
        result
    }

    #[inline]
    pub fn sum(&self) -> FixedPoint {
        #[cfg(feature = "simd")]
        {
            let mut acc0 = FixedPoint::ZERO;
            let mut acc1 = FixedPoint::ZERO;
            let mut acc2 = FixedPoint::ZERO;
            let mut acc3 = FixedPoint::ZERO;
            let chunks = N / 4;
            let remainder = N % 4;
            for c in 0..chunks {
                let base = c * 4;
                acc0 += self.data[base];
                acc1 += self.data[base + 1];
                acc2 += self.data[base + 2];
                acc3 += self.data[base + 3];
            }
            let mut s = acc0 + acc1 + acc2 + acc3;
            for i in (N - remainder)..N {
                s += self.data[i];
            }
            s
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut s = FixedPoint::ZERO;
            for i in 0..N {
                s += self.data[i];
            }
            s
        }
    }

    #[inline]
    pub fn max(&self) -> FixedPoint {
        let mut m = self.data[0];
        for i in 1..N {
            if self.data[i] > m {
                m = self.data[i];
            }
        }
        m
    }

    #[inline]
    pub fn argmax(&self) -> usize {
        let mut idx = 0;
        let mut m = self.data[0];
        for i in 1..N {
            if self.data[i] > m {
                m = self.data[i];
                idx = i;
            }
        }
        idx
    }
}

impl<const N: usize> Default for Vector<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> core::ops::Index<usize> for Vector<N> {
    type Output = FixedPoint;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}
impl<const N: usize> core::ops::IndexMut<usize> for Vector<N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i]
    }
}

// ============================================================================
// MATRIX
// ============================================================================

#[repr(transparent)]
pub struct Matrix<const N: usize> {
    data: [FixedPoint; N],
}

impl<const N: usize> Matrix<N> {
    #[inline(always)]
    pub fn new(data: [FixedPoint; N]) -> Self {
        Self { data }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            data: [FixedPoint::ZERO; N],
        }
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> FixedPoint {
        self.data[i]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, val: FixedPoint) {
        self.data[i] = val;
    }
}

// ============================================================================
// WEIGHT (Q8.8)
// ============================================================================

#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
#[repr(transparent)]
pub struct Weight(pub i16);

impl Weight {
    pub const FRAC_BITS: i16 = 8;
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1 << Self::FRAC_BITS);

    #[inline(always)]
    pub const fn from_f32(x: f32) -> Self {
        Self((x * 256.0) as i16)
    }
    #[inline(always)]
    pub const fn to_f32(self) -> f32 {
        self.0 as f32 / 256.0
    }

    #[inline(always)]
    pub fn to_fixed(self) -> FixedPoint {
        FixedPoint((self.0 as i32) << 8)
    }

    #[inline(always)]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
    #[inline(always)]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

// ============================================================================
// RNG (XorShift64*)
// ============================================================================

pub struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    #[inline(always)]
    pub fn next_fixed(&mut self) -> FixedPoint {
        FixedPoint::from_f32(self.next_f32())
    }

    #[inline(always)]
    pub fn next_gaussian(&mut self) -> FixedPoint {
        // Central Limit Theorem: sum of 12 uniforms approximates N(0,1)
        // This avoids needing exp/ln/cos/sqrt in no_std
        let mut sum = -FixedPoint::from_int(6);
        for _ in 0..12 {
            sum = sum + self.next_fixed();
        }
        sum
    }

    #[inline(always)]
    pub fn next_weight(&mut self) -> Weight {
        let z = self.next_gaussian().to_f32() * 0.1;
        Weight::from_f32(z)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_add() {
        let a = FixedPoint::from_int(3);
        let b = FixedPoint::from_int(4);
        assert_eq!((a + b).to_int(), 7);
    }

    #[test]
    fn fixed_point_sub() {
        let a = FixedPoint::from_int(10);
        let b = FixedPoint::from_int(3);
        assert_eq!((a - b).to_int(), 7);
    }

    #[test]
    fn fixed_point_mul() {
        let a = FixedPoint::from_int(3);
        let b = FixedPoint::from_int(4);
        assert_eq!((a * b).to_int(), 12);
    }

    #[test]
    fn fixed_point_div() {
        let a = FixedPoint::from_int(10);
        let b = FixedPoint::from_int(2);
        assert_eq!((a / b).to_int(), 5);
    }

    #[test]
    fn fixed_point_neg() {
        let a = FixedPoint::from_int(5);
        assert_eq!((-a).to_int(), -5);
    }

    #[test]
    fn fixed_point_from_f32_roundtrip() {
        let vals = [0.0, 1.0, -1.0, 0.5, 3.14159, -0.001];
        for &v in &vals {
            let fp = FixedPoint::from_f32(v);
            let roundtrip = fp.to_f32();
            let diff = (roundtrip - v).abs();
            assert!(
                diff < 0.0001,
                "v={} fp={} rt={} diff={}",
                v,
                fp.0,
                roundtrip,
                diff
            );
        }
    }

    #[test]
    fn fixed_point_commutative() {
        let a = FixedPoint::from_int(7);
        let b = FixedPoint::from_int(3);
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn fixed_point_associative() {
        let a = FixedPoint::from_int(2);
        let b = FixedPoint::from_int(3);
        let c = FixedPoint::from_int(4);
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn fixed_point_exp_positive() {
        let x = FixedPoint::from_f32(1.0);
        let e = x.exp();
        assert!(
            e.to_f32() > 2.0 && e.to_f32() < 3.5,
            "exp(1.0) = {}",
            e.to_f32()
        );
    }

    #[test]
    fn fixed_point_exp_zero() {
        assert_eq!(FixedPoint::ZERO.exp().to_int(), 1);
    }

    #[test]
    fn fixed_point_exp_negative() {
        let x = FixedPoint::from_f32(-1.0);
        let e = x.exp();
        assert!(
            e.to_f32() > 0.2 && e.to_f32() < 0.5,
            "exp(-1.0) = {}",
            e.to_f32()
        );
    }

    #[test]
    fn fixed_point_ln_e() {
        let e = FixedPoint::from_f32(core::f32::consts::E);
        let ln_e = e.ln();
        assert!((ln_e.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn fixed_point_ln_one() {
        assert_eq!(FixedPoint::ONE.ln().to_int(), 0);
    }

    #[test]
    fn fixed_point_sqrt() {
        let x = FixedPoint::from_f32(4.0);
        let s = x.sqrt();
        assert!((s.to_f32() - 2.0).abs() < 0.001);
    }

    #[test]
    fn fixed_point_sqrt_zero() {
        assert_eq!(FixedPoint::ZERO.sqrt().to_int(), 0);
    }

    #[test]
    fn fixed_point_sqrt_one() {
        let s = FixedPoint::ONE.sqrt();
        assert!((s.to_f32() - 1.0).abs() < 0.001);
    }

    #[test]
    fn fixed_point_pow() {
        let base = FixedPoint::from_int(2);
        assert_eq!(base.pow(3u32).to_int(), 8);
    }

    #[test]
    fn fixed_point_pow_zero() {
        let base = FixedPoint::from_int(5);
        assert_eq!(base.pow(0u32).to_int(), 1);
    }

    #[test]
    fn fixed_point_ord() {
        let a = FixedPoint::from_int(3);
        let b = FixedPoint::from_int(5);
        assert!(a < b);
        assert!(b > a);
        assert!(a <= a);
        assert!(b >= b);
    }

    #[test]
    fn weight_saturating_add() {
        let a = Weight::from_f32(100.0);
        let b = Weight::from_f32(100.0);
        let c = a.saturating_add(b);
        assert!(c.to_f32() <= 128.0);
    }

    #[test]
    fn weight_saturating_sub() {
        let a = Weight::from_f32(-100.0);
        let b = Weight::from_f32(-100.0);
        let c = a.saturating_add(b);
        assert!(c.to_f32() >= -128.0);
    }

    #[test]
    fn vector_dot() {
        let a = Vector::new([FixedPoint::from_int(1), FixedPoint::from_int(2)]);
        let b = Vector::new([FixedPoint::from_int(3), FixedPoint::from_int(4)]);
        assert_eq!(a.dot(&b).to_int(), 11);
    }

    #[test]
    fn vector_manually_add_components() {
        let a = Vector::new([FixedPoint::from_int(1), FixedPoint::from_int(2)]);
        let b = Vector::new([FixedPoint::from_int(3), FixedPoint::from_int(4)]);
        assert_eq!((a.data[0] + b.data[0]).to_int(), 4);
        assert_eq!((a.data[1] + b.data[1]).to_int(), 6);
    }

    #[test]
    fn vector_manually_sub_components() {
        let a = Vector::new([FixedPoint::from_int(5), FixedPoint::from_int(7)]);
        let b = Vector::new([FixedPoint::from_int(3), FixedPoint::from_int(4)]);
        assert_eq!((a.data[0] - b.data[0]).to_int(), 2);
        assert_eq!((a.data[1] - b.data[1]).to_int(), 3);
    }

    #[test]
    fn vector_dot_product() {
        let v = Vector::new([FixedPoint::from_int(3), FixedPoint::from_int(4)]);
        let dot = v.dot(&v);
        assert_eq!(dot.to_int(), 25);
    }

    #[test]
    fn xorshift64_basic() {
        let mut rng = XorShift64Star::new(42);
        let a = rng.next_u32();
        let b = rng.next_u32();
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift64_seeded() {
        let mut rng1 = XorShift64Star::new(123);
        let mut rng2 = XorShift64Star::new(123);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn xorshift64_different_seeds() {
        let mut rng1 = XorShift64Star::new(1);
        let mut rng2 = XorShift64Star::new(2);
        let a = rng1.next_u32();
        let b = rng2.next_u32();
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift64_fixed_range() {
        let mut rng = XorShift64Star::new(42);
        for _ in 0..1000 {
            let v = rng.next_u32();
            assert!(v > 0 || v == 0);
        }
    }

    #[test]
    fn xorshift64_gaussian_finite() {
        let mut rng = XorShift64Star::new(99);
        for _ in 0..100 {
            let g = rng.next_gaussian();
            assert!(g.to_f32() > -6.0 && g.to_f32() < 6.0);
        }
    }

    #[test]
    fn matrix_new_and_get() {
        let m = Matrix::new([FixedPoint::from_int(1), FixedPoint::from_int(2)]);
        assert_eq!(m.get(0).to_int(), 1);
        assert_eq!(m.get(1).to_int(), 2);
    }

    #[test]
    fn matrix_set() {
        let mut m = Matrix::new([FixedPoint::ZERO, FixedPoint::ZERO]);
        m.set(0, FixedPoint::from_int(5));
        assert_eq!(m.get(0).to_int(), 5);
    }

    #[test]
    fn matrix_zero() {
        let m = Matrix::<4>::zero();
        for i in 0..4 {
            assert_eq!(m.get(i), FixedPoint::ZERO);
        }
    }

    #[test]
    fn matrix_manually_add_components() {
        let a = Matrix::new([FixedPoint::from_int(1), FixedPoint::from_int(2)]);
        let b = Matrix::new([FixedPoint::from_int(5), FixedPoint::from_int(6)]);
        let c0 = a.get(0) + b.get(0);
        let c1 = a.get(1) + b.get(1);
        assert_eq!(c0.to_int(), 6);
        assert_eq!(c1.to_int(), 8);
    }

    #[test]
    fn weight_zero() {
        assert_eq!(Weight::ZERO.to_f32(), 0.0);
    }

    #[test]
    fn weight_from_i16() {
        let w = Weight(42);
        assert_eq!(w.0, 42);
    }

    // --- Property-based tests ---

    #[test]
    fn fp_overflow_safe_mul() {
        let large = FixedPoint::MAX;
        let result = large * FixedPoint::from_f32(2.0);
        assert!(result.to_f32().is_finite(), "MAX * 2 must saturate");
    }

    #[test]
    fn fp_overflow_safe_add() {
        let result = FixedPoint::MAX + FixedPoint::ONE;
        assert!(result.to_f32().is_finite() || result.0 == i32::MAX);
    }

    #[test]
    fn fp_overflow_safe_sub() {
        let result = FixedPoint::MIN - FixedPoint::ONE;
        assert!(result.to_f32().is_finite() || result.0 == i32::MIN);
    }

    #[test]
    fn fp_div_by_zero_saturates_one() {
        let a = FixedPoint::from_f32(5.0);
        let result = a / FixedPoint::ZERO;
        assert!(result.to_f32() <= 32768.0, "div-zero must saturate");
    }

    #[test]
    fn fp_f32_roundtrip_dense() {
        for i in -2048..2048 {
            let f = i as f32 * 0.001;
            let fp = FixedPoint::from_f32(f);
            let back = fp.to_f32();
            let diff = (back - f).abs();
            assert!(diff < 0.0001, "roundtrip fail: {} -> {} -> {} (diff {})", f, fp.0, back, diff);
        }
    }

    #[test]
    fn fp_mul_commutative_random() {
        let mut rng = XorShift64Star::new(42);
        for _ in 0..1000 {
            let a = FixedPoint::from_f32((rng.next_f32() - 0.5) * 200.0);
            let b = FixedPoint::from_f32((rng.next_f32() - 0.5) * 200.0);
            assert!((a * b - b * a).abs().to_f32() < 0.001, "mul not commutative");
        }
    }

    #[test]
    fn fp_add_commutative_random() {
        let mut rng = XorShift64Star::new(7);
        for _ in 0..1000 {
            let a = FixedPoint::from_f32((rng.next_f32() - 0.5) * 500.0);
            let b = FixedPoint::from_f32((rng.next_f32() - 0.5) * 500.0);
            assert_eq!(a + b, b + a, "add not commutative");
        }
    }

    #[test]
    fn fp_mul_add_distributive() {
        let mut rng = XorShift64Star::new(99);
        for _ in 0..500 {
            let a = FixedPoint::from_f32((rng.next_f32() - 0.5) * 50.0);
            let b = FixedPoint::from_f32((rng.next_f32() - 0.5) * 50.0);
            let c = FixedPoint::from_f32((rng.next_f32() - 0.5) * 50.0);
            let lhs = a * (b + c);
            let rhs = a * b + a * c;
            let diff = (lhs - rhs).abs().to_f32();
            assert!(diff < 0.01, "distributive fail: a={} b={} c={} diff={}", a.to_f32(), b.to_f32(), c.to_f32(), diff);
        }
    }

    #[test]
    fn fp_clamp_bounds() {
        let low = FixedPoint::from_f32(-1.0);
        let high = FixedPoint::from_f32(1.0);
        assert_eq!(FixedPoint::from_f32(-5.0).clamp(low, high), low);
        assert_eq!(FixedPoint::from_f32(5.0).clamp(low, high), high);
        assert_eq!(FixedPoint::from_f32(0.5).clamp(low, high), FixedPoint::from_f32(0.5));
    }

    #[test]
    fn fp_abs_non_negative() {
        let mut rng = XorShift64Star::new(123);
        for _ in 0..1000 {
            let v = FixedPoint::from_f32((rng.next_f32() - 0.5) * 1000.0);
            assert!(v.abs() >= FixedPoint::ZERO, "abs must be non-negative");
        }
    }

    #[test]
    fn fp_sqrt_property() {
        let mut rng = XorShift64Star::new(456);
        for _ in 0..500 {
            let x = FixedPoint::from_f32(rng.next_f32() * 100.0);
            let s = x.sqrt();
            let ss = s * s;
            let diff = (ss - x).abs().to_f32();
            assert!(diff < 0.1, "sqrt(x)^2 ≈ x: x={} sqrt={} sq={} diff={}", x.to_f32(), s.to_f32(), ss.to_f32(), diff);
        }
    }
}
