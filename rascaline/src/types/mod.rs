//! This module provides 3D vectors and matrix to be used in all other modules.
/// Implement $Lhs -- $Rhs arithmetic operations for all variation of by
/// value, by reference and by mutable reference of $Rhs and $Lhs.
macro_rules! impl_arithmetic {
    ($Lhs:ty, $Rhs:ty, $Op:ident, $op:ident, $Output:ty, $sel:ident, $other:ident, $res:expr) => (
        impl $Op<$Rhs> for $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: $Rhs) -> $Output {
                $res
            }
        }

        impl<'a> $Op<$Rhs> for &'a $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: $Rhs) -> $Output {
                $res
            }
        }

        impl<'a> $Op<&'a $Rhs> for $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a $Rhs) -> $Output {
                $res
            }
        }

        impl<'a, 'b> $Op<&'a $Rhs> for &'b $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a $Rhs) -> $Output {
                $res
            }
        }

        impl<'a, 'b> $Op<&'a mut $Rhs> for &'b mut $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a mut $Rhs) -> $Output {
                $res
            }
        }

        impl<'a, 'b> $Op<&'a mut $Rhs> for &'b $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a mut $Rhs) -> $Output {
                $res
            }
        }

        impl<'a, 'b> $Op<&'a $Rhs> for &'b mut $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a $Rhs) -> $Output {
                $res
            }
        }

        impl<'a> $Op<&'a mut $Rhs> for $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a mut $Rhs) -> $Output {
                $res
            }
        }

        impl<'a> $Op<$Rhs> for &'a mut $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: $Rhs) -> $Output {
                $res
            }
        }
    );
}

/// Implement operators `@=` for all variations of references for the right-hand
/// side.
macro_rules! impl_inplace_arithmetic {
    ($Lhs:ty, $Rhs:ty, $Op:ident, $op:ident, $sel:ident, $other:ident, $res:expr) => (
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a> $Op<$Rhs> for $Lhs {
            #[inline] fn $op(&mut $sel, $other: $Rhs) {
                $res
            }
        }

        impl<'a> $Op<&'a $Rhs> for $Lhs {
            #[inline] fn $op(&mut $sel, $other: &'a $Rhs) {
                $res
            }
        }

        impl<'a> $Op<&'a mut $Rhs> for $Lhs {
            #[inline] fn $op(&mut $sel, $other: &'a mut $Rhs) {
                $res
            }
        }
    )
}

/// Implement $Lhs -- scalar arithmetic operations for all variation of by
/// value, by reference and by mutable reference $Lhs.
macro_rules! lsh_scal_arithmetic {
    ($Lhs: ty, $Op:ident, $op:ident, $Output:ty, $sel:ident, $other:ident, $res:expr) => (
        impl $Op<f64> for $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: f64) -> $Output {
                $res
            }
        }

        impl<'a> $Op<f64> for &'a $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: f64) -> $Output {
                $res
            }
        }

        impl<'a> $Op<f64> for &'a mut $Lhs {
            type Output = $Output;
            #[inline] fn $op($sel, $other: f64) -> $Output {
                $res
            }
        }
    );
}

/// Implement scalar -- $Rhs arithmetic operations for all variation of by
/// value, by reference and by mutable reference of $Rhs.
macro_rules! rhs_scal_arithmetic {
    ($Rhs:ty, $Op:ident, $op:ident, $Output:ty, $sel:ident, $other:ident, $res:expr) => (
        impl $Op<$Rhs> for f64 {
            type Output = $Output;
            #[inline] fn $op($sel, $other: $Rhs) -> $Output {
                $res
            }
        }

        impl<'a> $Op<&'a $Rhs> for f64 {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a $Rhs) -> $Output {
                $res
            }
        }

        impl<'a> $Op<&'a mut $Rhs> for f64 {
            type Output = $Output;
            #[inline] fn $op($sel, $other: &'a mut $Rhs) -> $Output {
                $res
            }
        }
    );
}

mod vectors;
pub use self::vectors::Vector3D;

mod matrix;
pub use self::matrix::Matrix3;
