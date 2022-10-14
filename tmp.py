import mpmath
import scipy.special
import hyp

a = 0.0
b = 1.0
c = 0.0
x = 0.9

print("mpmath:", mpmath.hyp2f1(a, b, c, x))
print("scipy:", scipy.special.hyp2f1(a, b, c, x))
print("hyp:", hyp.hyp(x, a, b, c))
