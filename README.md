# Math and Numerical Algorithms Library (`rsa.hpp`)

mathematical algorithms and data structures in C/C++. Designed for learning, prototyping, and use in competitive programming. Below is a detailed breakdown of the functionalities and concepts in the file, with explanations.

---

## 1. **Number Theory and Modular Arithmetic**

- **GCD and Extended Euclidean Algorithm**
  - `getGCD(a, b)`: Computes the greatest common divisor of two integers using the classical Euclidean algorithm.
  - `extendedEuclideanAlgorithm(a, m, &x, &y)`: Finds integers `x` and `y` such that `ax + my = gcd(a, m)`. This is useful for solving Diophantine equations and for finding modular inverses.

- **Modular Operations**
  - `mod_pow(base, exp, mod)`: Computes `(base^exp) % mod` efficiently using exponentiation by squaring.
  - `modinv(a, mod)`: Computes the modular inverse of `a` modulo `mod`, assuming `mod` is prime (using Fermat’s Little Theorem).
  - `discrete_log(a, b, p)`: Solves for `x` in `a^x ≡ b (mod p)` using the baby-step giant-step algorithm (useful in cryptography).

- **Primality and Factorization**
  - `is_prime(n)`: Checks if `n` is prime using trial division (fast for small numbers).
  - `miller_rabin(n, k)`: Probabilistic primality test suitable for large numbers, with `k` iterations for accuracy.
  - `primeComputation(threshold, output, max_count)` and `sieveOfEratosthenes(limit, output, max_count)`: Generate lists of prime numbers using naive and sieve methods.

- **Other Functions**
  - `gcd_array(arr, n)`, `lcm(a, b)`, `lcm_array(arr, n)`: Compute the GCD or LCM of arrays of numbers.
  - `areCoprime(a, b)`: Checks if two numbers are coprime (GCD 1).

---

## 2. **Combinatorics**

- **Factorials, Permutations, Combinations**
  - `factorial(n, r)`: Computes `n!/(r!)` (partial factorial).
  - `permutationFunction(n, r, rep)`: Calculates permutations with or without repetition.
  - `combinationFunction(n, r, rep)`: Calculates combinations with or without repetition.
  - `binomial_coefficient(n, k)`: Direct computation of “n choose k”.

---

## 3. **Statistics and Data Analysis**

- **Descriptive Statistics**
  - `mean(data, n)`, `getMean(arr, n)`: Average of an array.
  - `getMedian(arr, n)`: Median value (sorts the array).
  - `getMode(arr, n, H)`: Most frequent value, assuming values in `[0, H]`.
  - `getRange(arr, n)`: Range (max - min) after sorting.

- **Variance, Standard Deviation, Correlation**
  - `variance(data, n)`, `stddev(data, n)`: Compute sample variance and standard deviation.
  - `pearson(x, y, n)`: Pearson correlation coefficient between two arrays.

---

## 4. **Algebra and Polynomials**

- **Roots and Polynomial Manipulation**
  - `rational_root(coeffs, degree)`: Attempts to find a rational root of a polynomial with integer coefficients using the Rational Root Theorem.
  - `quadratic_roots(a, b, c, &x1, &x2)`: Finds real roots of quadratic equations.
  - `cubic_roots(a, b, c, d, roots)`: Finds real roots for cubic polynomials.

- **Polynomial Operations**
  - `synthetic_division(poly, n, root, out, &rem)`: Efficiently divides a polynomial by `(x - root)`.
  - `poly_multiply(a, n, b, m, out)`: Multiplies two polynomials, returning the resulting coefficients.

---

## 5. **Matrix Operations**

- **Matrix Structure**
  - `Matrix`: Struct for a dense 2D matrix in row-major order.
  - Functions for creation, memory management, setting/getting values, and printing.

- **Matrix Arithmetic**
  - `add_matrix(a, b)`, `multiply_matrix(a, b)`: Element-wise addition and matrix multiplication.
  - `transpose_matrix(a)`: Returns the transpose.

- **Determinants, Inverses, and Rank**
  - `matrix_determinant(mat, n)`: Recursive determinant calculation for NxN matrices.
  - `matrix_inverse(mat, inv, n)`: Computes the inverse using Gauss-Jordan elimination.
  - `matrix_minor`/`matrix_cofactor`: Helper functions for determinants and adjugates.
  - `matrix_rank(A, m, n)`: Returns the rank of an m x n matrix using Gaussian elimination.

- **Solving Linear Systems**
  - `solve_linear_system(A, b, x, n)`, `gaussian_elimination(A, b, x, n)`: Solve `Ax = b` for `x` using elimination and back substitution.

---

## 6. **Sets**

- **Set Structure and Operations**
  - `Set`: Struct for an array-based integer set.
  - Functions for creating sets (removing duplicates), printing, union, intersection, difference, and subset checks.

---

## 7. **Sequences and Special Functions**

- **Fibonacci Sequence**
  - `fibonacciSequence(output, n)`: Fills an array with the first `n` Fibonacci numbers.

- **Fast Power**
  - `fast_pow(base, exp)`: Computes `base^exp` efficiently via exponentiation by squaring (integer version).

- **Special Functions**
  - `gamma_lanczos(z)`: Approximates the gamma function (generalizes factorial).
  - `erf_approx(x)`: Error function, widely used in statistics.
  - `bessel_j0(x)`: Bessel function of the first kind, order zero.
  - `elliptic_k(k)`, `elliptic_e(k)`: Complete elliptic integrals of the first and second kind.

---

## 8. **FFT and Complex Numbers**

- **Complex Number Structure**
  - `cplx`: Simple struct for real/imaginary parts.
- **FFT**
  - `fft(a, n, invert)`: In-place Fast Fourier Transform for polynomial multiplication and signal processing.

---

## 9. **Elliptic Curve Arithmetic**

- **ECPoint Structure**
  - `ECPoint`: Represents a point on an elliptic curve (for cryptographic use).
- **Point Addition**
  - `ec_add(P, Q, a, p)`: Adds two points on an elliptic curve modulo `p`.

---

## 10. **Numerical Methods**

- **Adaptive Simpson’s Rule**
  - `adaptive_simpson(f, a, b, eps, maxdepth)`: Numerically integrates a real function `f` over `[a, b]` with adaptive precision.

- **Power Iteration**
  - `power_iteration(A, n, eigenvector, max_iter, tol)`: Approximates the dominant eigenvalue and corresponding eigenvector of a matrix.

---

## 11. **Utility Functions**

- **isEven(a) / isOdd(a)**: Integer parity check.
- **isqrt(n)**: Integer square root using binary search.

---

## 12. **Demonstrations and Usage**

The `main()` function demonstrates nearly every feature, printing results for typical inputs. This is a useful reference for how to call each function and what output to expect.

---

### **How to Use**

- **Include** `rsa.hpp` in your project.
- Call the required functions directly. Many require the user to allocate output arrays.
- Functions use standard C types and require linking against the standard math library (`-lm` for gcc/clang).

---

### **Applications**

- **Learning**: Algorithms are implemented from scratch and are easy to read for educational purposes.
- **Prototyping**: Rapidly try out mathematical or algorithmic ideas.
- **Competitive Programming**: Handy for contests where re-implementing these utilities is time-consuming.
