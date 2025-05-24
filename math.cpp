#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to get modulo class of n with respect to m
int getModuloClass(int n, int m)
{
    if (n > 0 && m > 0)
    {
        return n % m;
    }
    return 0;
}

double mean(const double *data, int n)
{
    double sum = 0;
    for (int i = 0; i < n; ++i)
        sum += data[i];
    return sum / n;
}

int rational_root(const int *coeffs, int degree)
{
    int a0 = abs(coeffs[degree]), an = abs(coeffs[0]);
    for (int p = 1; p <= a0; ++p)
        if (a0 % p == 0)
        {
            for (int q = 1; q <= an; ++q)
                if (an % q == 0)
                {
                    for (int sign = -1; sign <= 1; sign += 2)
                    {
                        int root = sign * p / q, sum = 0, powx = 1;
                        for (int i = degree; i >= 0; --i)
                        {
                            sum += coeffs[i] * powx;
                            powx *= root;
                        }
                        if (sum == 0)
                            return root;
                    }
                }
        }
    return 0; // not found
}

// Function to compute GCD (Euclidean algorithm)
int getGCD(int a, int b)
{
    int dvd = a;
    int dvs = b;
    int rem = 0;
    while (dvs != 0)
    {
        rem = dvd % dvs;
        dvd = dvs;
        dvs = rem;
    }
    return dvd;
}

// Extended Euclidean Algorithm: computes GCD and coefficients x, y
int extendedEuclideanAlgorithm(int a, int m, int *x, int *y)
{
    int x1 = 0, y1 = 1;
    *x = 1;
    *y = 0;
    while (m != 0)
    {
        int q = a / m;
        int temp = a % m;
        a = m;
        m = temp;
        temp = *x;
        *x = x1;
        x1 = temp - q * x1;
        temp = *y;
        *y = y1;
        y1 = temp - q * y1;
    }
    return a;
}

// Returns 1 if a and b are coprime, 0 otherwise
int areCoprime(int a, int b)
{
    return getGCD(a, b) == 1;
}

// Prime computation (returns the number of found primes, primes written to output array)
int primeComputation(int threshold, int *output, int max_count)
{
    int k = 0;
    if (max_count > 0)
        output[k++] = 2;
    for (int i = 3; i <= threshold; i += 2)
    {
        if (k >= max_count)
            break;
        int p = 1;
        for (int j = 3; j <= (int)sqrt(i); j += 2)
        {
            if (i % j == 0)
            {
                p = 0;
                break;
            }
        }
        if (p)
        {
            output[k++] = i;
        }
    }
    return k;
}

// Sieve of Eratosthenes: writes found primes into output, returns count
int sieveOfEratosthenes(int limit, int *output, int max_count)
{
    char *isPrime = (char *)malloc((limit + 1) * sizeof(char));
    memset(isPrime, 1, (limit + 1) * sizeof(char));
    int k = 0;
    isPrime[0] = isPrime[1] = 0;
    for (int p = 2; p * p <= limit; ++p)
    {
        if (isPrime[p])
        {
            for (int i = p * p; i <= limit; i += p)
                isPrime[i] = 0;
        }
    }
    for (int p = 2; p <= limit; ++p)
    {
        if (isPrime[p] && k < max_count)
        {
            output[k++] = p;
        }
    }
    free(isPrime);
    return k;
}

// Raise b to the exponent e
int raise(int b, int e)
{
    int result = b;
    int c = 0;
    while (++c < e)
    {
        result *= b;
    }
    return result;
}

// Fibonacci sequence: fills output array, returns count
void fibonacciSequence(int *output, int n)
{
    if (n > 0)
        output[0] = 0;
    if (n > 1)
        output[1] = 1;
    for (int i = 2; i < n; ++i)
    {
        output[i] = output[i - 1] + output[i - 2];
    }
}

// Returns 1 if a is even, 0 otherwise
int isEven(int a)
{
    return a % 2 == 0;
}

// Returns 1 if a is odd, 0 otherwise
int isOdd(int a)
{
    return a % 2 != 0;
}

// Get mode of an array (assumes elements in 0..H), returns the mode value
int getMode(const int *arr, int n, int H)
{
    int *frequency = (int *)calloc(H + 1, sizeof(int));
    for (int i = 0; i < n; ++i)
    {
        frequency[arr[i]]++;
    }
    int mode = 0;
    for (int i = 1; i <= H; ++i)
    {
        if (frequency[i] > frequency[mode])
            mode = i;
    }
    free(frequency);
    return mode;
}

// Median of array (returns int, assumes array is copied and sorted)
int getMedian(int *arr, int n)
{
    // Sort the array
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (arr[i] > arr[j])
            {
                int tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
    }
    if (n % 2 == 0)
        return (arr[n / 2 - 1] + arr[n / 2]) / 2;
    else
        return arr[n / 2];
}

// Mean of array
int getMean(const int *arr, int n)
{
    int sum = 0;
    for (int i = 0; i < n; ++i)
        sum += arr[i];
    return sum / n;
}

// Range of array
int getRange(int *arr, int n)
{
    // Sort the array
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (arr[i] > arr[j])
            {
                int tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
    }
    return arr[n - 1] - arr[0];
}

// Factorial n down to r (n!/(r!)), if r>=n or n==0 returns 0
size_t factorial(size_t n, size_t r)
{
    if (n == 0 || r >= n)
        return 0;
    size_t result = n;
    while (--n > r)
    {
        result *= n;
    }
    return result;
}

// Permutation function: if rep is true, n^r; else, n!/(n-r)!
size_t permutationFunction(size_t n, size_t r, int rep)
{
    if (rep)
        return raise(n, r);
    else
        return factorial(n, n - r);
}

// Combination function: if rep is true, (n+r-1)!/(r!*(n-1)!); else, n!/(r!*(n-r)!)
size_t combinationFunction(size_t n, size_t r, int rep)
{
    if (n == 0 || r == 0)
        return 0;
    size_t num = rep ? factorial(n + r - 1, 1) : factorial(n, n - r);
    size_t denom = factorial(r, 1) * (rep ? factorial(n - 1, 1) : 1);
    return num / denom;
}

// Matrix type
typedef struct
{
    int rows;
    int cols;
    double *data; // Row-major order
} Matrix;

// Create a matrix
Matrix create_matrix(int rows, int cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double *)calloc(rows * cols, sizeof(double));
    return m;
}

// Free matrix memory
void free_matrix(Matrix *m)
{
    free(m->data);
    m->data = NULL;
}

// Set value at (i, j)
void set_matrix(Matrix *m, int i, int j, double val)
{
    m->data[i * m->cols + j] = val;
}

// Get value at (i, j)
double get_matrix(const Matrix *m, int i, int j)
{
    return m->data[i * m->cols + j];
}

// Print matrix
void print_matrix(const Matrix *m)
{
    for (int i = 0; i < m->rows; ++i)
    {
        for (int j = 0; j < m->cols; ++j)
        {
            printf("%8.2f ", get_matrix(m, i, j));
        }
        printf("\n");
    }
}

// Matrix addition
Matrix add_matrix(const Matrix *a, const Matrix *b)
{
    Matrix m = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; ++i)
        for (int j = 0; j < a->cols; ++j)
            set_matrix(&m, i, j, get_matrix(a, i, j) + get_matrix(b, i, j));
    return m;
}

// Matrix multiplication
Matrix multiply_matrix(const Matrix *a, const Matrix *b)
{
    Matrix m = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; ++i)
    {
        for (int j = 0; j < b->cols; ++j)
        {
            double sum = 0;
            for (int k = 0; k < a->cols; ++k)
            {
                sum += get_matrix(a, i, k) * get_matrix(b, k, j);
            }
            set_matrix(&m, i, j, sum);
        }
    }
    return m;
}

// Transpose matrix
Matrix transpose_matrix(const Matrix *a)
{
    Matrix m = create_matrix(a->cols, a->rows);
    for (int i = 0; i < a->rows; ++i)
        for (int j = 0; j < a->cols; ++j)
            set_matrix(&m, j, i, get_matrix(a, i, j));
    return m;
}

// Set type (for integers, max size N)
typedef struct
{
    int *data;
    int size;
} Set;

// Create a set from array (removes duplicates)
Set create_set(const int *arr, int n)
{
    Set s;
    s.data = (int *)malloc(n * sizeof(int));
    s.size = 0;
    for (int i = 0; i < n; ++i)
    {
        int found = 0;
        for (int j = 0; j < s.size; ++j)
        {
            if (arr[i] == s.data[j])
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            s.data[s.size++] = arr[i];
        }
    }
    return s;
}

// Free set
void free_set(Set *s)
{
    free(s->data);
    s->data = NULL;
    s->size = 0;
}

// Print set
void print_set(const Set *s)
{
    printf("{ ");
    for (int i = 0; i < s->size; ++i)
        printf("%d ", s->data[i]);
    printf("}\n");
}

// Set union
Set set_union(const Set *a, const Set *b)
{
    int *tmp = (int *)malloc((a->size + b->size) * sizeof(int));
    memcpy(tmp, a->data, a->size * sizeof(int));
    int k = a->size;
    for (int i = 0; i < b->size; ++i)
    {
        int found = 0;
        for (int j = 0; j < a->size; ++j)
            if (b->data[i] == a->data[j])
                found = 1;
        if (!found)
            tmp[k++] = b->data[i];
    }
    Set s = create_set(tmp, k);
    free(tmp);
    return s;
}

// Set intersection
Set set_intersection(const Set *a, const Set *b)
{
    int *tmp = (int *)malloc((a->size < b->size ? a->size : b->size) * sizeof(int));
    int k = 0;
    for (int i = 0; i < a->size; ++i)
    {
        for (int j = 0; j < b->size; ++j)
        {
            if (a->data[i] == b->data[j])
            {
                tmp[k++] = a->data[i];
                break;
            }
        }
    }
    Set s = create_set(tmp, k);
    free(tmp);
    return s;
}

// Set difference (A - B)
Set set_difference(const Set *a, const Set *b)
{
    int *tmp = (int *)malloc(a->size * sizeof(int));
    int k = 0;
    for (int i = 0; i < a->size; ++i)
    {
        int found = 0;
        for (int j = 0; j < b->size; ++j)
            if (a->data[i] == b->data[j])
                found = 1;
        if (!found)
            tmp[k++] = a->data[i];
    }
    Set s = create_set(tmp, k);
    free(tmp);
    return s;
}

// Set subset (returns 1 if a is subset of b)
int set_subset(const Set *a, const Set *b)
{
    for (int i = 0; i < a->size; ++i)
    {
        int found = 0;
        for (int j = 0; j < b->size; ++j)
            if (a->data[i] == b->data[j])
                found = 1;
        if (!found)
            return 0;
    }
    return 1;
}

// --- Binomial Coefficient (n choose k) ---
unsigned long long binomial_coefficient(int n, int k)
{
    if (k < 0 || n < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    if (k > n - k)
        k = n - k;
    unsigned long long res = 1;
    for (int i = 1; i <= k; ++i)
    {
        res = res * (n - i + 1) / i;
    }
    return res;
}

// --- Modular Exponentiation ---
int mod_pow(int base, int exp, int mod)
{
    int result = 1;
    base %= mod;
    while (exp > 0)
    {
        if (exp & 1)
            result = (long long)result * base % mod;
        base = (long long)base * base % mod;
        exp >>= 1;
    }
    return result;
}

// --- Modular Inverse (for prime mod only) ---
int modinv(int a, int mod)
{
    // Fermat's little theorem
    return mod_pow(a, mod - 2, mod);
}

// --- Matrix Determinant (recursive, NxN) ---
double matrix_determinant(double **mat, int n)
{
    if (n == 1)
        return mat[0][0];
    if (n == 2)
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    double det = 0.0;
    for (int p = 0; p < n; ++p)
    {
        double **submat = (double **)malloc((n - 1) * sizeof(double *));
        for (int i = 0; i < n - 1; ++i)
            submat[i] = (double *)malloc((n - 1) * sizeof(double));
        for (int i = 1; i < n; ++i)
        {
            int colIdx = 0;
            for (int j = 0; j < n; ++j)
            {
                if (j == p)
                    continue;
                submat[i - 1][colIdx++] = mat[i][j];
            }
        }
        det += (p % 2 == 0 ? 1 : -1) * mat[0][p] * matrix_determinant(submat, n - 1);
        for (int i = 0; i < n - 1; ++i)
            free(submat[i]);
        free(submat);
    }
    return det;
}

// --- Matrix Inverse (Gauss-Jordan, NxN) ---
int matrix_inverse(double **mat, double **inv, int n)
{
    // Copy mat to augmented matrix
    double **aug = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; ++i)
    {
        aug[i] = (double *)malloc(2 * n * sizeof(double));
        for (int j = 0; j < n; ++j)
            aug[i][j] = mat[i][j];
        for (int j = n; j < 2 * n; ++j)
            aug[i][j] = (i == j - n) ? 1.0 : 0.0;
    }
    // Gauss-Jordan elimination
    for (int i = 0; i < n; ++i)
    {
        double pivot = aug[i][i];
        if (fabs(pivot) < 1e-12)
        { // Singular
            for (int x = 0; x < n; ++x)
                free(aug[x]);
            free(aug);
            return 0;
        }
        for (int j = 0; j < 2 * n; ++j)
            aug[i][j] /= pivot;
        for (int k = 0; k < n; ++k)
        {
            if (k == i)
                continue;
            double factor = aug[k][i];
            for (int j = 0; j < 2 * n; ++j)
                aug[k][j] -= factor * aug[i][j];
        }
    }
    // Extract inverse
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            inv[i][j] = aug[i][j + n];
    for (int x = 0; x < n; ++x)
        free(aug[x]);
    free(aug);
    return 1;
}

// --- Matrix Minor (for element (row,col)) ---
double matrix_minor(double **mat, int n, int row, int col)
{
    double **submat = (double **)malloc((n - 1) * sizeof(double *));
    for (int i = 0; i < n - 1; ++i)
        submat[i] = (double *)malloc((n - 1) * sizeof(double));
    int subi = 0;
    for (int i = 0; i < n; ++i)
    {
        if (i == row)
            continue;
        int subj = 0;
        for (int j = 0; j < n; ++j)
        {
            if (j == col)
                continue;
            submat[subi][subj++] = mat[i][j];
        }
        ++subi;
    }
    double det = matrix_determinant(submat, n - 1);
    for (int i = 0; i < n - 1; ++i)
        free(submat[i]);
    free(submat);
    return det;
}

// --- Matrix Cofactor (for element (row,col)) ---
double matrix_cofactor(double **mat, int n, int row, int col)
{
    double minor = matrix_minor(mat, n, row, col);
    return ((row + col) % 2 == 0 ? 1 : -1) * minor;
}

// --- Integer Square Root (binary search) ---
int isqrt(int n)
{
    if (n < 0)
        return -1;
    int left = 0, right = n, ans = 0;
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (mid * mid <= n)
        {
            ans = mid;
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    return ans;
}

// --- Fast Power (exponentiation by squaring) ---
long long fast_pow(long long base, long long exp)
{
    long long res = 1;
    while (exp > 0)
    {
        if (exp & 1)
            res *= base;
        base *= base;
        exp >>= 1;
    }
    return res;
}

// --- Check if prime (deterministic for n < 2^32) ---
int is_prime(int n)
{
    if (n <= 1)
        return 0;
    if (n <= 3)
        return 1;
    if (n % 2 == 0 || n % 3 == 0)
        return 0;
    for (int i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    return 1;
}

// --- GCD of an array ---
int gcd_array(const int *arr, int n)
{
    int res = arr[0];
    for (int i = 1; i < n; ++i)
    {
        int a = res, b = arr[i];
        while (b)
        {
            int t = a % b;
            a = b;
            b = t;
        }
        res = a;
    }
    return res;
}

// --- LCM of two numbers ---
int lcm(int a, int b)
{
    int g = a, h = b;
    while (h)
    {
        int t = g % h;
        g = h;
        h = t;
    }
    return a / g * b;
}

// --- LCM of an array ---
int lcm_array(const int *arr, int n)
{
    int res = arr[0];
    for (int i = 1; i < n; ++i)
        res = lcm(res, arr[i]);
    return res;
}

#define PI 3.14159265358979323846

typedef struct
{
    double re, im;
} cplx;

void fft(cplx *a, int n, int invert)
{
    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < n; ++i)
    {
        int bit = n >> 1;
        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
        {
            cplx tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    // FFT
    for (int len = 2; len <= n; len <<= 1)
    {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cplx wlen = {cos(ang), sin(ang)};
        for (int i = 0; i < n; i += len)
        {
            cplx w = {1, 0};
            for (int j = 0; j < len / 2; ++j)
            {
                cplx u = a[i + j];
                cplx v = {a[i + j + len / 2].re * w.re - a[i + j + len / 2].im * w.im, a[i + j + len / 2].re * w.im + a[i + j + len / 2].im * w.re};
                a[i + j].re = u.re + v.re;
                a[i + j].im = u.im + v.im;
                a[i + j + len / 2].re = u.re - v.re;
                a[i + j + len / 2].im = u.im - v.im;
                double wr = w.re * wlen.re - w.im * wlen.im;
                w.im = w.re * wlen.im + w.im * wlen.re;
                w.re = wr;
            }
        }
    }
    // If inverse, divide by n
    if (invert)
        for (int i = 0; i < n; ++i)
        {
            a[i].re /= n;
            a[i].im /= n;
        }
}

// Multiply matrix A (n x n) with vector v (length n): result in out (length n)
void matvec_mul(double **A, double *v, double *out, int n)
{
    for (int i = 0; i < n; ++i)
    {
        out[i] = 0;
        for (int j = 0; j < n; ++j)
            out[i] += A[i][j] * v[j];
    }
}
// Returns dominant eigenvalue and fills eigenvector
double power_iteration(double **A, int n, double *eigenvector, int max_iter, double tol)
{
    double *b = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
        eigenvector[i] = 1.0;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        matvec_mul(A, eigenvector, b, n);
        double norm = 0;
        for (int i = 0; i < n; ++i)
            norm += b[i] * b[i];
        norm = sqrt(norm);
        for (int i = 0; i < n; ++i)
            b[i] /= norm;
        double diff = 0;
        for (int i = 0; i < n; ++i)
            diff += fabs(eigenvector[i] - b[i]);
        if (diff < tol)
            break;
        for (int i = 0; i < n; ++i)
            eigenvector[i] = b[i];
    }
    double *tmp = (double *)malloc(n * sizeof(double));
    matvec_mul(A, eigenvector, tmp, n);
    double lambda = 0;
    for (int i = 0; i < n; ++i)
        lambda += tmp[i] * eigenvector[i];
    free(b);
    free(tmp);
    return lambda;
}

// Computes (base^exp) % mod efficiently for 64-bit unsigned
uint64_t modpow64(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t result = 1;
    base %= mod;
    while (exp > 0)
    {
        if (exp & 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

// y^2 = x^3 + ax + b mod p
typedef struct
{
    uint64_t x, y;
    int is_inf;
} ECPoint;

ECPoint ec_add(ECPoint P, ECPoint Q, uint64_t a, uint64_t p)
{
    if (P.is_inf)
        return Q;
    if (Q.is_inf)
        return P;
    if (P.x == Q.x && (P.y + Q.y) % p == 0)
    {
        ECPoint inf = {0, 0, 1};
        return inf;
    }
    uint64_t m;
    if (P.x == Q.x && P.y == Q.y)
    { // doubling
        uint64_t num = (3 * P.x % p * P.x % p + a) % p;
        uint64_t den = (2 * P.y) % p;
        m = (num * modinv(den, p)) % p;
    }
    else
    { // addition
        uint64_t num = (Q.y + p - P.y) % p;
        uint64_t den = (Q.x + p - P.x) % p;
        m = (num * modinv(den, p)) % p;
    }
    uint64_t xr = (m * m % p + p - P.x + p - Q.x) % p;
    uint64_t yr = (m * (P.x + p - xr) % p + p - P.y) % p;
    ECPoint R = {xr, yr, 0};
    return R;
}

// Helper recursive function with function pointer as first argument
double recur(double (*f)(double), int maxdepth, double eps, double l, double r, double fl, double fr, double fm, double approx, int depth)
{
    double m = (l + r) / 2;
    double fml = f((l + m) / 2), fmr = f((m + r) / 2);
    double left = (fl + 4 * fml + fm) * (m - l) / 6;
    double right = (fm + 4 * fmr + fr) * (r - m) / 6;
    if (depth > maxdepth || fabs(left + right - approx) < 15 * eps)
        return left + right + (left + right - approx) / 15;
    return recur(f, maxdepth, eps, l, m, fl, fm, fml, left, depth + 1) + recur(f, maxdepth, eps, m, r, fm, fr, fmr, right, depth + 1);
}

// Adaptive Simpson's Rule function
double adaptive_simpson(double (*f)(double), double a, double b, double eps, int maxdepth)
{
    double c = (a + b) / 2;
    double fa = f(a), fb = f(b), fc = f(c);
    double simpson = (fa + 4 * fc + fb) * (b - a) / 6;

    return recur(f, maxdepth, eps, a, b, fa, fb, fc, simpson, 0);
}

int solve_linear_system(double **A, double *b, double *x, int n)
{
    // A: n x n matrix, b: n vector, x: solution (output)
    for (int i = 0; i < n; ++i)
    {
        // Partial pivot
        int maxrow = i;
        for (int j = i + 1; j < n; ++j)
            if (fabs(A[j][i]) > fabs(A[maxrow][i]))
                maxrow = j;
        if (fabs(A[maxrow][i]) < 1e-12)
            return 0;
        // Swap rows
        if (maxrow != i)
        {
            double *tmp = A[i];
            A[i] = A[maxrow];
            A[maxrow] = tmp;
            double tb = b[i];
            b[i] = b[maxrow];
            b[maxrow] = tb;
        }
        // Elimination
        for (int j = i + 1; j < n; ++j)
        {
            double f = A[j][i] / A[i][i];
            for (int k = i; k < n; ++k)
                A[j][k] -= f * A[i][k];
            b[j] -= f * b[i];
        }
    }
    // Back substitution
    for (int i = n - 1; i >= 0; --i)
    {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
    return 1;
}

// Helper: modular exponentiation for 64-bit
uint64_t powmod(uint64_t a, uint64_t b, uint64_t m)
{
    uint64_t res = 1;
    a %= m;
    while (b > 0)
    {
        if (b & 1)
            res = (res * a) % m;
        a = (a * a) % m;
        b >>= 1;
    }
    return res;
}

// Miller-Rabin test with k iterations
int miller_rabin(uint64_t n, int k)
{
    if (n < 2)
        return 0;
    if (n == 2 || n == 3)
        return 1;
    if (n % 2 == 0)
        return 0;
    uint64_t d = n - 1;
    int s = 0;
    while (d % 2 == 0)
    {
        d /= 2;
        ++s;
    }
    for (int i = 0; i < k; ++i)
    {
        uint64_t a = 2 + rand() % (n - 3);
        uint64_t x = powmod(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        int c = 0;
        for (; c < s - 1; ++c)
        {
            x = powmod(x, 2, n);
            if (x == n - 1)
                break;
        }
        if (c == s - 1)
            return 0;
    }
    return 1;
}

// Variance (sample)
double variance(const double *data, int n)
{
    double m = mean(data, n), v = 0.0;
    for (int i = 0; i < n; ++i)
        v += (data[i] - m) * (data[i] - m);
    return v / (n - 1);
}

// Standard deviation (sample)
double stddev(const double *data, int n)
{
    return sqrt(variance(data, n));
}

// Pearson correlation coefficient
double pearson(const double *x, const double *y, int n)
{
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; ++i)
    {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        syy += y[i] * y[i];
        sxy += x[i] * y[i];
    }
    double num = n * sxy - sx * sy;
    double den = sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    return num / den;
}

// Lanczos approximation for the gamma function
double gamma_lanczos(double z)
{
    static double p[] = {676.5203681218851,  -1259.1392167224028,  771.32342877765313,    -176.61502916214059,
                         12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
    if (z < 0.5)
        return M_PI / (sin(M_PI * z) * gamma_lanczos(1 - z));
    z -= 1;
    double x = 0.99999999999980993;
    for (int i = 0; i < 8; i++)
        x += p[i] / (z + i + 1);
    double t = z + 7.5;
    return sqrt(2 * M_PI) * pow(t, z + 0.5) * exp(-t) * x;
}

// Error function approximation (erf)
double erf_approx(double x)
{
    // constants
    double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    double a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    int sign = (x >= 0) ? 1 : -1;
    x = fabs(x);
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return sign * y;
}

// Bessel function J0 (order 0), series expansion
double bessel_j0(double x)
{
    double sum = 1.0, term = 1.0, xx = (x / 2) * (x / 2);
    for (int k = 1; k < 20; ++k)
    {
        term *= -xx / (k * k);
        sum += term;
        if (fabs(term) < 1e-15)
            break;
    }
    return sum;
}

// Complete Elliptic Integral of the First Kind (K)
// K(k) = ∫₀^{π/2} 1 / sqrt(1 - k^2 * sin^2 θ) dθ
double elliptic_k(double k)
{
    if (k < 0 || k >= 1)
        return NAN;
    double a = 1.0, b = sqrt(1.0 - k * k), c, sum = 0.0;
    int n = 0;
    while (fabs(a - b) > 1e-15 && n++ < 100)
    {
        c = (a + b) / 2.0;
        b = sqrt(a * b);
        a = c;
    }
    return M_PI / (2.0 * a);
}

// Complete Elliptic Integral of the Second Kind (E)
// E(k) = ∫₀^{π/2} sqrt(1 - k^2 * sin^2 θ) dθ
double elliptic_e(double k)
{
    if (k < 0 || k >= 1)
        return NAN;
    double a = 1.0, b = sqrt(1.0 - k * k), c, sum = 0.0, pow2 = 1.0;
    int n = 0;
    while (fabs(a - b) > 1e-15 && n++ < 100)
    {
        c = (a + b) / 2.0;
        double temp = a - b;
        sum += pow2 * temp * temp;
        pow2 *= 2.0;
        b = sqrt(a * b);
        a = c;
    }
    return M_PI / (2.0 * a) * (1.0 - sum / 2.0);
}

// Find x such that a^x ≡ b (mod p) using baby-step giant-step
// Returns x if found, or -1 if not found
int64_t discrete_log(uint64_t a, uint64_t b, uint64_t p)
{
    a %= p;
    b %= p;
    uint64_t m = (uint64_t)sqrt(p) + 1;

    // Baby step
    uint64_t *table = (uint64_t *)malloc(m * sizeof(uint64_t));
    for (uint64_t j = 0, cur = 1; j < m; ++j)
    {
        table[j] = cur;
        cur = (cur * a) % p;
    }

    // Giant step
    uint64_t am = powmod(a, m * (p - 2), p); // a^-m mod p
    uint64_t cur = b;
    for (uint64_t i = 0; i < m; ++i)
    {
        for (uint64_t j = 0; j < m; ++j)
        {
            if (table[j] == cur)
            {
                free(table);
                return i * m + j;
            }
        }
        cur = (cur * am) % p;
    }
    free(table);
    return -1; // No solution found
}

// Divides poly[] by (x - root). poly has degree n, returns quotient in out[] and remainder.
void synthetic_division(const double *poly, int n, double root, double *out, double *remainder)
{
    out[0] = poly[0];
    for (int i = 1; i < n; ++i)
    {
        out[i] = poly[i] + root * out[i - 1];
    }
    *remainder = poly[n] + root * out[n - 1];
}

// Multiplies two polynomials of degree n and m; result has degree n+m.
void poly_multiply(const double *a, int n, const double *b, int m, double *out)
{
    for (int i = 0; i <= n + m; ++i)
        out[i] = 0;
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= m; ++j)
            out[i + j] += a[i] * b[j];
}

// Solves Ax = b for x, where A is n x n, b is n x 1
int gaussian_elimination(double **A, double *b, double *x, int n)
{
    for (int i = 0; i < n; ++i)
    {
        // Partial pivot
        int maxrow = i;
        for (int k = i + 1; k < n; ++k)
            if (fabs(A[k][i]) > fabs(A[maxrow][i]))
                maxrow = k;
        // Swap rows
        double *tmp = A[i];
        A[i] = A[maxrow];
        A[maxrow] = tmp;
        double tb = b[i];
        b[i] = b[maxrow];
        b[maxrow] = tb;

        // Eliminate
        for (int k = i + 1; k < n; ++k)
        {
            double f = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j)
                A[k][j] -= f * A[i][j];
            b[k] -= f * b[i];
        }
    }
    // Back substitution
    for (int i = n - 1; i >= 0; --i)
    {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
    return 1;
}

// Returns the rank of matrix A (m x n)
int matrix_rank(double **A, int m, int n)
{
    int rank = 0;
    int *row_selected = (int *)calloc(m, sizeof(int));
    for (int col = 0; col < n; ++col)
    {
        int row;
        for (row = 0; row < m; ++row)
            if (!row_selected[row] && fabs(A[row][col]) > 1e-10)
                break;
        if (row < m)
        {
            ++rank;
            row_selected[row] = 1;
            for (int i = 0; i < m; ++i)
            {
                if (i != row)
                {
                    double f = A[i][col] / A[row][col];
                    for (int j = col; j < n; ++j)
                        A[i][j] -= f * A[row][j];
                }
            }
        }
    }
    free(row_selected);
    return rank;
}

// For 2x2 matrix: |A| = a11*a22 - a12*a21
double det2(double a11, double a12, double a21, double a22)
{
    return a11 * a22 - a12 * a21;
}

// For 3x3 matrix: (general formula)
double det3(double m[3][3])
{
    return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] + m[0][2] * m[1][0] * m[2][1] - m[0][2] * m[1][1] * m[2][0] - m[0][0] * m[1][2] * m[2][1] -
           m[0][1] * m[1][0] * m[2][2];
}

// Characteristic polynomial for 2x2: λ^2 - tr(A)λ + det(A)
void char_poly_2x2(double m[2][2], double *out)
{
    double tr = m[0][0] + m[1][1];
    double det = det2(m[0][0], m[0][1], m[1][0], m[1][1]);
    out[0] = 1;
    out[1] = -tr;
    out[2] = det;
}

int quadratic_roots(double a, double b, double c, double *x1, double *x2)
{
    double d = b * b - 4 * a * c;
    if (d < 0)
        return 0; // No real roots
    *x1 = (-b + sqrt(d)) / (2 * a);
    *x2 = (-b - sqrt(d)) / (2 * a);
    return 1;
}

// Solve ax^3 + bx^2 + cx + d = 0, returns number of real roots, stores them in roots[]
int cubic_roots(double a, double b, double c, double d, double *roots)
{
    if (fabs(a) < 1e-12)
        return quadratic_roots(b, c, d, roots, roots + 1);
    double A = b / a, B = c / a, C = d / a;
    double Q = (3 * B - A * A) / 9.0;
    double R = (9 * A * B - 27 * C - 2 * A * A * A) / 54.0;
    double D = Q * Q * Q + R * R;
    if (D >= 0)
    {
        double S = cbrt(R + sqrt(D));
        double T = cbrt(R - sqrt(D));
        roots[0] = -A / 3 + (S + T);
        return 1;
    }
    else
    {
        double theta = acos(R / sqrt(-Q * Q * Q));
        double r = 2 * sqrt(-Q);
        roots[0] = r * cos(theta / 3) - A / 3;
        roots[1] = r * cos((theta + 2 * M_PI) / 3) - A / 3;
        roots[2] = r * cos((theta + 4 * M_PI) / 3) - A / 3;
        return 3;
    }
}
double f(double x)
{
    return x * x;
}

int main()
{
    // 1. getModuloClass demo
    printf("getModuloClass(17, 5) = %d\n", getModuloClass(17, 5)); // Output: 2

    // 2. rational_root demo
    int coeffs[] = {1, -3, 2}; // x^2 - 3x + 2 = 0, roots: 1 and 2
    printf("rational_root({1,-3,2}, 2) = %d\n", rational_root(coeffs, 2));

    // 3. getGCD demo
    printf("getGCD(36, 24) = %d\n", getGCD(36, 24)); // Output: 12

    // 4. extendedEuclideanAlgorithm demo
    int x, y;
    int gcd = extendedEuclideanAlgorithm(30, 20, &x, &y);
    printf("extendedEuclideanAlgorithm(30, 20): gcd=%d, x=%d, y=%d\n", gcd, x, y);

    // 5. areCoprime demo
    printf("areCoprime(14, 15) = %d\n", areCoprime(14, 15)); // Output: 1

    // 6. primeComputation demo
    int primes[10], count;
    count = primeComputation(30, primes, 10);
    printf("primeComputation(30): ");
    for (int i = 0; i < count; i++)
        printf("%d ", primes[i]);
    printf("\n");

    // 7. sieveOfEratosthenes demo
    int sieve_primes[100];
    int sieve_count = sieveOfEratosthenes(30, sieve_primes, 100);
    printf("sieveOfEratosthenes(30): ");
    for (int i = 0; i < sieve_count; i++)
        printf("%d ", sieve_primes[i]);
    printf("\n");

    // 8. raise demo
    printf("raise(3,4) = %d\n", raise(3, 4)); // Output: 81

    // 9. fibonacciSequence demo
    int fib[10];
    fibonacciSequence(fib, 10);
    printf("fibonacciSequence(10): ");
    for (int i = 0; i < 10; i++)
        printf("%d ", fib[i]);
    printf("\n");

    // 10. isEven & isOdd demo
    printf("isEven(6) = %d, isOdd(6) = %d\n", isEven(6), isOdd(6)); // 1, 0

    // 11. getMode demo
    int arr1[] = {1, 2, 2, 3, 4, 2, 5};
    printf("getMode(arr1, 7, 5) = %d\n", getMode(arr1, 7, 5)); // Output: 2

    // 12. getMedian demo
    int arr2[] = {5, 3, 1, 4, 2};
    printf("getMedian(arr2, 5) = %d\n", getMedian(arr2, 5)); // Output: 3

    // 13. getMean demo
    int arr3[] = {1, 2, 3, 4, 5};
    printf("getMean(arr3, 5) = %d\n", getMean(arr3, 5)); // Output: 3

    // 14. getRange demo
    int arr4[] = {2, 7, 4, 1, 9};
    printf("getRange(arr4, 5) = %d\n", getRange(arr4, 5)); // Output: 8

    // 15. factorial demo
    printf("factorial(5, 2) = %zu\n", factorial(5, 2)); // 5*4*3 = 60

    // 16. permutationFunction demo
    printf("permutationFunction(5, 3, 0) = %zu\n", permutationFunction(5, 3, 0)); // 60

    // 17. combinationFunction demo
    printf("combinationFunction(5, 3, 0) = %zu\n", combinationFunction(5, 3, 0)); // 10

    // 18. Matrix demo
    Matrix m1 = create_matrix(2, 2), m2 = create_matrix(2, 2);
    set_matrix(&m1, 0, 0, 1);
    set_matrix(&m1, 0, 1, 2);
    set_matrix(&m1, 1, 0, 3);
    set_matrix(&m1, 1, 1, 4);
    set_matrix(&m2, 0, 0, 5);
    set_matrix(&m2, 0, 1, 6);
    set_matrix(&m2, 1, 0, 7);
    set_matrix(&m2, 1, 1, 8);
    printf("Matrix m1:\n");
    print_matrix(&m1);
    printf("Matrix m2:\n");
    print_matrix(&m2);
    Matrix msum = add_matrix(&m1, &m2);
    printf("Sum:\n");
    print_matrix(&msum);
    Matrix mmul = multiply_matrix(&m1, &m2);
    printf("Product:\n");
    print_matrix(&mmul);
    Matrix mtrans = transpose_matrix(&m1);
    printf("Transpose of m1:\n");
    print_matrix(&mtrans);
    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&msum);
    free_matrix(&mmul);
    free_matrix(&mtrans);

    // 19. Set demo
    int sarr[] = {1, 2, 2, 3, 4, 5, 3};
    Set s1 = create_set(sarr, 7);
    print_set(&s1);
    Set s2 = create_set((int[]){4, 5, 6}, 3);
    print_set(&s2);
    Set u = set_union(&s1, &s2), inter = set_intersection(&s1, &s2), diff = set_difference(&s1, &s2);
    printf("Union: ");
    print_set(&u);
    printf("Intersection: ");
    print_set(&inter);
    printf("Difference: ");
    print_set(&diff);
    printf("Is s2 subset of s1? %d\n", set_subset(&s2, &s1));
    free_set(&s1);
    free_set(&s2);
    free_set(&u);
    free_set(&inter);
    free_set(&diff);

    // 20. binomial_coefficient demo
    printf("binomial_coefficient(5, 2) = %llu\n", binomial_coefficient(5, 2)); // Output: 10

    // 21. mod_pow & modinv demo
    printf("mod_pow(2, 10, 1000) = %d\n", mod_pow(2, 10, 1000)); // 1024 % 1000 = 24
    printf("modinv(3, 11) = %d\n", modinv(3, 11));               // 4

    // 22. matrix_determinant & matrix_inverse demo
    double **mat = (double **)malloc(2 * sizeof(double *));
    double **inv = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        mat[i] = (double *)malloc(2 * sizeof(double));
        inv[i] = (double *)malloc(2 * sizeof(double));
    }
    mat[0][0] = 1;
    mat[0][1] = 2;
    mat[1][0] = 3;
    mat[1][1] = 4;
    printf("Determinant: %f\n", matrix_determinant(mat, 2));
    if (matrix_inverse(mat, inv, 2))
    {
        printf("Inverse:\n");
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
                printf("%8.4f ", inv[i][j]);
            printf("\n");
        }
    }
    for (int i = 0; i < 2; i++)
    {
        free(mat[i]);
        free(inv[i]);
    }
    free(mat);
    free(inv);

    // 23. matrix_minor & matrix_cofactor demo
    double m3[3][3] = {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}};
    double *pmat[3] = {m3[0], m3[1], m3[2]};
    printf("Minor (0,0): %f\n", matrix_minor(pmat, 3, 0, 0));
    printf("Cofactor (0,0): %f\n", matrix_cofactor(pmat, 3, 0, 0));

    // 24. isqrt demo
    printf("isqrt(17) = %d\n", isqrt(17)); // Output: 4

    // 25. fast_pow demo
    printf("fast_pow(2, 10) = %lld\n", fast_pow(2, 10)); // Output: 1024

    // 26. is_prime demo
    printf("is_prime(29) = %d\n", is_prime(29)); // Output: 1

    // 27. gcd_array & lcm_array demo
    int arrg[] = {12, 18, 24};
    printf("gcd_array: %d, lcm_array: %d\n", gcd_array(arrg, 3), lcm_array(arrg, 3));

    // 28. lcm demo
    printf("lcm(12, 18) = %d\n", lcm(12, 18)); // Output: 36

    // 29. FFT demo omitted for brevity (requires complex setup)

    // 30. modpow64 demo
    printf("modpow64(2, 62, 1000000007) = %llu\n", (unsigned long long)modpow64(2, 62, 1000000007));

    // 31. Elliptic curve point addition demo
    ECPoint P = {2, 4, 0}, Q = {5, 9, 0};
    ECPoint R = ec_add(P, Q, 2, 17); // y^2 = x^3 + 2x + b mod 17
    printf("ECPoint addition result: (%" PRIu64 ", %" PRIu64 "), is_inf=%d\n", R.x, R.y, R.is_inf);
    // 32. adaptive_simpson demo

    printf("adaptive_simpson for x^2 [0,1]: %.5f\n", adaptive_simpson(f, 0, 1, 1e-6, 15));

    // 33. solve_linear_system demo
    double **A = (double **)malloc(2 * sizeof(double *));
    double bvec[2] = {5, 11}, xvec[2];
    for (int i = 0; i < 2; i++)
    {
        A[i] = (double *)malloc(2 * sizeof(double));
    }
    A[0][0] = 1;
    A[0][1] = 2;
    A[1][0] = 3;
    A[1][1] = 4;
    solve_linear_system(A, bvec, xvec, 2);
    printf("Solution to Ax = b: x = [%.2f, %.2f]\n", xvec[0], xvec[1]);
    for (int i = 0; i < 2; i++)
        free(A[i]);
    free(A);

    // 34. powmod and miller_rabin demo
    printf("powmod(2, 10, 17) = %" PRIu64 "\n", powmod(2, 10, 17));
    printf("miller_rabin(17, 5) = %d\n", miller_rabin(17, 5));

    // 35. variance, stddev, pearson demo
    double data1[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double data2[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    printf("variance(data1,5) = %f\n", variance(data1, 5));
    printf("stddev(data1,5) = %f\n", stddev(data1, 5));
    printf("pearson(data1,data2,5) = %f\n", pearson(data1, data2, 5));

    // 36. gamma_lanczos, erf_approx, bessel_j0
    printf("gamma_lanczos(5) = %f\n", gamma_lanczos(5));
    printf("erf_approx(1.0) = %f\n", erf_approx(1.0));
    printf("bessel_j0(1.0) = %f\n", bessel_j0(1.0));

    // 37. elliptic_k, elliptic_e
    printf("elliptic_k(0.5) = %f\n", elliptic_k(0.5));
    printf("elliptic_e(0.5) = %f\n", elliptic_e(0.5));

    // 38. discrete_log demo
printf("discrete_log(2, 8, 17) = %" PRId64 "\n", discrete_log(2, 8, 17));
    // 39. synthetic_division demo
    double poly[] = {1, -6, 11, -6}; // (x-1)(x-2)(x-3)
    double out[3], rem;
    synthetic_division(poly, 3, 1.0, out, &rem);
    printf("synthetic_division by (x-1): quotient: ");
    for (int i = 0; i < 3; i++)
        printf("%.2f ", out[i]);
    printf(", remainder: %.2f\n", rem);

    // 40. poly_multiply demo
    double pa[] = {1, 2}, pb[] = {1, 3};
    double pm[3];
    poly_multiply(pa, 1, pb, 1, pm);
    printf("poly_multiply: ");
    for (int i = 0; i < 3; i++)
        printf("%.2f ", pm[i]);
    printf("\n");

    // 41. gaussian_elimination demo
    double **G = (double **)malloc(2 * sizeof(double *));
    double gb[2] = {6, 8}, gx[2];
    for (int i = 0; i < 2; i++)
    {
        G[i] = (double *)malloc(2 * sizeof(double));
    }
    G[0][0] = 2;
    G[0][1] = 1;
    G[1][0] = 5;
    G[1][1] = 7;
    gaussian_elimination(G, gb, gx, 2);
    printf("Gaussian elimination solution: x = [%.2f, %.2f]\n", gx[0], gx[1]);
    for (int i = 0; i < 2; i++)
        free(G[i]);
    free(G);

    // 42. matrix_rank demo
    double **M = (double **)malloc(3 * sizeof(double *));
    for (int i = 0; i < 3; i++)
    {
        M[i] = (double *)malloc(3 * sizeof(double));
    }
    M[0][0] = 1;
    M[0][1] = 2;
    M[0][2] = 3;
    M[1][0] = 4;
    M[1][1] = 5;
    M[1][2] = 6;
    M[2][0] = 7;
    M[2][1] = 8;
    M[2][2] = 9;
    printf("matrix_rank = %d\n", matrix_rank(M, 3, 3));
    for (int i = 0; i < 3; i++)
        free(M[i]);
    free(M);

    // 43. det2, det3, char_poly_2x2 demo
    printf("det2(1,2,3,4) = %f\n", det2(1, 2, 3, 4));
    double mat3[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    printf("det3(mat3) = %f\n", det3(mat3));
    double mat2[2][2] = {{1, 2}, {3, 4}}, cpoly[3];
    char_poly_2x2(mat2, cpoly);
    printf("char_poly_2x2: %.1f x^2 %+1.1f x %+1.1f\n", cpoly[0], cpoly[1], cpoly[2]);

    // 44. quadratic_roots, cubic_roots demo
    double rx1, rx2;
    if (quadratic_roots(1, -3, 2, &rx1, &rx2))
        printf("quadratic_roots: roots = %.2f, %.2f\n", rx1, rx2);
    double croots[3];
    int num_roots = cubic_roots(1, -6, 11, -6, croots);
    printf("cubic_roots: ");
    for (int i = 0; i < num_roots; i++)
        printf("%.2f ", croots[i]);
    printf("\n");

    return 0;
}
