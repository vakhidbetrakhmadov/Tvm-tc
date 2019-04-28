MATMUL = """
def matmul(float(M, K) A, float(K, N) B) -> (C) {
    C(m, n) +=! A(m, r_k) * B(r_k, n)
}
"""

MAP = """
def map(float(M) A) -> (B) {
    B(m) =! A(m) * 3.14
}
"""

REDUCE = """
def reduce(float(M) A) -> (B) {
    float b = 0
    b += A(m)
    B = b
}
"""