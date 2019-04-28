MATMUL = """
def matmul(float(M, K) A, float(K, N) B) -> (C) {
    C(m, n) +=! A(m, r_k) * B(r_k, n)
}
"""

MAP = """
def map(float(M) A) -> (B) {
    B(m) = A(m) * 3.14
}
"""

REDUCE = """
def reduce(float(M) A) -> (B) { 
    B(n) = 0 where n 0..0
    B(n) += A(m) where n 0..0
}
"""