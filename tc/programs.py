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

CONV2D = """
def conv2d(float(B, IP, H, W) IN, float(OP, IP, KH, KW) WEIGHT) -> (OUT) {
   OUT(b, op, h, w) +=! IN(b, ip, h + kh, w + kw) * WEIGHT(op, ip, kh, kw)
}
"""

TMM = """
def tmm(float(M,K) A, float(N,K) B) -> (C) {
     C(m,n) +=! A(m,kk) * B(n,kk)
}
"""

TBMM = """
def tbmm(float(B,N,M) X, float(B,K,M) Y) -> (Z) {
     Z(b,n,k) +=! X(b,n,m) * Y(b,k,m)
}
"""