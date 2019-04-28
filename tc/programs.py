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

CONV = """
def conv2d(float(B, IP, H, W) IN, float(OP, IP, KH, KW) WEIGHT) -> (OUT) {
   OUT(b, op, h, w) +=! IN(b, ip, h + kh, w + kw) * WEIGHT(op, ip, kh, kw)
}
"""