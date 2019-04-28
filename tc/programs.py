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
def conv2d(float(B, IP, H, W) in, float(OP, IP, KH, KW) weight) -> (out) {
   out(b, op, h, w) +=! in(b, ip, h + kh, w + kw) * weight(op, ip, kh, kw)
}
"""