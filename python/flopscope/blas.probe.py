# python/flopscope/blas_probe.py
import os, re, sys, tempfile
from contextlib import contextmanager

# Regex simples para MKL / OpenBLAS (ajustados a los logs típicos)
_GEMM_RE = re.compile(r"(?:c?blas_)?[sdcz]gemm.*?(?:m=|M=)?(\d+).*?(?:n=|N=)?(\d+).*?(?:k=|K=)?(\d+)", re.IGNORECASE)
_GEMV_RE = re.compile(r"(?:c?blas_)?[sdcz]gemv.*?(?:m=|M=)?(\d+).*?(?:n=|N=)?(\d+)", re.IGNORECASE)
_DOT_RE  = re.compile(r"(?:c?blas_)?[sdcz]dot.*?(?:n=|N=)?(\d+)", re.IGNORECASE)
# Puedes extender: syrk, trsm, axpy, etc.

def _flops_from_line(s: str) -> int:
    # MKL: "dgemm(...) m=..., n=..., k=..." ; OpenBLAS: "OpenBLAS : ... gemm ... m=...,n=...,k=..."
    m = _GEMM_RE.search(s)
    if m:
        M, N, K = map(int, m.groups())
        return 2 * M * N * K
    m = _GEMV_RE.search(s)
    if m:
        M, N = map(int, m.groups())
        return 2 * M * N
    m = _DOT_RE.search(s)
    if m:
        (N,) = map(int, m.groups())
        return 2 * N
    return 0

class _StderrCapture:
    """Redirige stderr (a nivel FD) a un archivo temporal y luego lo parsea."""
    def __init__(self):
        self._tmp = None
        self._orig_fd = None
        self.lines = []

    def __enter__(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+b")
        self._orig_fd = os.dup(2)          # dup de stderr
        os.dup2(self._tmp.fileno(), 2)     # redirige FD 2 -> tmp
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            os.dup2(self._orig_fd, 2)      # restaura stderr
            os.close(self._orig_fd)
        finally:
            self._tmp.flush()
            self._tmp.seek(0)
            for raw in self._tmp:
                try:
                    self.lines.append(raw.decode(errors="ignore"))
                except Exception:
                    pass
            path = self._tmp.name
            self._tmp.close()
            try: os.unlink(path)
            except Exception: pass

class BLASCountResult:
    def __init__(self, total: int, lines: list[str]):
        self.total = int(total)
        self.lines = lines  # por si quieres inspeccionar qué kernels aparecieron
    def __repr__(self):
        return f"BLASCountResult(total={self.total})"

@contextmanager
def count_blas_flops(backend: str = "auto"):
    """
    Context manager que activa logs de BLAS (MKL/OpenBLAS), captura stderr
    y devuelve un BLASCountResult con FLOPs aproximados.
      backend: "mkl" | "openblas" | "auto"
    """
    # Guardar/poner variables de entorno
    saved = {}
    def _setenv(k, v):
        if k in os.environ:
            saved[k] = os.environ[k]
        os.environ[k] = v

    if backend in ("auto", "mkl"):
        _setenv("MKL_VERBOSE", "1")
    if backend in ("auto", "openblas"):
        _setenv("OPENBLAS_VERBOSE", "2")

    # Capturar stderr a nivel de FD
    cap = _StderrCapture()
    flops_acc = 0
    with cap:
        yield  # ejecutar el bloque del usuario

    # Parsear líneas y sumar FLOPs
    for line in cap.lines:
        flops_acc += _flops_from_line(line)

    # Restaurar entorno
    for k, v in saved.items():
        os.environ[k] = v
    for k in ("MKL_VERBOSE", "OPENBLAS_VERBOSE"):
        if k not in saved and k in os.environ:
            del os.environ[k]

    # Entregar resultado en atributo global de conveniencia (o retorna por función)
    globals()["_last_blas_count"] = BLASCountResult(flops_acc, cap.lines)

def last_blas_flops() -> int:
    return int(getattr(globals().get("_last_blas_count", None), "total", 0))

def last_blas_result() -> BLASCountResult | None:
    return globals().get("_last_blas_count", None)
