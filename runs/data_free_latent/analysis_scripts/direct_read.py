"""Compare a file as the memory cache holds it with the same file read straight from disk.

usage: direct_read.py <file> [<file> ...]
"""
import hashlib
import mmap
import os
import sys

for path in sys.argv[1:]:
    cached = open(path, "rb").read()
    size = len(cached)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError as error:
        print(path, "cannot open for direct reading:", error)
        continue
    block = 1 << 20
    buf = mmap.mmap(-1, block)
    disk = bytearray()
    try:
        while True:
            got = os.readv(fd, [buf])
            if got <= 0:
                break
            disk += buf[:got]
    except OSError as error:
        print(path, "direct read failed:", error, "after", len(disk), "bytes")
    os.close(fd)
    disk = bytes(disk[:size])
    import numpy as np
    n = min(len(disk), size)
    differ = np.nonzero(np.frombuffer(disk, np.uint8, n) != np.frombuffer(cached, np.uint8, n))[0].tolist()
    print(path)
    print(f"  size {size}; read from disk {len(disk)}; bytes that differ: {len(differ)} at {differ[:8]}")
    for i in differ[:4]:
        print(f"    offset {i}: memory copy {cached[i]:#04x}, disk {disk[i]:#04x}, difference {cached[i] ^ disk[i]:#04x}; "
              f"memory {cached[max(0, i - 8):i + 8]!r} disk {disk[max(0, i - 8):i + 8]!r}")
    print("  memory copy sha256", hashlib.sha256(cached).hexdigest()[:16], "| disk sha256", hashlib.sha256(disk).hexdigest()[:16])
