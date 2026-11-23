"""Compare a file as the memory cache holds it with the same file read straight from disk (no numpy needed).

usage: direct_read_plain.py [--evict] <file> [<file> ...]
With --evict, a file whose cached copy differs from the disk is dropped from the memory cache afterwards
(the next read then comes from the disk again), and the comparison is repeated.
"""
import hashlib
import mmap
import os
import sys

args = sys.argv[1:]
evict = "--evict" in args
paths = [a for a in args if a != "--evict"]


def read_direct(path, size):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, 1 << 20)
    data = bytearray()
    try:
        while True:
            got = os.readv(fd, [buf])
            if got <= 0:
                break
            data += buf[:got]
    finally:
        os.close(fd)
    return bytes(data[:size])


def compare(path):
    cached = open(path, "rb").read()
    disk = read_direct(path, len(cached))
    differ = [i for i in range(min(len(disk), len(cached))) if disk[i] != cached[i]]
    print(path)
    print(f"  size {len(cached)}; bytes that differ between memory and disk: {len(differ)} at {differ[:8]}")
    for i in differ[:4]:
        print(f"    offset {i} (byte {i % 8} of its 8-byte word): memory {cached[i]:#04x}, disk {disk[i]:#04x}, difference {cached[i] ^ disk[i]:#04x}")
    print("  memory sha256", hashlib.sha256(cached).hexdigest()[:16], "| disk sha256", hashlib.sha256(disk).hexdigest()[:16])
    return len(differ)


for path in paths:
    bad = compare(path)
    if bad and evict:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        print("  dropped from the memory cache; reading again:")
        compare(path)
