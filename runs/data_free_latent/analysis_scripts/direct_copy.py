"""Copy a file by reading it straight from disk, bypassing the memory cache.

usage: direct_copy.py <source> <destination>
"""
import hashlib
import mmap
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
size = os.path.getsize(src)
fd = os.open(src, os.O_RDONLY | os.O_DIRECT)
buf = mmap.mmap(-1, 1 << 20)
data = bytearray()
while True:
    got = os.readv(fd, [buf])
    if got <= 0:
        break
    data += buf[:got]
os.close(fd)
data = bytes(data[:size])
with open(dst, "wb") as out:
    out.write(data)
os.chmod(dst, os.stat(src).st_mode & 0o777)
print(dst, len(data), "bytes, sha256", hashlib.sha256(data).hexdigest()[:16])
