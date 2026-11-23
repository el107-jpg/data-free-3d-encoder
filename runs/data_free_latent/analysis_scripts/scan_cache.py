"""Find files whose copy in the memory cache differs from the disk, and optionally drop those copies.

usage: scan_cache.py [--evict] <file or folder> [...]
Runs on the system Python (no numpy). Compares in 1 MB pieces; reports the byte offsets that differ.
With --evict a damaged copy is dropped from the cache and the file is compared once more.
"""
import mmap
import os
import sys

args = sys.argv[1:]
evict = "--evict" in args
roots = [a for a in args if a != "--evict"]
CHUNK = 1 << 20
buf = mmap.mmap(-1, CHUNK)


def differing(path):
    size = os.path.getsize(path)
    if size == 0:
        return []
    bad = []
    try:
        direct = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return None
    try:
        with open(path, "rb") as cached:
            offset = 0
            while offset < size:
                got = os.readv(direct, [buf])
                if got <= 0:
                    break
                want = min(got, size - offset)
                a = cached.read(want)
                b = buf[:want]
                if a != b:
                    bad += [offset + i for i in range(want) if a[i] != b[i]][:16]
                offset += want
    except OSError:
        return None
    finally:
        os.close(direct)
    return bad


files = []
for root in roots:
    if os.path.isfile(root):
        files.append(root)
    else:
        for folder, _, names in os.walk(root):
            files += [os.path.join(folder, n) for n in names if not os.path.islink(os.path.join(folder, n))]
checked = total = 0
found = []
for path in files:
    bad = differing(path)
    if bad is None:
        continue
    checked += 1
    total += os.path.getsize(path)
    if bad:
        note = ""
        if evict:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            again = differing(path)
            note = "  -> dropped, now matches the disk" if not again else f"  -> STILL differs after dropping ({len(again)} bytes; a running program has it loaded)"
        found.append(path)
        print(f"DAMAGED in memory: {path}\n  offsets {bad[:6]} (in-page {[o % 4096 for o in bad[:6]]}){note}", flush=True)
print(f"checked {checked} files, {total / 1e9:.2f} GB; damaged in memory: {len(found)}")
