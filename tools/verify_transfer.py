"""Check the transfer folder against its sources, reading both straight from disk (not the memory cache).

usage: verify_transfer.py verify      compare every copied file with its source
       verify_transfer.py sums        write SHA256SUMS for the whole folder (from disk reads)

This machine has shown damaged cached copies of files, so cached reads are not trusted here.
"""
import hashlib
import lzma
import mmap
import os
import subprocess
import sys

ROOT = "/home/eric-luo/Downloads/pointcloud-motion-planning"
DEST = ROOT + "/data_free_latent_transfer"
HOME = os.path.expanduser("~")
JOBS = HOME + "/.claude/jobs"
MEM = HOME + "/.claude/projects/-home-eric-luo-Downloads-pointcloud-motion-planning/memory"
V6 = "data_free_anchors_procedural_thick_plus_thin_20261113_v6"

_buf = mmap.mmap(-1, 1 << 22)


def read_direct(path):
    size = os.path.getsize(path)
    if size == 0:
        return b""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    data = bytearray()
    try:
        while True:
            got = os.readv(fd, [_buf])
            if got <= 0:
                break
            data += _buf[:got]
    finally:
        os.close(fd)
    return bytes(data)


def walk(top):
    for r, dirs, files in os.walk(top):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            yield os.path.join(r, f)


def git_blob_ids(ref):
    out = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", "-z", ref], capture_output=True, check=True).stdout
    ids = {}
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        ids[name.decode()] = meta.split()[2].decode()
    return ids


def blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify():
    subprocess.run(["sync"], check=True)
    pairs = []  # (dst, src)
    trees = [
        ("runs/data_free_latent", ROOT + "/runs/data_free_latent"),
        ("session_scratch/8dde3c62_data-free-latent-space-review", JOBS + "/8dde3c62/tmp"),
        ("session_scratch/f264ef61_decoder-shape-residual-parameterization", JOBS + "/f264ef61/tmp"),
        ("session_scratch/ce6bff81_sphere-latent-space-prior-work", JOBS + "/ce6bff81/tmp"),
        ("figures", ROOT + "/figures"),
        ("notes/claude_memory", MEM),
        ("data/" + V6, ROOT + "/datasets/" + V6),
    ]
    for rel, src_top in trees:
        for dst in walk(os.path.join(DEST, rel)):
            if dst.endswith("anchors.pt.xz"):
                continue  # made here from anchors.pt; checked separately below
            pairs.append((dst, os.path.join(src_top, os.path.relpath(dst, os.path.join(DEST, rel)))))
    meta_top = os.path.join(DEST, "data/other_anchor_sets_metadata_only")
    for dst in walk(meta_top):
        pairs.append((dst, os.path.join(ROOT, "datasets", os.path.relpath(dst, meta_top))))
    pairs += [
        (DEST + "/notes/Data-free latent space.md", ROOT + "/Data-free latent space.md"),
        (DEST + "/notes/CLAUDE.md", ROOT + "/CLAUDE.md"),
        (DEST + "/notes/run-monitor.agent.md", HOME + "/.claude/agents/run-monitor.md"),
        (DEST + "/code/gpu_capped.py", ROOT + "/scripts/gpu_capped.py"),
        (DEST + "/environment/requirements-source-snapshot.txt", ROOT + "/requirements-source-snapshot.txt"),
    ]
    same = grown = 0
    bad = []
    nbytes = 0
    for dst, src in pairs:
        if not os.path.exists(src):
            bad.append((dst, "source missing"))
            continue
        d, s = read_direct(dst), read_direct(src)
        nbytes += len(d)
        if d == s:
            same += 1
        elif len(s) > len(d) and s[: len(d)] == d:
            grown += 1
            print(f"  source has grown since the copy (live file): {os.path.relpath(dst, DEST)} {len(d)} -> {len(s)}")
        else:
            bad.append((dst, f"differs from source ({len(d)} vs {len(s)} bytes)"))
    print(f"copied files: {len(pairs)} checked, {same} identical on disk, {grown} live files grown since, {len(bad)} bad, {nbytes / 1e6:.0f} MB read")

    # files taken from git: compare with the blob ids of the commit
    git_checked = 0
    for name, ref in [("rounded-box-base", "rounded-box-base"), ("j7-shell-surface", "experiment/j7-shell-surface"),
                      ("procedural-anchors", "procedural-anchors")]:
        ids = git_blob_ids(ref)
        top = os.path.join(DEST, "code", name)
        for dst in walk(top):
            rel = os.path.relpath(dst, top)
            if rel == "SOURCE_COMMIT.txt":
                continue
            git_checked += 1
            if rel not in ids:
                bad.append((dst, "not in commit"))
            elif blob_sha1(read_direct(dst)) != ids[rel]:
                bad.append((dst, "differs from git blob"))
    ids = git_blob_ids("rounded-box-base")
    for dst, rel in [(DEST + "/notes/docs/data_free_connectivity_plan.md", "docs/data_free_connectivity_plan.md"),
                     (DEST + "/notes/docs/data_free_implementation_spec.md", "docs/data_free_implementation_spec.md"),
                     (DEST + "/environment/pyproject.toml", "pyproject.toml")]:
        git_checked += 1
        if blob_sha1(read_direct(dst)) != ids[rel]:
            bad.append((dst, "differs from git blob"))
    print(f"files from git: {git_checked} checked against their commit")
    xz_path = f"{DEST}/data/{V6}/anchors.pt.xz"
    if os.path.exists(xz_path):
        if lzma.decompress(read_direct(xz_path)) == read_direct(f"{DEST}/data/{V6}/anchors.pt"):
            print("anchors.pt.xz: unpacks to exactly anchors.pt")
        else:
            bad.append((xz_path, "does not unpack to anchors.pt"))
    for dst, why in bad:
        print("  BAD", os.path.relpath(dst, DEST), "-", why)
    print("RESULT:", "OK" if not bad else f"{len(bad)} PROBLEMS")
    return 1 if bad else 0


def sums():
    subprocess.run(["sync"], check=True)
    lines = []
    for path in sorted(walk(DEST)):
        rel = os.path.relpath(path, DEST)
        if rel == "SHA256SUMS":
            continue
        lines.append(f"{hashlib.sha256(read_direct(path)).hexdigest()}  ./{rel}\n")
    with open(DEST + "/SHA256SUMS", "w") as f:
        f.writelines(lines)
    print(f"SHA256SUMS written for {len(lines)} files")
    return 0


if __name__ == "__main__":
    sys.exit(verify() if sys.argv[1] == "verify" else sums())
