# -*- coding: utf-8 -*-
"""多线程分段下载器：单连接被限速时用并发把带宽吃满。

用法:
    python pget.py <url> <out_path> [--threads 16] [--chunk-mb 8]

特性:
    - HTTP Range 分段，每段独立重试（默认 5 次）
    - 已完成的段落盘为 <out>.partNNN，中断后重跑会跳过已完成段
    - 服务器不支持 Range 时自动回退单线程
"""
import argparse
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 pget/1.0"}
_lock = threading.Lock()
_done = 0


def _range_size(url, timeout=30):
    """返回 (总字节数, 是否支持 Range)。"""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status == 206:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1]), True
        length = r.headers.get("Content-Length")
        return (int(length) if length else 0), False


def _fetch(url, start, end, path, retries=5):
    """下载 [start, end] 闭区间到 path，成功返回字节数。"""
    global _done
    want = end - start + 1
    if path.exists() and path.stat().st_size == want:
        with _lock:
            _done += want
        return want
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                got = 0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    with _lock:
                        _done += len(buf)
            if got == want:
                return got
            with _lock:
                _done -= got
            raise IOError(f"段长度不符 {got}/{want}")
        except Exception as e:
            print(f"  [重试 {attempt}/{retries}] {start}-{end}: {type(e).__name__} {e}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"段下载失败: {start}-{end}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    global _done
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total, ranged = _range_size(args.url)
    if total <= 0:
        raise SystemExit("拿不到文件大小")
    print(f"总大小 {total/1048576:.1f} MB, Range={ranged}", flush=True)

    if not ranged:
        print("服务器不支持 Range，单线程下载", flush=True)
        _fetch(args.url, 0, total - 1, out, args.retries)
        return

    chunk = args.chunk_mb * 1024 * 1024
    spans = [(s, min(s + chunk - 1, total - 1)) for s in range(0, total, chunk)]
    parts = [out.with_suffix(out.suffix + f".part{i:04d}") for i in range(len(spans))]
    print(f"分 {len(spans)} 段, {args.threads} 并发", flush=True)

    t0 = time.time()
    stop = threading.Event()

    def reporter():
        while not stop.wait(15):
            with _lock:
                d = _done
            el = time.time() - t0
            pct = d / total * 100
            print(f"  {d/1048576:8.1f}/{total/1048576:.1f} MB  {pct:5.1f}%  {d/1048576/el:6.2f} MB/s  剩余约 {(total-d)/max(d/el,1)/60:5.1f} 分钟", flush=True)

    th = threading.Thread(target=reporter, daemon=True)
    th.start()

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = [ex.submit(_fetch, args.url, s, e, p, args.retries)
                    for (s, e), p in zip(spans, parts)]
            for f in as_completed(futs):
                f.result()
    finally:
        stop.set()

    with open(out, "wb") as wf:
        for p in parts:
            with open(p, "rb") as rf:
                while True:
                    buf = rf.read(1 << 22)
                    if not buf:
                        break
                    wf.write(buf)
            p.unlink()
    el = time.time() - t0
    print(f"完成 {out}  {out.stat().st_size/1048576:.1f} MB  用时 {el/60:.1f} 分钟  均速 {total/1048576/el:.2f} MB/s", flush=True)


if __name__ == "__main__":
    main()
