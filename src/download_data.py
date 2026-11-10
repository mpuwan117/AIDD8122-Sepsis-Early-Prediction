# -*- coding: utf-8 -*-
"""Download the public PhysioNet/CinC Challenge 2019 sepsis training data.

Source: https://physionet.org/content/challenge-2019/1.0.0/
DOI: 10.13026/v64v-d857
Open training files are served without credentials from the
`physionet-open` S3 bucket (`--no-sign-request` equivalent).

The raw patient files are NOT committed to the repository: only this
downloader is version-controlled, so the study stays reproducible while
respecting the data-use terms.

Usage:
    python src/download_data.py --out ../data/raw --limit-per-set 0
    python src/download_data.py --out ../data/raw --limit-per-set 2000
"""
import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

BUCKET = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "challenge-2019/1.0.0/training/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
SETS = ("training_setA", "training_setB")


def list_keys(set_name):
    """Page through the public S3 bucket listing for one training set."""
    prefix = PREFIX + set_name + "/"
    keys, marker = [], ""
    while True:
        url = BUCKET + "?prefix=" + prefix + "&max-keys=1000"
        if marker:
            url += "&marker=" + marker
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        contents = root.findall(NS + "Contents")
        for item in contents:
            key = item.findtext(NS + "Key")
            if key and key.endswith(".psv"):
                keys.append(key)
        if root.findtext(NS + "IsTruncated") != "true":
            break
        marker = keys[-1] if keys else ""
        if not marker:
            break
    return keys


def fetch(key, dest_root, retries=3):
    rel = key[len("challenge-2019/1.0.0/training/"):]
    path = os.path.join(dest_root, rel.replace("/", os.sep))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip", key
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = BUCKET + key
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if not data:
                raise ValueError("empty response")
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return "ok", key
        except Exception as exc:
            if attempt == retries - 1:
                return "fail:" + str(exc)[:80], key
            time.sleep(1 + attempt)
    return "fail", key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/raw", help="destination directory for raw psv files")
    ap.add_argument("--limit-per-set", type=int, default=0,
                    help="max patients per set; 0 = full dataset (40,336 patients)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    dest = os.path.abspath(args.out)
    os.makedirs(dest, exist_ok=True)

    total_stats = {}
    for set_name in SETS:
        print("[download] listing", set_name, flush=True)
        keys = list_keys(set_name)
        if args.limit_per_set > 0:
            keys = keys[:args.limit_per_set]
        print("[download]", set_name, "files to fetch:", len(keys), flush=True)
        counts = {"ok": 0, "skip": 0, "fail": 0}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, k, dest) for k in keys]
            for i, fut in enumerate(as_completed(futures), 1):
                status, key = fut.result()
                if status == "ok":
                    counts["ok"] += 1
                elif status == "skip":
                    counts["skip"] += 1
                else:
                    counts["fail"] += 1
                    sys.stderr.write("FAILED " + key + " " + status + "\n")
                if i % 1000 == 0:
                    print("[download]", set_name, i, "/", len(keys),
                          "elapsed", round(time.time() - t0, 1), "s", flush=True)
        total_stats[set_name] = counts
        print("[download]", set_name, "done", counts,
              "elapsed", round(time.time() - t0, 1), "s", flush=True)

    print("[download] SUMMARY", total_stats, flush=True)
    print("[download] raw data at", dest, flush=True)


if __name__ == "__main__":
    main()
