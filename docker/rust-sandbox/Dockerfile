FROM rust:1.89-slim@sha256:8cffb8fe4e8a95cf0d6a2060375e5a28aff4c752155aa9f1f9193530769bdf66

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/* \
    && rustc --version | grep -F "rustc 1.89.0 " \
    && python3 -I -S -c 'import json, os, signal, subprocess, sys, tempfile, time; from pathlib import Path' \
    && command -v timeout >/dev/null

USER 65534:65534
WORKDIR /tmp
