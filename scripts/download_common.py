#!/usr/bin/env python3
"""Shared route, mirror, proxy, auth, and actionable error helpers for scripts/download_*.py."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_MS_ENDPOINT = "https://modelscope.cn"
USER_AGENT = "omni-rs/0.1"


def add_network_args(parser):
    """Add optional mirror/proxy flags without altering existing CLI contracts."""
    parser.add_argument(
        "--mirror",
        help="Custom Hugging Face mirror base URL (overrides HF_ENDPOINT, e.g. https://hf-mirror.com)",
    )
    parser.add_argument(
        "--proxy",
        help="Optional HTTP(S) proxy URL (overrides HTTP_PROXY/HTTPS_PROXY)",
    )


def _validate_http_url(label, value):
    trimmed = value.strip()
    parsed = urllib.parse.urlparse(trimmed)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid {label}: {trimmed!r}; expected an http:// or https:// URL / "
            f"{label}无效，请填写以 http:// 或 https:// 开头的地址。"
        )
    return trimmed.rstrip("/")


def hf_endpoint(mirror=None):
    raw = (mirror or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT).strip()
    if not raw:
        raw = DEFAULT_HF_ENDPOINT
    return _validate_http_url("HF_ENDPOINT / --mirror", raw)


def ms_endpoint():
    raw = (os.environ.get("OMNI_HUB_MS_ENDPOINT") or DEFAULT_MS_ENDPOINT).strip()
    if not raw:
        raw = DEFAULT_MS_ENDPOINT
    return _validate_http_url("ModelScope endpoint", raw)


def resolve_token(source):
    env_name = "MODELSCOPE_API_TOKEN" if source == "modelscope" else "HF_TOKEN"
    value = os.environ.get(env_name, "").strip()
    return value or None


def known_tokens():
    tokens = []
    for name in ("HF_TOKEN", "MODELSCOPE_API_TOKEN"):
        val = os.environ.get(name, "").strip()
        if val:
            tokens.append(val)
    return tokens


def redact_tokens(text):
    sanitized = str(text)
    for token in known_tokens():
        if token in sanitized:
            sanitized = sanitized.replace(token, "[REDACTED]")
        encoded = urllib.parse.quote(token, safe="")
        if encoded and encoded in sanitized:
            sanitized = sanitized.replace(encoded, "[REDACTED]")
    return sanitized


def auth_headers(source, extra=None):
    headers = {"User-Agent": USER_AGENT}
    token = resolve_token(source)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def build_opener(proxy=None):
    if proxy and proxy.strip():
        validated = _validate_http_url("--proxy", proxy)
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": validated, "https": validated})
        )
    return urllib.request.build_opener()


def hf_api_url(repository, revision, mirror=None):
    base = hf_endpoint(mirror)
    rev = urllib.parse.quote(revision, safe="")
    return f"{base}/api/models/{repository}/revision/{rev}?blobs=true"


def hf_resolve_url(repository, revision, filename, mirror=None):
    base = hf_endpoint(mirror)
    rev = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(filename, safe="/")
    return f"{base}/{repository}/resolve/{rev}/{encoded_path}"


def ms_api_url(repository, revision):
    base = ms_endpoint()
    rev = urllib.parse.quote(revision, safe="")
    return f"{base}/api/v1/models/{repository}/repo/files?Revision={rev}&Recursive=true"


def ms_resolve_url(repository, revision, filename):
    base = ms_endpoint()
    rev = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(filename, safe="/")
    return f"{base}/models/{repository}/resolve/{rev}/{encoded_path}"


def format_network_error(error, source, proxy=None):
    """Convert raw urllib/OS exceptions into actionable bilingual recovery hints."""
    token_env = "MODELSCOPE_API_TOKEN" if source == "modelscope" else "HF_TOKEN"
    source_name = "ModelScope" if source == "modelscope" else "Hugging Face"
    has_proxy = bool(
        (proxy and proxy.strip())
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )

    if isinstance(error, urllib.error.HTTPError):
        code = error.code
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if code in (401, 403):
            return redact_tokens(
                f"Authentication failed (HTTP {code}, {source_name}): check {token_env} "
                f"and verify gated-repo access agreement / "
                f"鉴权失败（HTTP {code}，{source_name}）：请检查环境变量 {token_env} 或确认已同意模型访问协议。"
            )
        if code == 429:
            wait_hint = f" (Retry-After: {retry_after}s)" if retry_after else ""
            return redact_tokens(
                f"Rate limited (HTTP 429, {source_name}){wait_hint}: retry later, set {token_env}, "
                f"or use HF_ENDPOINT mirror / 请求限流（HTTP 429），请稍后重试、配置 {token_env} 或切换镜像。"
            )
        return redact_tokens(
            f"HTTP {code} from {source_name}: check route/mirror/network connectivity / "
            f"{source_name} 返回 HTTP {code}，请检查线路或网络状态。"
        )

    if isinstance(error, urllib.error.URLError):
        reason = redact_tokens(error.reason)
        if has_proxy:
            return (
                f"Network/proxy connection failed ({reason}): verify --proxy / HTTPS_PROXY "
                f"or try --mirror / HF_ENDPOINT / 网络或代理连接失败（{reason}），请检查代理配置或切换镜像。"
            )
        return (
            f"Network connection failed ({reason}): try setting HF_ENDPOINT (e.g. https://hf-mirror.com) "
            f"or HTTPS_PROXY / 网络连接失败（{reason}），可设置 HF_ENDPOINT 镜像或 HTTPS_PROXY 代理。"
        )

    if isinstance(error, OSError) and getattr(error, "errno", None) == 28:
        return "Insufficient disk space (ENOSPC): free disk space or choose another --output directory / 磁盘空间不足，请释放空间或更换 --output 目录。"

    return redact_tokens(str(error))


def open_url(url, source, headers=None, timeout=120, proxy=None):
    req_headers = auth_headers(source, headers)
    request = urllib.request.Request(url, headers=req_headers)
    opener = build_opener(proxy)
    try:
        return opener.open(request, timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
        raise RuntimeError(format_network_error(error, source, proxy=proxy)) from error


def fetch_json(url, source, timeout=60, proxy=None):
    with open_url(url, source, timeout=timeout, proxy=proxy) as response:
        return json.load(response)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_safe_relative_path(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not name:
        raise RuntimeError(f"Unsafe repository path: {name}")


def download_file(
    url,
    destination,
    source="huggingface",
    expected_size=None,
    expected_hash=None,
    proxy=None,
    attempts=4,
    report_interval=20,
):
    """Download a file atomically via .part with HTTP Range resume and SHA256/size checks."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size_ok = expected_size is None or destination.stat().st_size == expected_size
        if size_ok:
            if expected_hash:
                actual = file_sha256(destination)
                if actual == expected_hash:
                    print(f"Already verified: {destination}", flush=True)
                    return actual
            else:
                print(f"Already present: {destination}", flush=True)
                return file_sha256(destination)

    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(attempts):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if expected_size is not None and offset > expected_size:
            raise RuntimeError(f"Oversized partial download: {temporary}")
        if expected_size is not None and offset == expected_size:
            break
        extra_headers = {"Range": f"bytes={offset}-"} if offset > 0 else None
        try:
            with open_url(url, source, headers=extra_headers, timeout=120, proxy=proxy) as response:
                status = getattr(response, "status", 200)
                if status == 206:
                    actual_range = response.headers.get("Content-Range", "")
                    if not actual_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Unexpected range for {destination.name}: {actual_range}")
                    mode = "ab"
                else:
                    mode = "wb"
                    offset = 0
                length_header = response.headers.get("Content-Length")
                target_size = expected_size
                if target_size is None and length_header is not None:
                    target_size = (offset + int(length_header)) if mode == "ab" else int(length_header)
                if target_size:
                    print(
                        f"Downloading {destination.name}: {offset / 1e9:.2f}/{target_size / 1e9:.2f} GB",
                        flush=True,
                    )
                else:
                    print(f"Downloading {destination.name}", flush=True)
                last_report = time.monotonic()
                with temporary.open(mode) as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        offset += len(block)
                        if target_size is not None and offset > target_size:
                            raise RuntimeError(f"Oversized response for {destination.name}")
                        if target_size and time.monotonic() - last_report > report_interval:
                            print(
                                f"{destination.name}: {offset / 1e9:.2f}/{target_size / 1e9:.2f} GB",
                                flush=True,
                            )
                            last_report = time.monotonic()
            if target_size is not None and offset != target_size:
                raise RuntimeError(
                    f"Truncated download: {destination.name} ({offset}/{target_size} bytes); "
                    "rerun to resume / 下载被截断，重新运行可断点续传。"
                )
            break
        except RuntimeError as error:
            msg = str(error)
            # Never silently retry auth/401/403/disk-full errors.
            if (
                "HTTP 401" in msg
                or "HTTP 403" in msg
                or "ENOSPC" in msg
                or attempt == attempts - 1
            ):
                raise
            print(f"Retry {destination.name}: {redact_tokens(error)}", flush=True)
            time.sleep(min(2 ** attempt, 8))

    actual = file_sha256(temporary)
    if expected_hash and actual != expected_hash:
        raise RuntimeError(
            f"SHA256 mismatch: {destination.name}; retained .part for inspection / "
            f"文件校验失败，已保留 .part 文件以便排查或删除后重试。"
        )
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise RuntimeError(f"Unexpected file length: {destination.name}")
    temporary.replace(destination)
    return actual


def write_manifest_atomic(directory, manifest):
    directory = Path(directory)
    temporary = directory / "download-manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "download-manifest.json")
