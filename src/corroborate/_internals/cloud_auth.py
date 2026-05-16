"""Pre-flight cloud-credential validation.

The framework uses boto3 / botocore's standard credential
resolution chain (no `.env` auto-loading). The chain searches in
this order:

1. explicit kwargs to the client constructor
2. environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
   AWS_SESSION_TOKEN, AWS_PROFILE, AWS_ENDPOINT_URL
3. shared credentials file (~/.aws/credentials), per-profile
4. AWS config file (~/.aws/config), including `[services X]`
   blocks for per-service `endpoint_url`
5. container / instance metadata (ECS / EC2 IAM role)

Cloudflare R2 needs `endpoint_url` to be set somewhere in this
chain — most commonly via `AWS_ENDPOINT_URL` env var, but the
`~/.aws/config` `[services r2]` block works too.

`preflight()` runs once at CLI entry for cloud-touching
subcommands so the user sees a clean, actionable error before
any expensive work starts. Library callers opt in by calling
`preflight()` explicitly — no magic at import time.
"""
from __future__ import annotations

import re
from typing import Literal

import botocore.exceptions
import botocore.session


PreflightStage = Literal[
    'no_credentials',  # boto3 chain returned nothing
    'auth_failed',     # credentials present but rejected (403)
    'bucket_missing',  # 404 — bucket name doesn't exist
    'network',         # transport-level failure (endpoint unreachable)
]


class CloudAuthError(RuntimeError):
    """Pre-flight failure with a typed `stage` + an actionable hint.

    Designed for CLI handlers to catch + render — see the framework's
    other exception types (`ConflictingArchive`, `ArchivePrecondition`)
    for the pattern."""

    def __init__(
        self, stage: PreflightStage, message: str, hint: str,
    ) -> None:
        super().__init__(f'[{stage}] {message}\n  → {hint}')
        self.stage: PreflightStage = stage
        self.message: str = message
        self.hint: str = hint


_S3_URI_RE = re.compile(r'^s3://([^/]+)')


def _bucket_from_prefix(remote_prefix: str) -> str:
    """Extract the bucket name from an `s3://bucket/...` URI."""
    m = _S3_URI_RE.match(remote_prefix)
    if m is None:
        raise ValueError(
            f'remote_prefix must be an s3:// URI; got: {remote_prefix!r}',
        )
    return m.group(1)


def preflight(
    remote_prefix: str,
    *,
    profile: str | None = None,
) -> None:
    """Verify cloud auth + bucket reachability before expensive work.

    Raises `CloudAuthError(stage, message, hint)` on any of:

    - `no_credentials`: the boto3 chain returned nothing
      (no env vars, no `~/.aws/credentials`, no IAM role).
    - `auth_failed`: credentials present but rejected by the
      bucket (403). Typical cause: stale key/secret or wrong
      bucket permissions.
    - `bucket_missing`: 404 on `head_bucket`. The credentials
      worked but no bucket at that name exists at the configured
      endpoint — often a wrong `AWS_ENDPOINT_URL`.
    - `network`: transport-level failure (endpoint unreachable,
      DNS error). Most commonly a malformed or missing endpoint
      for non-AWS S3-compatible backends.

    Pre-flight is cheap (one `head_bucket` call) and runs once
    per CLI invocation. Library callers opt in via explicit
    `preflight()` call; no implicit pre-flight on every cloud op."""
    bucket = _bucket_from_prefix(remote_prefix)

    # Step 1 — credential resolution. `get_credentials()` is
    # synchronous and walks the full chain.
    session = botocore.session.Session(profile=profile)
    creds = session.get_credentials()
    if creds is None:
        profile_clause = f' for profile {profile!r}' if profile else ''
        raise CloudAuthError(
            stage='no_credentials',
            message=(
                f'No AWS-style credentials resolved{profile_clause}.'
            ),
            hint=(
                'Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in the '
                'environment, OR add a profile to ~/.aws/credentials '
                'and pass --profile <name>, OR run from an environment '
                'with an IAM role attached.'
            ),
        )

    # Step 2 — auth + bucket check via head_bucket. botocore picks
    # up `AWS_ENDPOINT_URL` from the env (>=1.36); for older
    # versions / per-profile endpoints, the `~/.aws/config`
    # `[services X]` block carries it. Don't fight the resolution
    # chain here.
    s3 = session.create_client('s3')
    try:
        _ = s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.EndpointConnectionError as e:
        raise CloudAuthError(
            stage='network',
            message=f'Could not reach the S3 endpoint: {e}',
            hint=(
                'Check AWS_ENDPOINT_URL (or the [services] block in '
                '~/.aws/config) — for Cloudflare R2 it should be '
                'https://<account-id>.r2.cloudflarestorage.com. The '
                'host might be down or DNS-blocked.'
            ),
        ) from e
    except botocore.exceptions.NoCredentialsError as e:
        # Race / partial-resolve: get_credentials() returned non-None
        # above but the client invocation still rejected.
        raise CloudAuthError(
            stage='no_credentials',
            message='Credentials partially resolved but rejected by boto3.',
            hint='Re-verify ~/.aws/credentials format or the env vars.',
        ) from e
    except botocore.exceptions.ClientError as e:
        code: str = str(e.response.get('Error', {}).get('Code', ''))
        if code in ('403', 'Forbidden', 'AccessDenied'):
            profile_clause = (
                f' via profile {profile!r}' if profile else ''
            )
            raise CloudAuthError(
                stage='auth_failed',
                message=(
                    f'Authentication failed against bucket {bucket!r}'
                    f'{profile_clause} (HTTP 403).'
                ),
                hint=(
                    'Credentials were accepted by boto3 but rejected '
                    'by the bucket. Verify the access key + secret are '
                    'current and have read access to the bucket.'
                ),
            ) from e
        if code in ('404', 'NoSuchBucket'):
            raise CloudAuthError(
                stage='bucket_missing',
                message=f'Bucket {bucket!r} not found at the endpoint.',
                hint=(
                    'Credentials valid but no bucket at that name. '
                    'Check the remote_prefix URI AND AWS_ENDPOINT_URL '
                    '(R2 buckets only exist at the R2 endpoint, not '
                    'at s3.amazonaws.com).'
                ),
            ) from e
        raise CloudAuthError(
            stage='auth_failed',
            message=f'Cloud auth check failed: {code} ({e}).',
            hint='See the botocore error code above.',
        ) from e


__all__ = ['CloudAuthError', 'PreflightStage', 'preflight']
