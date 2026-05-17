"""Tests for cloud-auth preflight.

Each failure mode is exercised by monkeypatching the
`botocore.session.Session` factory to inject the desired
behavior; the head_bucket round-trip stays in unit-test scope
(no live cloud)."""
from __future__ import annotations

import re
from typing import cast

import botocore.exceptions
import pytest

from corroborate._internals.cloud_auth import (
    CloudAuthError,
    PreflightStage,
    _bucket_from_prefix,  # pyright: ignore[reportPrivateUsage]
    preflight,
)


# ============ URI parsing ============

def test_bucket_from_prefix_strips_scheme_and_path() -> None:
    assert _bucket_from_prefix('s3://my-bucket/foo/bar') == 'my-bucket'
    assert _bucket_from_prefix('s3://my-bucket/') == 'my-bucket'
    assert _bucket_from_prefix('s3://my-bucket') == 'my-bucket'


def test_bucket_from_prefix_rejects_non_s3_uri() -> None:
    with pytest.raises(ValueError, match='s3://'):
        _bucket_from_prefix('file:///tmp/foo')


# ============ Failure-mode fakes ============

class _FakeCreds:
    """Truthy stand-in for botocore.credentials.Credentials."""


class _StubClient:
    """Stub `s3` client whose `head_bucket` raises a configurable
    exception (or returns clean on `success=True`)."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises

    def head_bucket(self, *, Bucket: str) -> dict[str, object]:
        if self._raises is not None:
            raise self._raises
        return {'Bucket': Bucket}


class _StubSession:
    """Stub botocore.session.Session — controls cred resolution
    AND the client returned by `create_client`."""

    def __init__(
        self,
        *,
        creds_returned: bool = True,
        client: _StubClient | None = None,
        profile: str | None = None,
    ) -> None:
        self._creds = _FakeCreds() if creds_returned else None
        self._client = client or _StubClient()
        self.profile = profile

    def get_credentials(self) -> _FakeCreds | None:
        return self._creds

    def create_client(self, service: str) -> _StubClient:
        assert service == 's3'
        return self._client


def _install_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    creds_returned: bool = True,
    client: _StubClient | None = None,
) -> None:
    """Replace `botocore.session.Session` with the stub factory."""
    def _factory(profile: str | None = None) -> _StubSession:
        return _StubSession(
            creds_returned=creds_returned,
            client=client,
            profile=profile,
        )
    monkeypatch.setattr(
        'corroborate._internals.cloud_auth.botocore.session.Session',
        _factory,
    )


# ============ Stage: no_credentials ============

def test_preflight_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_session(monkeypatch, creds_returned=False)
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    err = exc_info.value
    assert err.stage == 'no_credentials'
    assert 'AWS_ACCESS_KEY_ID' in err.hint
    assert 'no aws-style credentials resolved' in err.message.lower()


def test_preflight_no_credentials_mentions_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session(monkeypatch, creds_returned=False)
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/', profile='r2-prod')
    assert "'r2-prod'" in exc_info.value.message


# ============ Stage: auth_failed (403) ============

def test_preflight_auth_failed_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = botocore.exceptions.ClientError(
        error_response={'Error': {'Code': '403', 'Message': 'Forbidden'}},
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'auth_failed'
    assert '403' in exc_info.value.message


# Note: `head_bucket` only ever produces numeric Code values
# (per botocore parsers.py — head responses have no body, so the
# parser uses the HTTP status code as the Code). Body-coded errors
# like `AccessDenied` / `NoSuchBucket` only appear from
# list/get/put operations. The fallback `auth_failed` branch
# handles any non-403/404/transient codes if they ever appear.


# ============ Stage: bucket_missing (404) ============

def test_preflight_bucket_missing_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = botocore.exceptions.ClientError(
        error_response={'Error': {'Code': '404', 'Message': 'Not Found'}},
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'bucket_missing'
    assert 'my-bucket' in exc_info.value.message
    assert 'AWS_ENDPOINT_URL' in exc_info.value.hint  # R2 hint


# `NoSuchBucket` string-code is unreachable through head_bucket
# (see note above the auth_failed section). 404 alone is enough.


# ============ Stage: network — transient + endpoint ============

def test_preflight_throttle_classified_as_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SlowDown (HTTP 503 throttling) must NOT classify as
    auth_failed — it's transient, not an auth problem. The
    operator's response (retry-with-backoff) differs from an
    auth fix (rotate credentials)."""
    err = botocore.exceptions.ClientError(
        error_response={
            'Error': {'Code': 'SlowDown', 'Message': 'Reduce rate'},
        },
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'network'
    assert 'SlowDown' in exc_info.value.message
    assert 'throttled' in exc_info.value.hint.lower()


# ============ Stage: network ============

def test_preflight_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = botocore.exceptions.EndpointConnectionError(
        endpoint_url='https://nope.invalid/',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'network'
    assert 'r2.cloudflarestorage.com' in exc_info.value.hint


# ============ Happy path ============

def test_preflight_clean_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session(monkeypatch, client=_StubClient())
    assert preflight('s3://my-bucket/') is None


# ============ Error rendering ============

def test_cloud_auth_error_str_includes_stage_and_hint() -> None:
    err = CloudAuthError(
        stage=cast(PreflightStage, 'auth_failed'),
        message='Authentication failed.',
        hint='Rotate the key in the dashboard.',
    )
    s = str(err)
    assert '[auth_failed]' in s
    assert 'Authentication failed' in s
    assert 'Rotate the key' in s


def test_cloud_auth_error_carries_chained_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`raise CloudAuthError(...) from e` preserves the original
    `__cause__` for debugging."""
    err = botocore.exceptions.ClientError(
        error_response={'Error': {'Code': '404'}},
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.__cause__ is err


# ============ CLI integration: --profile must reach fsspec ============

def test_preflight_or_exit_exports_aws_profile_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: --profile <name> on the CLI must propagate to
    the downstream cloud op (which goes through fsspec / s3fs, NOT
    through the botocore.session used for preflight). The fix sets
    `os.environ['AWS_PROFILE']` after preflight succeeds; s3fs
    reads that env var when constructing its S3FileSystem.

    Without this, `archive --profile r2` would pass preflight via
    the r2 profile but fail the actual upload (default chain)."""
    import os
    from corroborate.__main__ import _preflight_or_exit  # pyright: ignore[reportPrivateUsage]
    monkeypatch.delenv('AWS_PROFILE', raising=False)
    _install_session(monkeypatch, client=_StubClient())
    rc = _preflight_or_exit('s3://my-bucket/', profile='r2')
    assert rc is None  # success
    assert os.environ.get('AWS_PROFILE') == 'r2'


def test_preflight_or_exit_no_profile_leaves_env_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --profile is not passed, don't clobber an existing
    AWS_PROFILE env var (or set one when none was set)."""
    import os
    from corroborate.__main__ import _preflight_or_exit  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv('AWS_PROFILE', 'preexisting')
    _install_session(monkeypatch, client=_StubClient())
    rc = _preflight_or_exit('s3://my-bucket/', profile=None)
    assert rc is None
    assert os.environ.get('AWS_PROFILE') == 'preexisting'


def test_preflight_or_exit_does_not_set_env_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If preflight fails, the downstream cloud op never runs, so
    don't pollute the env with the profile that didn't authenticate."""
    import os
    from corroborate.__main__ import _preflight_or_exit  # pyright: ignore[reportPrivateUsage]
    monkeypatch.delenv('AWS_PROFILE', raising=False)
    _install_session(monkeypatch, creds_returned=False)
    rc = _preflight_or_exit('s3://my-bucket/', profile='broken')
    assert rc == 1
    assert 'AWS_PROFILE' not in os.environ


# ============ Unknown-code fallback ============

def test_preflight_unknown_client_error_falls_through_to_auth_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely unknown error code (not 403/404, not a transient)
    falls through to `auth_failed` with the code carried in the
    message so the operator can investigate."""
    err = botocore.exceptions.ClientError(
        error_response={
            'Error': {'Code': 'UnusualThing', 'Message': 'huh'},
        },
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'auth_failed'
    assert re.search(r'\bUnusualThing\b', exc_info.value.message)
