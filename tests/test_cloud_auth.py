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


def test_preflight_auth_failed_access_denied_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The S3 API surfaces 403 with `Code='AccessDenied'` sometimes."""
    err = botocore.exceptions.ClientError(
        error_response={'Error': {'Code': 'AccessDenied'}},
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'auth_failed'


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


def test_preflight_bucket_missing_no_such_bucket_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = botocore.exceptions.ClientError(
        error_response={'Error': {'Code': 'NoSuchBucket'}},
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'bucket_missing'


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


# ============ Unknown-code fallback ============

def test_preflight_unknown_client_error_classifies_as_auth_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected botocore error code (e.g. throttling) shouldn't
    crash; surface it as `auth_failed` with the code in the message."""
    err = botocore.exceptions.ClientError(
        error_response={
            'Error': {'Code': 'SlowDown', 'Message': 'Reduce rate'},
        },
        operation_name='HeadBucket',
    )
    _install_session(monkeypatch, client=_StubClient(raises=err))
    with pytest.raises(CloudAuthError) as exc_info:
        preflight('s3://my-bucket/')
    assert exc_info.value.stage == 'auth_failed'
    assert re.search(r'\bSlowDown\b', exc_info.value.message)
