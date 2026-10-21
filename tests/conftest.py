# Mandatory File Service test adapter; existing processing fixtures are retained.
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_spec = spec_from_file_location(
    "image_file_service_testing", Path(__file__).with_name("fake_files.py")
)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
install = _module.install
from image_translation.clients.file_service_client import FileServiceClient, FileServiceError


@pytest.fixture(autouse=True)
def file_service(monkeypatch):
    return install(monkeypatch, FileServiceClient, FileServiceError, "image")
