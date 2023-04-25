import logging
import os
from pathlib import Path
from typing import Union

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


class FolderUploader:
    """Google Drive API wrapper to upload files to a specific folder."""

    def __init__(self, folder_id: str) -> None:
        self._drive = GoogleDrive(self._login())
        self._folder_id = folder_id
        # Set a higher log level for oauth2client to avoid noisy logs
        oauth2client_logger = logging.getLogger("oauth2client")
        oauth2client_logger.setLevel(logging.WARNING)

    def upload(self, file_path: Union[str, os.PathLike]) -> None:
        """Upload file to Google Drive folder."""
        file_path = Path(file_path)
        metadata = {
            "parents": [{"id": self._folder_id}],
            "title": file_path.name,
        }
        file = self._drive.CreateFile(metadata=metadata)
        file.SetContentFile(str(file_path))
        file.Upload()

    @staticmethod
    def _login() -> GoogleAuth:
        """Log in to Google Drive service with a service account."""
        settings = {
            "client_config_backend": "service",
            "service_config": {
                "client_json_file_path": "resources/service-secrets.json",
            },
        }
        gauth = GoogleAuth(settings=settings)
        gauth.ServiceAuth()
        return gauth
