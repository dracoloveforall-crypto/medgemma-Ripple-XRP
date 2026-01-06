# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data accessor for generic DICOM images stored in a DICOM store."""

from concurrent import futures
import contextlib
import functools
import os
import tempfile
from typing import Iterator, Mapping, Optional, Sequence

from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_generic import data_accessor_definition
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import generic_dicom_handler

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


# Transfer Syntax UID for uncompressed little endian.
_UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX_UID = '1.2.840.10008.1.2.1'


def _can_decode_transfer_syntax(transfer_syntax_uid: str) -> bool:
  if (
      transfer_syntax_uid
      in generic_dicom_handler.VALID_UNENCAPSULATED_DICOM_TRANSFER_SYNTAXES
  ):
    return True
  return dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
      transfer_syntax_uid
  )


def _download_dicom_instance(
    dwi: dicom_web_interface.DicomWebInterface,
    temp_dir: str,
    series_path: dicom_path.Path,
    index_transfer_syntax_uid_sop_instance_uid: tuple[int, str, str],
) -> str:
  """Downloads DICOM instance to a local file."""
  index, transfer_syntax_uid, sop_instance_uid = (
      index_transfer_syntax_uid_sop_instance_uid
  )
  instance_path = dicom_path.FromPath(
      series_path, instance_uid=sop_instance_uid
  )
  temp_file = os.path.join(temp_dir, f'{index}.dcm')
  with open(temp_file, 'wb') as output_file:
    try:
      if _can_decode_transfer_syntax(transfer_syntax_uid):
        dwi.download_instance_untranscoded(instance_path, output_file)
      else:
        # transcode to uncompressed little endian.
        dwi.download_instance(
            instance_path,
            _UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX_UID,
            output_file,
        )
    except ez_wsi_errors.HttpError as exp:
      raise data_accessor_errors.HttpError(str(exp)) from exp
  return temp_file


def _download_dicom_instances(
    stack: contextlib.ExitStack,
    instance: data_accessor_definition.DicomGenericImage,
    max_parallel_download_workers: int,
) -> Sequence[str]:
  """Downloads DICOM instances to a local file."""
  dwi = dicom_web_interface.DicomWebInterface(instance.credential_factory)
  temp_dir = stack.enter_context(tempfile.TemporaryDirectory())

  if instance.dicomweb_paths[0].type == dicom_path.Type.SERIES:
    if not instance.dicom_instances_metadata:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Missing DICOM instances metadata.'
      )
    selected_md_list = instance.dicom_instances_metadata
  else:
    series_metadata = {
        md.sop_instance_uid: md for md in instance.dicom_instances_metadata
    }
    # enable edge case of duplicate instance uids path list.
    selected_md_list = []
    for path in instance.dicomweb_paths:
      instance_md = series_metadata.get(path.instance_uid)
      if instance_md is None:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Missing DICOM instances metadata for SOPInstanceUID:'
            f' {path.instance_uid}'
        )
      selected_md_list.append(instance_md)

  series_path = instance.dicomweb_paths[0].GetSeriesPath()
  instance_list = []
  for i, md in enumerate(selected_md_list):
    instance_list.append((i, md.transfer_syntax_uid, md.sop_instance_uid))

  if len(instance_list) == 1 or max_parallel_download_workers == 1:
    return [
        _download_dicom_instance(
            dwi,
            temp_dir,
            series_path,
            li,
        ) for li in instance_list
    ]
  with futures.ThreadPoolExecutor(
      max_workers=max_parallel_download_workers
  ) as executor:
    return list(
        executor.map(
            functools.partial(
                _download_dicom_instance, dwi, temp_dir, series_path
            ),
            instance_list,
        )
    )


def _get_dicom_image(
    instance: data_accessor_definition.DicomGenericImage,
    local_file_paths: Sequence[str],
    modality_default_image_transform: Mapping[
        str, generic_dicom_handler.ModalityDefaultImageTransform
    ],
    max_parallel_download_workers: int,
) -> Iterator[np.ndarray]:
  """Returns image patch bytes from DICOM series."""
  dicom_handler = generic_dicom_handler.GenericDicomHandler(
      modality_default_image_transform,
      raise_error_if_invalid_dicom=True,
  )
  with contextlib.ExitStack() as stack:
    if not local_file_paths:
      local_file_paths = _download_dicom_instances(
          stack, instance, max_parallel_download_workers
      )
    try:
      yield from dicom_handler.process_files(
          instance.patch_coordinates,
          instance.base_request,
          abstract_handler.InputFileIterator(local_file_paths),
      )
    except pydicom.errors.InvalidDicomError as exp:
      raise data_accessor_errors.DicomError(str(exp)) from exp


class DicomGenericData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.DicomGenericImage, np.ndarray
    ]
):
  """Data accessor for generic DICOM images stored in a DICOM store."""

  def __init__(
      self,
      instance_class: data_accessor_definition.DicomGenericImage,
      modality_default_image_transform: Optional[
          Mapping[str, generic_dicom_handler.ModalityDefaultImageTransform]
      ] = None,
      max_parallel_download_workers: int = 1,
  ):
    super().__init__(instance_class)
    self._local_file_paths = []
    self._modality_default_image_transform = (
        modality_default_image_transform
        if modality_default_image_transform is not None
        else {}
    )
    self._max_parallel_download_workers = max(1, max_parallel_download_workers)

  @contextlib.contextmanager
  def _reset_local_file_path(self, *args, **kwds):
    del args, kwds
    try:
      yield
    finally:
      self._local_file_paths = []

  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator.

    Required that context manger must exist for life time of data accesor
    iterator after data is loaded.

    Args:
     stack: contextlib.ExitStack to manage resources.

    Returns:
      None
    """
    if self._local_file_paths:
      return
    self._local_file_paths = _download_dicom_instances(
        stack, self.instance, self._max_parallel_download_workers
    )
    stack.enter_context(self._reset_local_file_path())

  def data_iterator(self) -> Iterator[np.ndarray]:
    return _get_dicom_image(
        self.instance,
        self._local_file_paths,
        self._modality_default_image_transform,
        self._max_parallel_download_workers,
    )

  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is inline with request."""
    return False
