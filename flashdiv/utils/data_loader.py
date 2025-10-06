import h5py
import numpy as np
import torch
import warnings
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Any, Optional, Union


DEFAULT_FLOAT_DTYPE = torch.float32


@dataclass(frozen=True)
class DistributionSpec:
    distribution: Any
    n_samples: Optional[int] = None
    sample_fn: str = "sample"
    sample_kwargs: Optional[dict[str, Any]] = None
    device: Optional[Union[str, torch.device]] = None
    dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class FileSpec:
    path: Union[str, Path]
    key: Optional[str] = None
    mmap_mode: Optional[str] = "r"
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[str, torch.device]] = None


@dataclass(frozen=True)
class TensorSpec:
    tensor: torch.Tensor
    device: Optional[Union[str, torch.device]] = None
    dtype: Optional[torch.dtype] = None


DataSourceLike = Union[
    DistributionSpec,
    FileSpec,
    TensorSpec,
    torch.Tensor,
    np.ndarray,
    str,
    Path,
]


class _TensorAccessor:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.length = tensor.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


class _NpyAccessor:
    def __init__(
        self,
        path: Path,
        *,
        dtype: torch.dtype,
        mmap_mode: Optional[str],
        device: Optional[torch.device],
    ):
        self.path = str(path)
        self.mmap_mode = mmap_mode
        self.dtype = dtype
        self.device = device
        self._array = np.load(self.path, mmap_mode=self.mmap_mode)
        self.length = int(self._array.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._array is None:
            self._array = np.load(self.path, mmap_mode=self.mmap_mode)
        array_slice = self._array[idx]
        tensor = torch.as_tensor(array_slice).to(self.dtype)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state


class _H5Accessor:
    def __init__(
        self,
        path: Path,
        key: str,
        *,
        dtype: torch.dtype,
        device: Optional[torch.device],
    ):
        self.path = str(path)
        self.key = key
        self.dtype = dtype
        self.device = device
        with h5py.File(self.path, "r") as f:
            if key not in f:
                raise KeyError(f"Dataset '{key}' not found in {self.path}")
            self.length = int(f[key].shape[0])
        self._file = None

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        array = self._file[self.key][idx]
        tensor = torch.as_tensor(array).to(self.dtype)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state


class _AccessorBackedDataset(Dataset):
    def __init__(self, source_accessor, target_accessor, length: int):
        self.source_accessor = source_accessor
        self.target_accessor = target_accessor
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        return self.source_accessor[idx], self.target_accessor[idx]


def _maybe_device(device: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
    if device is None:
        return None
    return torch.device(device)


def _normalize_source(source: DataSourceLike, role: str):
    if isinstance(source, (DistributionSpec, FileSpec, TensorSpec)):
        return source
    if torch.is_tensor(source):
        return TensorSpec(tensor=source)
    if isinstance(source, np.ndarray):
        return TensorSpec(tensor=torch.from_numpy(source))
    if isinstance(source, (str, Path)):
        return FileSpec(path=source)
    sample_fn = getattr(source, "sample", None)
    if callable(sample_fn):
        return DistributionSpec(distribution=source)
    raise TypeError(f"Unsupported {role} data source type: {type(source)!r}")


def _make_accessor(
    spec,
    role: str,
    n_samples: Optional[int],
    device: Optional[Union[str, torch.device]],
    default_dtype: Optional[torch.dtype],
):
    base_device = _maybe_device(device)

    if isinstance(spec, TensorSpec):
        tensor = spec.tensor
        dtype_to_use = spec.dtype or default_dtype or tensor.dtype
        tensor = tensor.to(dtype=dtype_to_use)
        dest_device = _maybe_device(spec.device) or base_device
        if dest_device is not None:
            tensor = tensor.to(dest_device)
        return _TensorAccessor(tensor)

    if isinstance(spec, DistributionSpec):
        actual_n = spec.n_samples or n_samples
        if actual_n is None:
            raise ValueError(
                f"Number of samples is required for distribution-based {role} data source."
            )
        sampler = getattr(spec.distribution, spec.sample_fn)
        kwargs = spec.sample_kwargs or {}
        samples = sampler(actual_n, **kwargs)
        if isinstance(samples, tuple):
            samples = samples[0]
        tensor = torch.as_tensor(samples)
        dtype_to_use = spec.dtype or default_dtype or tensor.dtype
        tensor = tensor.to(dtype=dtype_to_use)
        dest_device = _maybe_device(spec.device) or base_device
        if dest_device is not None:
            tensor = tensor.to(dest_device)
        return _TensorAccessor(tensor)

    if isinstance(spec, FileSpec):
        path = Path(spec.path)
        dtype_to_use = spec.dtype or default_dtype or DEFAULT_FLOAT_DTYPE
        dest_device = _maybe_device(spec.device) or base_device
        ext = path.suffix.lower()
        if ext in {".h5", ".hdf5"}:
            if spec.key is None:
                raise ValueError(f"HDF5 {role} source requires 'key' to be specified.")
            return _H5Accessor(path, spec.key, dtype=dtype_to_use, device=dest_device)
        if ext == ".npy":
            return _NpyAccessor(path, dtype=dtype_to_use, mmap_mode=spec.mmap_mode, device=dest_device)
        if ext == ".npz":
            if spec.key is None:
                raise ValueError(f"NPZ {role} source requires 'key' to be specified.")
            with np.load(path) as data:
                if spec.key not in data:
                    raise KeyError(f"Array '{spec.key}' not found in {path}")
                array = data[spec.key]
            tensor = torch.from_numpy(array).to(dtype=dtype_to_use)
            if dest_device is not None:
                tensor = tensor.to(dest_device)
            return _TensorAccessor(tensor)
        if ext in {".pt", ".pth"}:
            tensor = torch.load(path)
            if not torch.is_tensor(tensor):
                raise ValueError(f"Torch file at {path} does not contain a tensor.")
            tensor = tensor.to(dtype=dtype_to_use)
            if dest_device is not None:
                tensor = tensor.to(dest_device)
            return _TensorAccessor(tensor)
        raise ValueError(f"Unsupported file extension '{ext}' for {role} data source at {path}.")

    raise TypeError(f"Unexpected specification type for {role}: {type(spec)!r}")


def create_paired_dataset(
    source: DataSourceLike,
    target: DataSourceLike,
    *,
    n_samples: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
):
    """Construct a Dataset pairing samples from arbitrary data sources."""

    source_spec = _normalize_source(source, role="source")
    target_spec = _normalize_source(target, role="target")
    source_accessor = _make_accessor(source_spec, "source", n_samples, device, dtype)
    target_accessor = _make_accessor(target_spec, "target", n_samples, device, dtype)

    candidates = [source_accessor.length, target_accessor.length]
    if n_samples is not None:
        candidates.append(n_samples)
    length = min(candidates)

    if length <= 0:
        raise ValueError("Paired dataset must have at least one sample.")

    if length < source_accessor.length or length < target_accessor.length:
        warnings.warn(
            "Mismatch between source and target sample counts; truncating to the minimum.",
            stacklevel=2,
        )

    dataset = _AccessorBackedDataset(source_accessor, target_accessor, length)
    return dataset, length





def get_dataset_keys(path):
    """Return all dataset paths inside the HDF5 file."""
    keys = []
    with h5py.File(path, "r") as f:
        f.visit(lambda name: keys.append(name) if isinstance(f[name], h5py.Dataset) else None)
    return keys


def get_frame_count(path):
    keys = get_dataset_keys(path)
    if not keys:
        raise ValueError(f"No datasets found in {path}")
    example = keys[0]
    with h5py.File(path, "r") as f:
        shape = f[example].shape
    return shape[0]


def load_training_dataset(
    device,
    *,
    source: Optional[DataSourceLike] = None,
    target: Optional[DataSourceLike] = None,
    n_samples: Optional[int] = None,
    dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
):
    """Create the dataset specified by CLI args."""

    if source is not None or target is not None:
        if source is None or target is None:
            raise ValueError("Provide both source and target when specifying custom data sources.")
        dataset, dataset_len = create_paired_dataset(
            source,
            target,
            n_samples=n_samples,
            device=device,
            dtype=dtype,
        )
        return dataset, dataset_len



def create_data_loaders(dataset, batch_size, val_split, *, shuffle=True):
    """Split dataset and create train/val DataLoader objects."""
    val_frac = float(val_split)
    val_frac = max(0.0, min(0.9, val_frac))
    train_size = int((1.0 - val_frac) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def prepare_dataloaders(
    args,
    device,
    *,
    source: Optional[DataSourceLike] = None,
    target: Optional[DataSourceLike] = None,
    n_samples: Optional[int] = None,
    dtype: torch.dtype = DEFAULT_FLOAT_DTYPE,
):
    dataset, nbsamples = load_training_dataset(
        device=device,
        source=source,
        target=target,
        n_samples=n_samples,
        dtype=dtype,
    )
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    return train_loader, val_loader, nbsamples

