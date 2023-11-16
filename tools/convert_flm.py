#!/usr/bin/env python3
"""
    Author: Coder LSF (Liu Shaofeng)
      Date: 2023/11/15
"""

import os
import re
import io
import sys
import json
import math
import enum
import click
import struct
import pickle
import zipfile
import itertools
import numpy as np
from pathlib        import Path
from enum           import Enum
from dataclasses    import dataclass
from sentencepiece  import SentencePieceProcessor
from abc            import ABCMeta, abstractmethod
from typing         import IO, TYPE_CHECKING, Any, Callable, Generator, Iterable, Literal, Sequence, TypeVar

#if TYPE_CHECKING:
from typing import TypeAlias

NDArray: TypeAlias = 'np.ndarray[Any, Any]'

class ModelType(Enum):
    NONE  = 0
    LLAMA = 1

class QuantType(Enum):
    NONE  = 0
    INT16 = 1
    INT8  = 2
    INT4  = 3

class TokenType(Enum):
    UNKNOWN      = 0
    NORMAL       = 1
    CONTROL      = 2
    BYTE         = 3
    USER_DEFINED = 4
    UNUSED       = 5

class SpecialTokenType(Enum):
    NONE = 0
    BOS  = 1
    EOS  = 2
    PAD  = 3

    MAX  = 8

class ActType(Enum):
    NONE    = 0
    SILU    = 1
    SWIGLU  = 2

class RopeScalingType(Enum):
    NONE   = 'none'
    LINEAR = 'linear'
    YARN   = 'yarn'

class VocabType(Enum):
    NONE = 0
    BPE  = 1
    SPM  = 2

class DataType:
    __dtype: str

    def __init__(self, dtype:str):
        dtype = dtype.lower()
        if dtype == 'f32':
            dtype = 'float32'
        if dtype not in ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']:
            raise ValueError(f"Unsupported data_type[{dtype}]")
        self.__dtype = dtype

    def is_float64(self):
        return self.__dtype == 'float64'
    def is_float32(self):
        return self.__dtype == 'float32'
    def is_float16(self):
        return self.__dtype == 'float16'
    def is_int8(self):
        return self.__dtype == 'int8'
    def is_int16(self):
        return self.__dtype == 'int16'
    def is_int32(self):
        return self.__dtype == 'int32'
    def is_int64(self):
        return self.__dtype == 'int64'
    def is_uint8(self):
        return self.__dtype == 'uint8'
    def is_uint16(self):
        return self.__dtype == 'uint16'
    def is_uint32(self):
        return self.__dtype == 'uint32'
    def is_uint64(self):
        return self.__dtype == 'uint64'

    def elements_to_bytes(self, n_elements: int) -> int:
        return n_elements * self.itemsize

    @property
    def dtype(self):
        return self.__dtype

    @property
    def itemsize(self):
        if self.dtype in ["int64", "uint64", "float64"]:
            return 8
        elif self.dtype in ["int32", "uint32", "float32"]:
            return 4
        elif self.dtype in ["int16", "uint16", "float16"]:
            return 2
        return 1

@dataclass
class LazyStorageKind:
    data_type: DataType

@dataclass
class LazyStorage:
    load: Callable[[int, int], NDArray]
    kind: LazyStorageKind
    description: str

class QuantType(Enum):
    NONE  = 0
    INT16 = 1
    INT8  = 2
    INT4  = 3

    def to_type(self, key):
        if type(key) is BlockType:
            key = key.dtype
        else:
            key = key.lower()
            if key == 'f32':
                key = 'float32'
            elif key == 'f64':
                key = 'float64'
            elif key == 'i32':
                key = 'int32'
            elif key == 'i16':
                key = 'int16'
            elif key == 'i8':
                key = 'int8'

        if key == 'int16':
            return QuantType.INT16
        elif key == 'int8':
            return QuantType.INT8
        elif key == 'int4':
            return QuantType.INT4
        else:
            return QuantType.NONE

    def to_id(self, key):
        return QuantType.to_type(key).value

def print_ndarray(arr, dump_file_path=None):
    cols = arr.shape[-1]
    arr = arr.reshape(arr.size // cols, cols)
    if dump_file_path:
        ofile = open(dump_file_path, 'w')
        maxl = arr.shape[0]
    else:
        ofile = sys.stdout
        maxl = 10
    maxr = 20

    if arr.ndim == 1:
        arr = [arr]
    for i, row in enumerate(arr):
        if i >= maxl and i < len(arr) - 1:
            if i == maxl:
                print("      ...")
            continue
        if len(row) <= maxr:
            sarr = ['%9.6f' % v for v in row]
        else:
            sarr = ['%9.6f' % v for v in row[:maxr-2]]
            sarr.append('...')
            sarr += ['%9.6f' % v for v in row[-2:]]
        print('%5d:[%s]' % (i, ', '.join(sarr)), file=ofile)

@dataclass
class TensorLoader:
    _load: Callable[[], Any]
    shape: list[int]
    data_type: DataType
    description: str
    data: np.ndarray = None

    def load(self) -> Any:
        if self.data is None:
            tsr = self._load()
            assert tsr.dtype == self.data_type or (self.data_type.dtype == tsr.data_type.dtype), \
                (self.data_type, tsr.data_type, self.description)
            self.data = tsr
        return self.data

    def astype(self, data_type: DataType) -> Any:
        def load() -> Any:
            return self.load().astype(data_type)
        return TensorLoader(load, self.shape, data_type, f'convert({data_type}) {self.description}')

    @staticmethod
    def quantize(arr:np.ndarray, data_type: DataType, group_size=64):
        if (arr.size % group_size) != 0:
            raise ValueError(f"tensor's size:{arr.size} is not multiple of group_size:{group_size}")

        arr = arr.copy()

        shape = arr.shape
        if data_type.is_int8():
            scale_range = 127
        elif data_type.is_int16:
            scale_range = int(math.sqrt((1<<31) / group_size))
        else:
            raise ValueError("Unsupported quantization type")

        arr = arr.reshape(arr.size // group_size, group_size)

        scales = np.abs(arr).max(axis=1) / scale_range
        arr /= scales[:, None]
        if data_type.is_int8():
            arr = arr.astype(np.int8)
        else:
            arr = arr.astype(np.int16)

        scales_shape = list(shape)
        scales_shape[-1] = shape[-1] // group_size
        scales = scales.astype(np.float32).reshape(*scales_shape)

        return arr.reshape(shape), scales.astype(np.float32)

class LazyUnpickler(pickle.Unpickler):
    def __init__(self, fp: IO[bytes], data_base_path: str, zip_file: zipfile.ZipFile):
        super().__init__(fp)
        self.data_base_path = data_base_path
        self.zip_file = zip_file

    def persistent_load(self, pid: Any) -> Any:
        assert pid[0] == 'storage'
        assert isinstance(pid[1], LazyStorageKind)
        data_type = pid[1].data_type
        filename = f'{self.data_base_path}/{pid[2]}'
        info = self.zip_file.getinfo(filename)

        def load(offset: int, elm_count: int) -> NDArray:
            fp = self.zip_file.open(info)
            itemsize = data_type.itemsize
            fp.seek(offset * itemsize)
            size = elm_count * itemsize
            data = fp.read(size)
            assert len(data) == size
            return np.frombuffer(data, data_type.dtype)
        description = f'storage data_type={data_type} path-in-zip={filename} path={self.zip_file.filename}'
        return LazyStorage(load=load, kind=pid[1], description=description)

    @staticmethod
    def rebuild_tensor_v2(storage: Any, storage_offset: Any, size: Any, stride: Any,
                               requires_grad: Any, backward_hooks: Any, metadata: Any = None) -> TensorLoader:
        assert isinstance(storage, LazyStorage)

        def load():
            elm_count = stride[0] * size[0]
            return storage.load(storage_offset, elm_count).reshape(size)
        description = f'pickled storage_offset={storage_offset} in {storage.description}'
        return TensorLoader(load, list(size), storage.kind.data_type, description)

    @staticmethod
    def rebuild_from_type_v2(func, new_type, args, state):
        return func(*args)

    CLASSES: dict[tuple[str, str], Any] = {
        # getattr used here as a workaround for mypy not being smart enough to detrmine
        # the staticmethods have a __func__ attribute.
        ('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'),
        ('torch._utils',  '_rebuild_tensor_v2'):    getattr(rebuild_tensor_v2,    '__func__'),
        ('torch', 'FloatStorage'):                  LazyStorageKind(DataType("float32")),
        ('torch', 'HalfStorage'):                   LazyStorageKind(DataType("float16")),
        ('torch', 'Tensor'):                        TensorLoader,
    }

    def find_class(self, module: str, name: str) -> Any:
        if not module.startswith('torch'):
            return super().find_class(module, name)
        return self.CLASSES[(module, name)]


def nth_multifile_path(path: Path, n: int) -> Path | None:
    '''Given any path belonging to a multi-file model (e.g. foo.bin.1), return
    the nth path in the model.
    '''
    # Support the following patterns:
    patterns: list[tuple[str, str]] = [
        # - x.00.pth, x.01.pth, etc.
        (r'\.[0-9]{2}\.pth$', f'.{n:02}.pth'),
        # - x-00001-of-00002.bin, x-00002-of-00002.bin, etc.
        (r'-[0-9]{5}-of-(.*)$', fr'-{n:05}-of-\1'),
        # x.bin, x.bin.1, etc.
        (r'(\.[0-9]+)?$', r'\1' if n == 0 else fr'\1.{n}')
    ]
    for regex, replacement in patterns:
        if re.search(regex, path.name):
            new_path = path.with_name(re.sub(regex, replacement, path.name))
            if new_path.exists():
                return new_path
    return None

def find_multifile_paths(path: Path) -> list[Path]:
    ret: list[Path] = []
    for i in itertools.count():
        nth_path = nth_multifile_path(path, i)
        if nth_path is None:
            break
        ret.append(nth_path)
    if not ret:
        return [path]
    return ret

@dataclass
class ModelConfig:
    name:str                = ""
    model_type:ModelType    = ModelType.LLAMA
    act_type:ActType        = ActType.SWIGLU
    quant_type:QuantType    = QuantType.NONE

    vocab_size:         int = 0 
    dim:                int = 0
    hidden_dim:         int = 0
    n_heads:            int = 0
    n_kv_heads:         int = 0
    n_layers:           int = 0
    max_length:         int = 0

    bos_token_id:       int = 0 
    eos_token_id:       int = 0
    pad_token_id:       int = 0

    rms_norm_eps:       float = 0.
    rope_theta:         float = 10000.0
    #rope_scaling:      float = 0.
    quant_group_size:   int = 64;

    def load(self, model_path, quant_type:QuantType, group_size:int) -> bool:
        if model_path.is_dir():
            model_path = model_path / "config.json"
        with open(model_path, 'r') as f:
            conf = json.load(f)

        self.quant_type = quant_type
        self.quant_group_size = group_size

        kk_mapping = {
            '_name_or_path':       'name',
            'vocab_size':          'vocab_size',
            'hidden_size':         'dim',
            'intermediate_size':   'hidden_dim',
            'num_attention_heads': 'n_heads',
            'num_key_value_heads': 'n_kv_heads',
            'num_hidden_layers':   'n_layers',
            "hidden_act":          'act_type',
            'max_position_embeddings': 'max_length',
        }
        for k, v in conf.items():
            if k in self.__dict__:
                if type(self.__dict__[k]) == type(v):
                    self.__dict__[k] = v
                elif type(type(self.__dict__[k])) is enum.EnumMeta:
                    self.__dict__[k] = type(self.__dict__[k])[v.upper()]
            elif k in kk_mapping:
                self.__dict__[kk_mapping[k]] = v

        return True

    def serialize_as_flf(self, is_little_endian:bool=True) -> bytes:
        obs = io.BytesIO()
        out = FLFWriter(obs, is_little_endian)
        for i, (k, v) in enumerate(self.__dict__.items()):
            if type(type(v)) is enum.EnumMeta:
                v = v.value
            if type(v) is int:
                out.dump_named_int32(k, v)
            elif type(v) is float:
                out.dump_named_float32(k, v)
            elif type(v) is str:
                out.dump_named_string(k, v)
        return obs.getvalue()

    def print_summary(self):
        kn = 5 + max([len(k) for k in self.__dict__])
        for k, v in self.__dict__.items():
            print(f'%*s: \x1b[33m{v}\x1b[0m' % (kn, k))

FLM_VERSION  = 0x01000000 # 1.0.0
FLM_FILE_EXT = "flm"

class BlockType(Enum):
    BASE_ITEM     = 0
    DICT          = 1
    TENSOR        = 2
    ARRAY         = 3
    STRING        = 4
    STRING_ARRAY  = 5

class BlockDataType(Enum):
    NONE    =  0

    CHAR    =  1
    INT8    =  1
    INT16   =  2
    INT32   =  3
    INT64   =  4

    UINT8   =  5
    UINT16  =  6
    UINT32  =  7
    UINT64  =  8

    FLOAT16 = 10 
    FLOAT32 = 11 
    FLOAT64 = 12

    BLOCK   = 15

class TensorType(Enum):
    NONE              =  0

    TOKEN_EMBD_TABLE  =  1
    OUTPUT_NORM       =  2
    CLASSIFIER        =  3

    LAYER_TYPE        = 16
    LAYER_INPUT_NORM  = 17
    LAYER_ATTN_Q      = 18
    LAYER_ATTN_K      = 19
    LAYER_ATTN_V      = 20
    LAYER_ATTN_O      = 21
    LAYER_MLP_GATE    = 22
    LAYER_MLP_UP      = 23
    LAYER_MLP_DOWN    = 24
    LAYER_POST_NORM   = 25

@dataclass
class Vocab:
    typ:    VocabType
    texts:  list[str]
    scores: list[float]
    types:  list[int]

    @property
    def vocab_size(self):
        return len(self.texts)

class FLFWriter:
    def __init__(self, output_file : str | Path | io.BytesIO | io.TextIOWrapper, is_little_endian:bool=True):
        self.is_little_endian = is_little_endian
        if type(output_file) in (io.BytesIO, io.TextIOWrapper):
            self.ofile = output_file
        else:
            self.ofile = open(output_file, 'wb')

        self.type_name_mapping = {
            'CHAR':   'b',
            'INT8':   'b',
            'UINT8':  'B',
            'INT16':  'h',
            'UINT16': 'H',
            'INT32':  'i',
            'UINT32': 'I',
            'INT64':  'q',
            'UINT64': 'Q',
            'FLOAT32':'f',
            'FLOAT64':'d',
        }

    def pack_item(self, type_name:str, value):
        typ = self.type_name_mapping.get(type_name.upper(), type_name)
        endseq = '<' if self.is_little_endian else '>'
        return struct.pack(f'{endseq}{typ}', value)

    def pack_ndarray(self, arr:np.ndarray):
        data_type = BlockDataType[str(arr.dtype).upper()]
        if not self.is_little_endian:
            if arr.dtype == 'int16':
                arr.astype('>i2')
            elif arr.dtype == 'int32':
                arr.astype('>i4')
            elif arr.dtype == 'int64':
                arr.astype('>i8')
            elif arr.dtype == 'uint16':
                arr.astype('>u2')
            elif arr.dtype == 'uint32':
                arr.astype('>u4')
            elif arr.dtype == 'uint64':
                arr.astype('>u8')
            elif arr.dtype == 'float16':
                arr.astype('>f2')
            elif arr.dtype == 'float32':
                arr.astype('>f4')
            elif arr.dtype == 'float64':
                arr.astype('>f8')
            else:
                raise ValueError(f"unsupported ndarray data type:{arr.dtype}")
        return arr.tobytes()

    def dump_item(self, type_name, value):
        data = self.pack_item(type_name, value)
        self.ofile.write(data)
        return len(data)

    def dump_char(self, value) -> int:
        if type(value) is not bytes:
            value = value.encode()
        return self.dump_int8(value[0])

    def dump_int8(self, value:int) -> int:
        return self.dump_item('int8', value)
    def dump_int16(self, value:int) -> int:
        return self.dump_item('int16', value)
    def dump_int32(self, value:int) -> int:
        return self.dump_item('int32', value)
    def dump_int64(self, value:int) -> int:
        return self.dump_item('int64', value)
    def dump_uint8(self, value:int) -> int:
        return self.dump_item('uint8', value)
    def dump_uint16(self, value:int) -> int:
        return self.dump_item('uint16', value)
    def dump_uint32(self, value:int) -> int:
        return self.dump_item('uint32', value)
    def dump_uint64(self, value:int) -> int:
        return self.dump_item('uint64', value)
    def dump_float32(self, value:float) -> int:
        return self.dump_item('float32', value)
    def dump_float64(self, value:float) -> int:
        return self.dump_item('float64', value)
    def dump_bytes(self, value:bytes) -> int:
        return self.ofile.write(value)
    def dump_padding(self, pad_size:int) -> int:
        if pad_size > 0:
            self.ofile.write(b'\x00' * pad_size)
        return pad_size

    def dump_named_int8(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('int8',   value), BlockDataType.INT8)
    def dump_named_int16(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('int16',  value), BlockDataType.INT16)
    def dump_named_int32(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('int32',  value), BlockDataType.INT32)
    def dump_named_int64(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('int64',  value), BlockDataType.INT64)
    def dump_named_uint8(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('uint8',  value), BlockDataType.UINT8)
    def dump_named_uint16(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('uint16', value), BlockDataType.UINT16)
    def dump_named_uint32(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('uint32', value), BlockDataType.UINT32)
    def dump_named_uint64(self, name:str, value:int) -> int:
        return self.dump_named_item(name, self.pack_item('uint64', value), BlockDataType.UINT64)
    def dump_named_float32(self, name:str, value:float) -> int:
        return self.dump_named_item(name, self.pack_item('float32', value), BlockDataType.FLOAT32)
    def dump_named_float64(self, name:str, value:float) -> int:
        return self.dump_named_item(name, self.pack_item('float64', value), BlockDataType.FLOAT64)
    def dump_named_string(self, name:str, data:str, enctype='utf-8') -> int:
        data = data.encode(enctype) + b'\x00';
        return self.dump_block(name, data, BlockType.STRING, BlockDataType.CHAR)

    def dump_named_ndarray(self, name:str, arr:np.ndarray, align_size:int=8) -> int:
        data = self.pack_ndarray(arr)
        dtype = BlockDataType[str(arr.dtype).upper()]
        return self.dump_named_bytes_array(name, data, dtype, align_size)

    def dump_named_item(self, name:str, data:bytes, data_type:BlockDataType) -> int:
        """
        struct ItemHeader {
            uint8_t   block_type;
            uint8_t   data_type;
            uint8_t   header_size;
            uint8_t   data_size;
            uint32_t  __padding__; // only for int64, uint64 and float64
            union {
                int8_t      i8;
                int16_t     i16;
                int32_t     i32;
                uint8_t     u8;
                uint16_t    u16;
                uint32_t    u32;
                float       f32;
                int64_t     i64;
                uint64_t    u64;
                double      f64;
            };
            };
            char        name[2];
        }
        """
        name = name.encode('utf-8') + b'\x00'
        if len(name) > 240:
            raise ValueError(f"name:[{name}] is too long")

        data_size = len(data)

        if data_size <= 4:
            item_size = 8 + len(name)
        else:
            item_size = 16 + len(name)

        header_size = (item_size + 7) & ~7;

        self.dump_uint8(BlockType.BASE_ITEM.value)
        self.dump_uint8(data_type.value)
        self.dump_uint8(header_size)
        self.dump_uint8(data_size)
        if data_size > 4:
            self.dump_int32(0)
        self.dump_bytes(data)
        self.dump_padding(4 - data_size if data_size <= 4 else 8 - data_size)
        self.dump_bytes(name)
        self.dump_padding(header_size - item_size)
        return header_size

    @staticmethod
    def padding_align(data:bytes, align_size:int) -> bytes:
        pad_size = ((len(data) + align_size - 1) & ~(align_size - 1)) - len(data)
        if pad_size > 0:
            data += b'\x00' * pad_size
        return data

    def dump_named_string_array(self, name:str, arr:list[str], enctype="utf-8", align_size=8) -> int:
        offsets = []
        data    = b''
        offset  = 0
        for i, text in enumerate(arr):
            offsets.append(offset)
            bs = self.padding_align(text.encode(enctype) + b'\x00', 8)
            offset += len(bs)
            data += bs
        str_size = len(data)
        data = self.pack_item("uint64", len(arr)) + self.pack_item("uint64", str_size) + self.pack_ndarray(np.array(offsets), dtype=np.uint64).tobytes() + data
        return self.dump_block(name, data, BlockType.STRING_ARRAY, BlockDataType.CHAR, align_size)

    def dump_named_bytes_array(self, name:str, data:bytes, data_type:BlockDataType, align_size=8) -> int:
        return self.dump_block(name, data, BlockType.ARRAY, data_type, align_size)

    def dump_named_tensor(self, name:str, tensor:np.ndarray, scales:np.ndarray=None, tensor_type:TensorType=TensorType.NONE, layer_id:int=0, align_size=64) -> int:
        """
        struct TensorHeader {
            uint32_t   shape[4];
            uint16_t   tensor_type;
            uint16_t   layer_id;
            uint32_t   scales_size;
        };
        """

        data = self.pack_ndarray(tensor)
        dtype = BlockDataType[str(tensor.dtype).upper()]
        if scales is not None:
            scales_size = scales.size
            scales_data = self.pack_ndarray(scales)
            data += scales_data
        else:
            scales_size = 0

        shape = [0] * 4
        for i, v in enumerate(tensor.shape):
            shape[i] = v

        header_data  = self.pack_ndarray(np.array(shape, dtype=np.uint32))
        header_data += self.pack_item("uint16", tensor_type.value)
        header_data += self.pack_item("uint16", layer_id)
        header_data += self.pack_item("uint32", scales_size)

        return self.dump_block(name, data, BlockType.TENSOR, dtype, align_size, header_data)

    def dump_block(self, name:str, data:bytes, block_type:BlockType, data_type:BlockDataType=BlockDataType.NONE, align_size=8, header_data=None) -> int:
        """
        struct BlockHeader {
            uint8_t    block_type;
            uint8_t    data_type;
            uint8_t    header_size;
            uint8_t    header_data_size;
            uint8_t    name_offset;
            uint8_t    name_size;
            uint16_t   pad_size;
            uint64_t   data_size;
            char       header_data[...]; // optional
            char       name[2];
        }
        """
        try:
            file_pos = self.ofile.tell()
        except:
            file_pos = 0

        if name:
            name = name.encode('utf-8')
            name_size = len(name)
            name = name + b'\x00'
            if len(name) > 240:
                raise ValueError(f"name:[{name}] is too long")
        else:
            name = b''
            name_size = 0

        if not header_data:
            header_data = b''
        else:
            header_data = self.padding_align(header_data, 8)

        data_size = len(data)
        name_offset = 16 + len(header_data)

        header_size   = name_offset + len(name)
        head_pad_size = (file_pos + header_size) % align_size
        if head_pad_size > 0:
            head_pad_size = align_size - head_pad_size 
            header_size += head_pad_size

        block_size = (header_size + data_size + align_size - 1) & ~(align_size-1)

        tail_pad_size = block_size - header_size - data_size

        self.dump_uint8(block_type.value)
        self.dump_uint8(data_type.value)
        self.dump_uint8(header_size)
        self.dump_uint8(len(header_data))
        self.dump_uint8(name_offset)
        self.dump_uint8(name_size)
        self.dump_uint16(tail_pad_size)
        self.dump_uint64(data_size)
        if header_data:
            self.dump_bytes(header_data)
        if name:
            self.dump_bytes(name)
        self.dump_padding(head_pad_size)
        self.dump_bytes(data)
        self.dump_padding(tail_pad_size)
        return block_size

def load_bpe_vocab(fname_tokenizer: Path, fname_added_tokens: Path | None) -> Vocab:
    texts:list[str]    = []
    scores:list[float] = []
    types:list[int]    = []

    with open(str(tokenizer_path), encoding="utf-8") as f:
        from transformers.models.gpt2 import tokenization_gpt2
        bpe_tokenizer = json.load(f)
        reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.items()}
        for i, _ in enumerate(tokenizer):
            texts.append(reverse_vocab[i])
            scores.append(0.)
            types.append(TokenType.NORMAL.value)

    if fname_added_tokens:
        with open(fname_added_tokens, encoding='utf-8') as f:
            added_tokens = json.load(f)
    else:
        fname_tokenizer_json = fname_tokenizer.parent / 'tokenizer.json'
        if not tokenizer_json_file.is_file():
            added_tokens = {}
        else:
            with open(fname_tokenizer_json, encoding='utf-8') as f:
                tokenizer_json = json.load(f)
            added_tokens = {item['content']: item['id']
                for item in tokenizer_json.get('added_tokens', [])
                if item['content'] not in self.bpe_tokenizer}

    vocab_size = len(texts)
    if added_tokens:
        added_items = sorted([(tokid, text) for text, tokid in added_tokens.items() if tokid >= vocab_size], key=lambda item: text[0])
        if added_items[0][0] != len(vocabs) or added_items[-1][0] != added_items[0][0] + len(vocabs) - 1:
            raise ValueError("Invalid added tokens")
        for tokid, text in added_items:
            texts.append(text)
            scores.append(-1000.)
            types.append(TokenType.CONTROL.value)

    for i, text in enumerate(texts):
        if type(text) is bytes:
            texts[i] = text.decode()
    return Vocab(VocabType.BPE, texts, scores, types)

def load_spm_vocab(fname_tokenizer: Path, fname_added_tokens: Path | None) -> Vocab:
    texts = []
    scores = []
    types = []

    tokenizer = SentencePieceProcessor(str(fname_tokenizer))
    for i in range(tokenizer.vocab_size()):
        piece        = tokenizer.id_to_piece(i)
        text: bytes  = piece.encode("utf-8")
        score: float = tokenizer.get_score(i)

        toktype = TokenType.NORMAL
        if tokenizer.is_unknown(i):
            toktype = TokenType.UNKNOWN
        if tokenizer.is_control(i):
            toktype = TokenType.CONTROL

        if tokenizer.is_unused(i):
            toktype = TokenType.UNUSED
        if tokenizer.is_byte(i):
            toktype = TokenType.BYTE

        texts.append(text)
        scores.append(score)
        types.append(toktype.value)

    vocab_size = len(texts)
    if fname_added_tokens is not None:
        with open(fname_added_tokens, encoding="utf-8") as f:
            added_tokens = json.load(f)
        added_items = sorted([(tokid, text) for text, tokid in added_tokens if tokid >= vocab_size], key=lambda item:item[0])
        if added_items[0][0] != vocab_size or added_items[-1][0] != added_items[0][0] + len(added_items) - 1:
            raise ValueError("Invalid added tokens")

        for tokid, text in added_items:
            texts.append(text)
            scores.append(-1000.)
            types.append(TokenType.USER_DEFINED.value)

    for i, text in enumerate(texts):
        if type(text) is bytes:
            texts[i] = text.decode()
    return Vocab(VocabType.SPM, texts, scores, types)

@dataclass
class Tokenizer:
    vocab_type: str = None
    vocab: Vocab    = None
    special_tokens  = {}

    def load(self, path: Path, vocab_type) -> bool:
        if not self.load_vocab(path, vocab_type):
            return False
        if not self.load_special_tokens(path):
            return False
        return True 

    def load_vocab(self, path: Path, vocab_type: str | None) -> bool:
        if path.is_dir():
            if not vocab_type:
                if (path / 'vocab.json').exists():
                    vocab_type = 'bpe'
                elif (path / 'tokenizer.model').exists():
                    vocab_type = 'spm'
            if vocab_type == 'bpe':
                path = path / 'vocab.json'
            elif vocab_type == 'spm':
                path = path / 'tokenizer.model'
            else:
                raise ValueError("no valid vocab")

        if not path.exists():
            raise ValueError(f"File not found:[{path}]")

        self.vocab_type = vocab_type

        print(f"Loading vocab file '{path}', type '{vocab_type}'")

        added_tokens_path = path.parent / "added_tokens.json"
        if not added_tokens_path.exists():
            added_tokens_path = None
        if vocab_type == "bpe":
            self.vocab = load_bpe_vocab(path, added_tokens_path)
        elif vocab_type == "spm":
            self.vocab = load_spm_vocab(path, added_tokens_path)
        else:
            raise ValueError(f"Unsupported vocabulary type {vocab_type}")
        return True

    def load_special_tokens(self, path: Path) -> dict:
        special_names = ['bos', 'eos', 'unk', 'sep', 'pad']

        def try_load_from_tokenizer_config() -> bool:
            tokenizer_path = path / 'tokenizer.json'
            if not tokenizer_path.exists():
                return False

            with open(tokenizer_path) as f:
                tokenizer = json.load(f)

            cnt = 0

            if self.vocab_type == 'bpe':
                merges = tokenizer.get('model', {}).get('merges')
                if isinstance(merges, list) and len(merges) > 0 and isinstance(merges[0], str):
                    self.merges = merges
            conf_path = path / 'tokenizer_config.json'
            added_tokens = tokenizer.get('added_tokens')
            if added_tokens is None or not conf_path.is_file():
                return True

            with open(conf_path, encoding = 'utf-8') as f:
                conf = json.load(f)
            for name in special_names:
                item = conf.get(f'{name}_token')

                if isinstance(item, str):
                    content = entry
                elif isinstance(entry, dict):
                    entry_content = entry.get('content')
                    if not isinstance(entry_content, str):
                        continue
                    content = entry_content
                else:
                    continue
                tokid = next((tok.get('id') for tok in added_tokens if atok.get('content') == content), None)
                self.special_tokens[name] = tokid
            return True

        def try_load_from_config_json() -> bool:
            conf_path = path / 'config.json'
            if not conf_path.exists():
                return False

            with open(conf_path) as f:
                conf = json.load(f)

            for name in special_names:
                key = f'{name}_token_id'
                val = conf.get(key, -1)
                if val >= 0:
                    self.special_tokens[key] = val
            return True

        if not try_load_from_tokenizer_config():
            try_load_from_config_json()

        return True

    def serialize_as_flf(self, is_little_endian:bool=True) -> bytes:
        """
        struct Tokenizer {
            uint32_t vocab_type;
            uint32_t conn_tag_pos; // offset in text_data
            int32_t  special_tokens[SpecialTokenType.MAX]; // -1表示为设置
            uint32_t vocab_size;
            uint32_t text_data_size;
            struct Token {
                uint32_t index_text_pos; // offset in text_data
                uint32_t show_text_pos;  // offset in text_data
                uint32_t token_type;
                float    score;
            } tokens[vocab_size];
            char text_data[text_data_size]
        }
        """

        obs = io.BytesIO()

        @dataclass
        class Token:
            index_text:int
            show_text:int
            score:float
            tktype:int

        enctype = 'utf-8'
        out = FLFWriter(obs, is_little_endian)
        def enc_text(text:str, align_size=8):
            return out.padding_align(text.encode(enctype) + b'\x00', align_size)

        tokn_data = b''
        text_data = b''

        conn_tag = "▁"
        conn_num = 0
        for i, text in enumerate(self.vocab.texts):
            score = self.vocab.scores[i]
            ttype = self.vocab.types[i]

            index_text_pos = len(text_data)

            text_data += enc_text(text)
            if text.startswith(conn_tag):
                conn_num += 1
                show_text_pos = len(text_data)
                text_data += enc_text(" " + text[len(conn_tag):])
            else:
                show_text_pos = index_text_pos
            
            tokn_data += out.pack_item("int32",   index_text_pos)
            tokn_data += out.pack_item("int32",   show_text_pos)
            tokn_data += out.pack_item("int32",   ttype)
            tokn_data += out.pack_item("float32", score)

        conn_pos = len(text_data)
        text_data += enc_text(conn_tag)

        special_tokens = [-1] * SpecialTokenType.MAX.value
        for name, tokid in self.special_tokens.items():
            if name.endswith('_token_id'):
                name = name.split('_', 1)[0]
            special_tokens[SpecialTokenType[name.upper()].value] = tokid

        out.dump_uint32(self.vocab.typ.value)
        out.dump_uint32(conn_pos)
        for v in special_tokens:
            out.dump_item("int32", v)
        out.dump_uint32(self.vocab.vocab_size)
        out.dump_uint32(len(text_data))
        out.dump_bytes(tokn_data)
        out.dump_bytes(text_data)

        return obs.getvalue()

def permute_qk(weights: np.ndarray, n_head: int, n_head_kv: int) -> np.ndarray:
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

@dataclass
class ModelConverter:
    config:      ModelConfig             = ModelConfig()
    tokenizer:   Tokenizer               = Tokenizer()
    tensor_map:  dict[str, TensorLoader] = None

    def load(self, path: Path, vocab_type:str, quant_type:QuantType, quant_group_size:int) -> bool:
        if not path.is_dir():
            raise ValueError(f"path is not a directory:{path}")

        if not self.config.load(path, quant_type, quant_group_size):
            return False

        if not self.tokenizer.load(path, vocab_type):
            return False

        return self._load_model(path) 

    def print_summary(self):
        self.config.print_summary()

    def _load_model(self, path: Path) -> bool:
        globs = ["consolidated.00.pth", "pytorch_model-00001-of-*.bin", "*.pt", "pytorch_model.bin"]
        files = [file for glob in globs for file in path.glob(glob)]
        if not files:
            raise Exception(f"Can't find model in directory {path}")

        if len(files) > 1:
            raise Exception(f"Found multiple models in {path}, not sure which to pick: {files}")

        tensor_map = {}
        paths = find_multifile_paths(files[0])
        for path in paths:
            print(f"Loading model file {path} ...")
            r = self._load_torch_file(open(path, 'rb'), path)
            tensor_map = {**tensor_map, **r}
        self.tensor_map = tensor_map
        return True

    @staticmethod
    def _load_torch_file(outer_fp: IO[bytes], path: Path):
        zf = zipfile.ZipFile(outer_fp)
        pickle_paths = [name for name in zf.namelist() if name.endswith('.pkl')]
        assert len(pickle_paths) == 1, pickle_paths
        pickle_fp = zf.open(pickle_paths[0], 'r')
        unpickler = LazyUnpickler(pickle_fp,
                                  data_base_path=pickle_paths[0][:-4],
                                  zip_file=zf)
        model = unpickler.load()
        return dict(model.items())

    def dump(self, output_file_path: Path, out_type:str, is_little_endian:bool=True):
        if output_file_path.is_dir():
            output_file_path = output_file_path / f"{output_file_path.parent.name}-{out_type}.{FLMF_FILE_EXT}"

        print(f"Dumping to file:{output_file_path}")
        outf = FLFWriter(output_file_path, is_little_endian)

        self._dump_file_header(outf)

        data = self.config.serialize_as_flf(outf.is_little_endian)
        outf.dump_block("model_config", data, BlockType.DICT)

        data = self.tokenizer.serialize_as_flf(outf.is_little_endian)
        outf.dump_block("tokenizer", data, BlockType.DICT)

        self._dump_tensors(outf, out_type)

    def _dump_file_header(self, outf: FLFWriter) -> int:
        """
        struct FileHeader {
            uint32_t  file_tag;
            uint8_t   version1;
            uint8_t   version2;
            uint16_t  version3;
        }
        """
        FILE_TAG  = 0xFA571AEA
        outf.dump_uint32(0xFA571AEA)
        version = (1,0,0)
        outf.dump_uint8(version[0])
        outf.dump_uint8(version[1])
        outf.dump_uint16(version[2])

    def _dump_tensors(self, outf: FLFWriter, out_type:str):
        data_type = DataType(out_type)
        group_size = 64
        def get_tensor_size(tensor):
            r = 1
            for v in tensor.shape:
                r *= v
            return r
        total_size = 0
        ofile_size = 0

        for name, tensor_loader in self.tensor_map.items():
            tensor_type = TensorType.NONE
            tensor = tensor_loader.load()
            layer_id = 0
            if name == 'model.embed_tokens.weight':
                tensor_type = TensorType.TOKEN_EMBD_TABLE
            elif name == 'model.norm.weight':
                tensor_type = TensorType.OUTPUT_NORM
            elif name == 'lm_head.weight':
                tensor_type = TensorType.CLASSIFIER
            elif name.startswith('model.layers.'):
                layer_id = int(name.split('.')[2])
                layer_name = name.split('.', 3)[-1].rstrip('.weight')
                tensor_type_map = {
                    'input_layernorm':          TensorType.LAYER_INPUT_NORM,
                    'self_attn.q_proj':         TensorType.LAYER_ATTN_Q,
                    'self_attn.k_proj':         TensorType.LAYER_ATTN_K,
                    'self_attn.v_proj':         TensorType.LAYER_ATTN_V,
                    'self_attn.o_proj':         TensorType.LAYER_ATTN_O,
                    'post_attention_layernorm': TensorType.LAYER_POST_NORM,
                    'mlp.gate_proj':            TensorType.LAYER_MLP_GATE,
                    'mlp.up_proj':              TensorType.LAYER_MLP_UP,
                    'mlp.down_proj':            TensorType.LAYER_MLP_DOWN,
                }
                if layer_name not in tensor_type_map:
                    print(f"Unknown tensor name:[{name}], shape:{tensor_loader.shape}", file=sys.stderr)
                    continue

                tensor_type = tensor_type_map[layer_name]
                if tensor_type in [TensorType.LAYER_ATTN_Q, TensorType.LAYER_ATTN_K]:
                    tensor = permute_qk(tensor, self.config.n_heads, self.config.n_kv_heads)
            else:
                raise ValueError(f"Unknown tensor name:[{name}], shape:{tensor_loader.shape}")

            shape = tuple(tensor_loader.shape)
            needq = tensor_type not in [TensorType.TOKEN_EMBD_TABLE] and (data_type.is_int8() or data_type.is_int16()) and len(shape) > 1
            dtype = data_type if needq else DataType(BlockDataType.FLOAT32.name)
            dsize = get_tensor_size(tensor_loader)
            total_size += dsize * dtype.itemsize

            tips = f'Dumping tensor:\x1b[33m{name:50s}\x1b[0m\tshape:{str(shape):32s}\toutput_type:\x1b[33m{dtype.dtype:10s}\x1b[0m'
            if needq:
                tips += f'\tgroup_size:{group_size}'
            print(tips)

            if needq:
                tensor, scales = TensorLoader.quantize(tensor, data_type, group_size)
            else:
                tensor = tensor_loader.load().astype(np.float32)
                scales = None
            ofile_size += outf.dump_named_tensor(name, tensor, scales, tensor_type, layer_id)

        print(f"Total data size:\x1b[33m{total_size/(1<<30):.2f}\x1b[0mGB\toutput_size:\x1b[33m{ofile_size / (1<<30):.2f}\x1b[0mGB")

@click.command()
@click.option('--model-path', '-m', type=Path,                                 required=True,  help="Path of the model to be converted")
@click.option('--vocab-type', '-b', type=click.Choice(['bpe','spm']),          required=False, help="type of vocab, should be bpe or spm")
@click.option('--out-type',   '-t', type=click.Choice(['f32','int16','int8']), required=True,  help="output type should be f32, int16 or int8")
@click.option('--group-size', '-g', type=click.Choice([32,64,128,256]),        default=64,     help="group size for quantization")
@click.option('--output-path','-o', type=Path,                                 required=False, help="output path")
def main(model_path:Path, vocab_type:str, out_type:str, group_size:int, output_path:Path) -> int:
    print(f"model_path:{model_path}\tvocab_type:{vocab_type}", file=sys.stderr)

    qtype: QuantType = QuantType.NONE
    if out_type == 'int16':
        qtype = QuantType.INT16
    elif out_type == 'int8':
        qtype = QuantType.INT8

    model = ModelConverter()
    if not model.load(model_path, vocab_type, qtype, group_size):
        print("Failed to load model", file=sys.stderr)
        return -1

    model.print_summary()

    if not output_path:
        output_path = model_path if model_path.is_dir() else model_path.parent
    if output_path.is_dir():
        output_path = output_path / f'{out_type}.{FLM_FILE_EXT}'

    model.dump(output_path, out_type, True)
    return 0

if __name__ == '__main__':
    sys.exit(main())

