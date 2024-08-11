from tftensor_core import Tensor_f32, Tensor_f64, Tensor_u8, Tensor_u16, Tensor_u32, Tensor_u64, Tensor_i8, Tensor_i16, Tensor_i32, Tensor_i64
from tftensor_core import TensorType

from typing import List, NewType, Union, Any, Optional, Tuple
import struct

try: import numpy # required to convert tfTensor to and from numpy
except ImportError: pass

float32 = NewType('float32', float)
float64 = NewType('float64', float)
int8 = NewType('int8', int)
int16 = NewType('int16', int)
int32 = NewType('int32', int)
int64 = NewType('int64', int)
uint8 = NewType('uint8', int)
uint16 = NewType('uint16', int)
uint32 = NewType('uint32', int)
uint64 = NewType('uint64', int)

def base_class(dtype)->TensorType:
    """
    Returns the corresponding TensorType class based on the dtype provided.

    :param dtype: Data type for which the TensorType class is needed.
    :return: The corresponding TensorType class.
    :raises TypeError: If the dtype is unsupported.
    """
    if dtype == float32: return Tensor_f32
    elif dtype == float64: return Tensor_f64
    elif dtype == int8: return Tensor_i8
    elif dtype == int16: return Tensor_i16
    elif dtype == int32: return Tensor_i32
    elif dtype == int64: return Tensor_i64
    elif dtype == uint8: return Tensor_u8
    elif dtype == uint16: return Tensor_u16
    elif dtype == uint32: return Tensor_u32
    elif dtype == uint64: return Tensor_u64
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")
class Tensor:
    """
    A Tensor class that wraps a TensorType instance and provides additional tensor operations.

    :param data: An instance of TensorType that holds the tensor data.
    """
    def __init__(self, data)->"Tensor":
        """
        Initialize the Tensor with a TensorType instance.

        :param data: TensorType instance to initialize the Tensor.
        """
        self.__data: TensorType = data
        
    @staticmethod
    def flatten_and_get_shape(nested_list):
        """
        Flatten a multi-dimensional list and return a tuple containing
        the flattened list and its shape. Raises an error if the list has
        irregular shapes across any axis.
        :param nested_list: The multi-dimensional list to flatten.
        :return: A tuple where the first element is the flattened list and
                 the second element is the shape of the original list.
        :raises ValueError: If the multi-dimensional list has inconsistent shapes.
        """
        def get_shape(lst):
            shape = []
            current = lst
            while isinstance(current, list):
                shape.append(len(current))
                if not current:
                    break
                current = current[0]
            return tuple(shape)
        def flatten(lst, shape):
            if not shape:
                return [lst]
            if len(lst) != shape[0]:
                raise ValueError("Inconsistent shapes in the nested list")
            flattened = []
            for item in lst:
                flattened.extend(flatten(item, shape[1:]))
            return flattened
        shape = get_shape(nested_list)
        flattened_list = flatten(nested_list, shape)
        return flattened_list, shape
    @staticmethod
    def unflatten_list(flattened_list, shape):
        """
        Convert a flattened list back into a nested list structure given the original shape.

        :param flattened_list: The flattened list to be converted back to a nested structure.
        :param shape: A tuple representing the shape of the original nested list.
        :return: The reconstructed nested list.
        :raises ValueError: If the length of the flattened list doesn't match the given shape.
        """
        if not shape:
            return flattened_list[0]
        
        def prod(iterable):
            """
            Calculate the product of all elements in the iterable.

            :param iterable: An iterable of numbers.
            :return: The product of all elements.
            """
            result = 1
            for x in iterable:
                result *= x
            return result

        if len(flattened_list) != prod(shape):
            raise ValueError("The length of the flattened list doesn't match the given shape")

        def nest(flat_list, shape):
            if len(shape) == 1:
                return flat_list[:shape[0]]
            size = prod(shape[1:])
            return [nest(flat_list[i*size:(i+1)*size], shape[1:]) for i in range(shape[0])]

        return nest(flattened_list, shape)     
    @classmethod
    def randn(cls, shape: List[int], seed: Optional[int] = None, dtype=float32)->"Tensor": 
        """
        Create a Tensor with random values drawn from simple linear congruential generator (LCG).

        :param shape: Shape of the tensor.
        :param seed: Seed for random number generation.
        :param dtype: Data type of the tensor.
        :return: A Tensor instance with random values.
        """
        data = base_class(dtype).randn(shape, seed)
        return cls(data)
    @classmethod
    def zeros(cls, shape: List[int], dtype=float32)->"Tensor": 
        """
        Create a Tensor filled with zeros.

        :param shape: Shape of the tensor.
        :param dtype: Data type of the tensor.
        :return: A Tensor instance filled with zeros.
        """
        data = base_class(dtype).zeros(shape)
        return cls(data)
    @classmethod
    def ones(cls, shape: List[int], dtype=float32)->"Tensor":
        """
        Create a Tensor filled with ones.

        :param shape: Shape of the tensor.
        :param dtype: Data type of the tensor.
        :return: A Tensor instance filled with ones.
        """
        data = base_class(dtype).ones(shape)
        return cls(data)
    @classmethod
    def full(cls, shape: List[int], fill_value, dtype=float32)->"Tensor": 
        """
        Create a Tensor filled with a specified value.

        :param shape: Shape of the tensor.
        :param fill_value: Value to fill the tensor with.
        :param dtype: Data type of the tensor.
        :return: A Tensor instance filled with the specified value.
        """
        data = base_class(dtype).full(shape, fill_value)
        return cls(data)
    @classmethod
    def arange(cls, start, stop, step = None, dtype=float32)->"Tensor":
        """
        Create a Tensor with values in a specified range.

        :param start: Starting value of the range.
        :param stop: End value of the range (exclusive).
        :param step: Step size between values.
        :param dtype: Data type of the tensor.
        :return: A Tensor instance with values in the specified range.
        """
        data = base_class(dtype).arange(start, stop, step)
        return cls(data)
    def str(self, spacing_size:int) -> str:
        return self.__data.str(spacing_size)
    def __repr__(self) -> str: 
        """
        Return a string representation of the Tensor.

        :return: String representation of the Tensor.
        """
        return self.__data.__repr__()
    def __len__(self): return self.size
    def __getitem__(self, index):
        core_index = []
        if isinstance(index, tuple):
            for i, idx in enumerate(index):
                if isinstance(idx, slice):
                    core_index.append([idx.start, idx.stop, idx.step])
                elif isinstance(idx, int):
                    core_index.append([idx, idx, None])
                else:
                    raise TypeError('Invalid index type.')
        elif isinstance(index, slice):
            core_index.append([index.start, index.stop, index.step])
        elif isinstance(index, int):
            core_index.append([index, index, None])
        else:
            raise TypeError('Index must be a tuple/slice/int.')
        while len(core_index) < self.ndim:
            core_index.append([None, None, None])
        
        single_value = True
        for i in core_index:
            if i[0] == None or i[1] == None or i[0] != i[1]: single_value = False
        if single_value:
            core_index[-1][1] = core_index[-1][0] + 1
            return self.__data.get_by_slicing(core_index).item()
        return Tensor(self.__data.get_by_slicing(core_index))
    def __setitem__(self, index, value):
        core_index = []
        if isinstance(index, tuple):
            for i, idx in enumerate(index):
                if isinstance(idx, slice):
                    core_index.append([idx.start, idx.stop, idx.step])
                elif isinstance(idx, int):
                    core_index.append([idx, idx, None])
                else:
                    raise TypeError('Invalid index type.')
        elif isinstance(index, slice):
            core_index.append([index.start, index.stop, index.step])
        elif isinstance(index, int):
            core_index.append([index, index, None])
        else:
            raise TypeError('Index must be a tuple/slice/int.')
        while len(core_index) < self.ndim:
            core_index.append([None, None, None])
        
        single_value = True
        for i in core_index:
            if i[0] == None or i[1] == None or i[0] != i[1]: single_value = False
        if single_value:
            core_index[-1][1] = core_index[-1][0] + 1
            if isinstance(value, int):
                value = [value]
            
        if isinstance(value, list):
            flattened, shape = self.flatten_and_get_shape(value)
            self.__data.set_by_slicing(core_index, flattened, shape)
        elif isinstance(value, Tensor):
            flattened, shape = value.flatten().to_list(), value.shape
            self.__data.set_by_slicing(core_index, flattened, shape)
        else:
            raise TypeError("Wrong Type is passed")
        
    def __add__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise addition of two Tensors.

        :param other: Another Tensor to add.
        :return: A new Tensor instance resulting from the addition.
        """
        return Tensor(self.__data.__add__(other.__data))
    def __iadd__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise in-place addition of another Tensor.

        :param other: Another Tensor to add in place.
        :return: The current Tensor instance after in-place addition.
        """
        self.__data.__iadd__(other.__data)
        return self
    def __sub__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise subtraction of two Tensors.

        :param other: Another Tensor to subtract.
        :return: A new Tensor instance resulting from the subtraction.
        """
        return Tensor(self.__data.__sub__(other.__data))
    def __isub__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise in-place subtraction of another Tensor.

        :param other: Another Tensor to subtract in place.
        :return: The current Tensor instance after in-place subtraction.
        """
        self.__data.__isub__(other.__data)
        return self
    def __mul__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise multiplication of two Tensors.

        :param other: Another Tensor to multiply.
        :return: A new Tensor instance resulting from the multiplication.
        """
        return Tensor(self.__data.__mul__(other.__data))
    def __imul__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise in-place multiplication of another Tensor.

        :param other: Another Tensor to multiply in place.
        :return: The current Tensor instance after in-place multiplication.
        """
        self.__data.__imul__(other.__data)
        return self
    def __div__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise division of two Tensors.

        :param other: Another Tensor to divide by.
        :return: A new Tensor instance resulting from the division.
        """
        return Tensor(self.__data.__div__(other.__data))
    def __idiv__(self, other: "Tensor")->"Tensor": 
        """
        Element-wise in-place division of another Tensor.

        :param other: Another Tensor to divide by in place.
        :return: The current Tensor instance after in-place division.
        """
        self.__data.__idiv__(other.__data)
        return self
    def __matmul__(self, other: "Tensor")->"Tensor": 
        """
        Matrix multiplication of two Tensors.

        :param other: Another Tensor to multiply with.
        :return: A new Tensor instance resulting from the matrix multiplication.
        """
        return Tensor(self.__data.__matmul__(other.__data))
    def matmul(self, other: "Tensor")->"Tensor": 
        """
        Matrix multiplication of two Tensors (alternative method using @).

        :param other: Another Tensor to multiply with.
        :return: A new Tensor instance resulting from the matrix multiplication.
        """
        return Tensor(self.__data.matmul(other.__data))
    def dot(self, other: "Tensor")->"Tensor": 
        """
        Dot product of two Tensors.

        :param other: Another Tensor to compute the dot product with.
        :return: A new Tensor instance resulting from the dot product.
        """
        return Tensor(self.__data.dot(other.__data))
    @property
    def shape(self)->List[int]: 
        """
        Get the shape of the Tensor.

        :return: The shape of the Tensor as a list of integers.
        """
        return self.__data.shape
    @property
    def strides(self)->List[int]: 
        """
        Get the strides of the Tensor.

        :return: The strides of the Tensor as a list of integers.
        """
        return self.__data.strides
    @property
    def ndim(self)->int: 
        """
        Get the number of dimensions of the Tensor.

        :return: The number of dimensions of the Tensor.
        """
        return self.__data.ndim
    @property
    def size(self)->int:
        """
        Get the total size of the Tensor.

        :return: The total number of elements in the Tensor.
        """
        return self.__data.size
    def reshape_(self, shape: List[int]): 
        """
        Reshape the Tensor in-place.

        :param shape: New shape for the Tensor.
        """
        self.__data.reshape_(shape)
    def reshape(self, shape: List[int])->"Tensor": 
        """
        Return a new Tensor with the specified shape.

        :param shape: New shape for the Tensor.
        :return: A new Tensor instance with the specified shape.
        """
        return Tensor(self.__data.reshape(shape))
    @property
    def T(self)->"Tensor": 
        """
        Transpose the Tensor.

        :return: A new Tensor instance that is the transpose of the current Tensor.
        """
        return Tensor(self.__data.T)
    def transpose(self, axis_order: List[int])->"Tensor": 
        """
        Transpose the Tensor according to the specified axis order.

        :param axis_order: List of integers specifying the new axis order.
        :return: A new Tensor instance with the specified axis order.
        """
        return Tensor(self.__data.transpose(axis_order))
    
    @classmethod
    def from_list(cls, value: List[Union[List, int]], dtype=float32)->"Tensor":
        """
        Create a Tensor from a flattened list of data, a shape, and a dtype.

        :param data_flatten: Flattened list of data values.
        :param shape: Shape of the tensor.
        :param dtype: Data type of the tensor.
        :return: A Tensor instance with the specified data, shape, and dtype.
        """
        flattened, shape = cls.flatten_and_get_shape(value)
        data = base_class(dtype)(flattened, shape)
        return cls(data)
    
    def to_list(self)->List[Union[List, int]]:
        buffer = self.to_bytes()
        size = struct.unpack('<Q',  buffer[:8])[0]  # Read the usize value as little endian
        def get_type_info(buffer):
            if buffer == 1: return 4, lambda byte_array: struct.unpack('<f', byte_array)[0]
            elif buffer == 2: return 8, lambda byte_array: struct.unpack('<d', byte_array)[0]
            elif buffer == 3: return 1, lambda byte_array: struct.unpack('<B', byte_array)[0]
            elif buffer == 4: return 2, lambda byte_array: struct.unpack('<H', byte_array)[0]
            elif buffer == 5: return 4, lambda byte_array: struct.unpack('<I', byte_array)[0]
            elif buffer == 6: return 8, lambda byte_array: struct.unpack('<Q', byte_array)[0]
            elif buffer == 7: return 1, lambda byte_array: struct.unpack('<b', byte_array)[0]
            elif buffer == 8: return 2, lambda byte_array: struct.unpack('<h', byte_array)[0]
            elif buffer == 9: return 4, lambda byte_array: struct.unpack('<i', byte_array)[0]
            elif buffer == 10: return 8, lambda byte_array: struct.unpack('<q', byte_array)[0]
            else: raise ValueError("Unsupported buffer type")
        byte_size, loader = get_type_info(struct.unpack('<Q', buffer[8:16])[0])
        offset = 16
        data = [loader(bytearray(buffer[offset + i*byte_size + j] 
                            for j in range(byte_size)))
            for i in range(size)]
        assert size == len(data), "Size mismatch"
        shape = []
        strides = []
        for i in range(16 + size * byte_size, len(buffer), 16):
            usize_bytes = buffer[i:i+8]
            num = struct.unpack('<Q', usize_bytes)[0]  # Read the usize value as little endian
            shape.append(num)
            usize_bytes = buffer[i+8:i+16]
            num = struct.unpack('<Q', usize_bytes)[0]  # Read the usize value as little endian
            strides.append(num)
        
        strides_original = [1]*len(shape)
        for i in reversed(range(0,len(shape)-1)):
            strides_original[i] = strides_original[i + 1] * shape[i + 1]
        if strides_original != strides:
            raise NotImplementedError("UnLikely")
        else:
            return self.unflatten_list(data, shape)
        
    def flatten(self)->"Tensor":
        return self.reshape(shape=[self.size])
    
    @classmethod
    def from_bytes(cls, buffer: bytearray)->"Tensor":
        """
        The function `frombytes` takes a bytearray buffer as input and returns a Tensor object.
        
        :param buffer: The `buffer` parameter in the `frombytes` method is expected to be a `bytearray`
        object containing the data that needs to be converted into a `Tensor` object
        :type buffer: bytearray
        """
        def get_dtype(buffer):
            if buffer == 1: return float32
            elif buffer == 2: return float64
            elif buffer == 3: return uint8
            elif buffer == 4: return uint16
            elif buffer == 5: return uint32
            elif buffer == 6: return uint64
            elif buffer == 7: return int8
            elif buffer == 8: return int16
            elif buffer == 9: return int32
            elif buffer == 10: return int64
            else: raise ValueError("Unsupported buffer type")
        dtype = get_dtype(struct.unpack('<Q', buffer[8:16])[0])
        data = base_class(dtype).frombytes(buffer)
        return cls(data)
    def to_bytes(self)->bytearray:
        """
        The function `tobytes` returns a `bytearray` object.
        """
        return bytearray(self.__data.tobytes())
    
    @classmethod
    def from_numpy(cls, ndarray: "numpy.ndarray")->"Tensor":
        shape = ndarray.shape
        strides = [1]*len(shape)
        for i in reversed(range(0,len(shape)-1)):
            strides[i] = strides[i + 1] * shape[i + 1]
        def get_dtype_int(dtype: numpy.dtype):
            if dtype == numpy.float32: return 1, lambda byte_array: struct.pack('<f', byte_array)
            elif dtype == numpy.float64: return 2, lambda byte_array: struct.pack('<d', byte_array)
            elif dtype == numpy.uint8: return 3, lambda byte_array: struct.pack('<B', byte_array)
            elif dtype == numpy.uint16: return 4, lambda byte_array: struct.pack('<H', byte_array)
            elif dtype == numpy.uint32: return 5, lambda byte_array: struct.pack('<I', byte_array)
            elif dtype == numpy.uint64: return 6, lambda byte_array: struct.pack('<Q', byte_array)
            elif dtype == numpy.int8: return 7, lambda byte_array: struct.pack('<b', byte_array)
            elif dtype == numpy.int16: return 8, lambda byte_array: struct.pack('<h', byte_array)
            elif dtype == numpy.int32: return 9, lambda byte_array: struct.pack('<i', byte_array)
            elif dtype == numpy.int64: return 10, lambda byte_array: struct.pack('<q', byte_array)
            else: raise ValueError("Unsupported dtype yet")
        didx, loader = get_dtype_int(ndarray.dtype)

        data = bytearray()
        data.extend(struct.pack('<Q', ndarray.size))
        data.extend(struct.pack('<Q', didx))

        for num in ndarray.flatten():
            data.extend(loader(num))

        for (num_shape, num_strides) in zip(shape, strides):
            data.extend(struct.pack('<Q', num_shape))
            data.extend(struct.pack('<Q', num_strides))
        return cls.from_bytes(data)
    def to_numpy(self)->"numpy.ndarray":
        buffer = self.to_bytes()
        size = struct.unpack('<Q',  buffer[:8])[0]  # Read the usize value as little endian
        def get_type_info(buffer):
            if buffer == 1: return numpy.float32, 4, lambda byte_array: struct.unpack('<f', byte_array)[0]
            elif buffer == 2: return numpy.float64, 8, lambda byte_array: struct.unpack('<d', byte_array)[0]
            elif buffer == 3: return numpy.uint8, 1, lambda byte_array: struct.unpack('<B', byte_array)[0]
            elif buffer == 4: return numpy.uint16, 2, lambda byte_array: struct.unpack('<H', byte_array)[0]
            elif buffer == 5: return numpy.uint32, 4, lambda byte_array: struct.unpack('<I', byte_array)[0]
            elif buffer == 6: return numpy.uint64, 8, lambda byte_array: struct.unpack('<Q', byte_array)[0]
            elif buffer == 7: return numpy.int8, 1, lambda byte_array: struct.unpack('<b', byte_array)[0]
            elif buffer == 8: return numpy.int16, 2, lambda byte_array: struct.unpack('<h', byte_array)[0]
            elif buffer == 9: return numpy.int32, 4, lambda byte_array: struct.unpack('<i', byte_array)[0]
            elif buffer == 10: return numpy.int64, 8, lambda byte_array: struct.unpack('<q', byte_array)[0]
            else: raise ValueError("Unsupported buffer type")
        dt, byte_size, loader = get_type_info(struct.unpack('<Q', buffer[8:16])[0])
        offset = 16
        data = [loader(bytearray(buffer[offset + i*byte_size + j] 
                            for j in range(byte_size)))
            for i in range(size)]
        assert size == len(data), "Size mismatch"
        shape = []
        strides = []
        for i in range(16 + size * byte_size, len(buffer), 16):
            usize_bytes = buffer[i:i+8]
            num = struct.unpack('<Q', usize_bytes)[0]  # Read the usize value as little endian
            shape.append(num)
            usize_bytes = buffer[i+8:i+16]
            num = struct.unpack('<Q', usize_bytes)[0]  # Read the usize value as little endian
            strides.append(num)
        arr = numpy.array(data, dtype=dt).reshape(shape)
        strides_original = [1]*len(shape)
        for i in reversed(range(0,len(shape)-1)):
            strides_original[i] = strides_original[i + 1] * shape[i + 1]
        if strides_original != strides:
            return arr.T
        else:
            return arr
        
        