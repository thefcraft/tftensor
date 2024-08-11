use std::fs::File;
use std::io::{Read, Write};

use crate::core::dtype::DType;
use crate::core::ops::TensorLike;

use super::{Tensor, NumLike};

impl<T: NumLike> Tensor<T> {
    pub fn save(&self, path: &str){
        let mut file = File::create(path).unwrap();
        file.write_all(&Self::to_bytes(self)).unwrap();
    }
    pub fn load(path: &str) -> Self {
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();
        Self::from_bytes(buffer)
    }
    
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result: Vec<u8> = Vec::<u8>::with_capacity(8+8+(self.get_size()*self.get_dtype().byte_size())+(2*8*self.get_ndim())); 
        result.extend(self.get_size().to_ne_bytes()); // size
        result.extend(self.get_dtype().to_le_bytes()); // dtype
        for num in self.get_data().iter(){
            result.extend(num.bytes()); // data
        }
        for (&num_shape, &num_strides) in self.get_shape().iter().zip(self.get_strides().iter()) {
            result.extend(num_shape.to_ne_bytes());
            result.extend(num_strides.to_ne_bytes());
        }
        result
    }
    pub fn from_bytes(buffer: Vec<u8>) -> Self {
        let bytes = [
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ];
        let size = usize::from_ne_bytes(bytes);
        let bytes = [
            buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14], buffer[15],
        ];
        let dtype = DType::from_le_bytes(bytes);
        assert!(T::dtype() == dtype, "dtype mismatch the original dtype is different USE DTYPE: {}", dtype);
        
        let byte_size = dtype.byte_size();
        let mut data = Vec::<T>::with_capacity(size);
        
        // Convert bytes to T values
        let offset: usize = 16;
        for i in 0..size {
            let mut bytes = Vec::<u8>::with_capacity(byte_size);
            for j in 0..byte_size {
                bytes.push(buffer[offset + i*byte_size + j]);
            }
            data.push(T::from_le_bytes(bytes));
        }
        
        // Calculate number of T values based on byte length
        let num_values = buffer.len() - size*byte_size - 16;
        
        let mut shape = Vec::<usize>::new();
        let mut strides = Vec::<usize>::new();
        let mut i = 0;
        let offset = size*byte_size + 16;
        while i<num_values {
            let bytes = [
                buffer[offset + i],
                buffer[offset + i + 1],
                buffer[offset + i + 2],
                buffer[offset + i + 3],
                buffer[offset + i + 4],
                buffer[offset + i + 5],
                buffer[offset + i + 6],
                buffer[offset + i + 7],
            ];
            let num = usize::from_ne_bytes(bytes);
            shape.push(num);
            i+=8;
            let bytes = [
                buffer[offset + i],
                buffer[offset + i + 1],
                buffer[offset + i + 2],
                buffer[offset + i + 3],
                buffer[offset + i + 4],
                buffer[offset + i + 5],
                buffer[offset + i + 6],
                buffer[offset + i + 7],
            ];
            let num = usize::from_ne_bytes(bytes);
            strides.push(num);
            i+=8;
        }
        Self::from(data, shape, strides)
    }
}