use pyo3::prelude::*;
use pyo3::types::PyType;
mod core;
use core::{NumLike, PrintHelper, ReshapeTensor, Slice, SlicedTensor, TensorLike, TransposeTensor};


struct Tensor<T>
    where T: core::NumLike
{
    value: core::Tensor<T>
}

macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            inner: Tensor<$type>,
        }
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(data_flatten: Vec<$type>, shape: Vec<usize>) -> Self {
                Self {
                    inner: Tensor { value: core::Tensor::<$type>::new(data_flatten, shape) },
                }
            }
            
            #[classmethod]
            #[pyo3(signature = (shape, seed=None))]
            pub fn randn(_cls: &Bound<'_, PyType>, shape: Vec<usize>, seed: Option<u64>) -> PyResult<Self> {
                Ok(Self {
                    inner: Tensor { value: core::Tensor::<$type>::randn(shape, seed) },
                })
            }   
            #[classmethod]
            pub fn zeros(_cls: &Bound<'_, PyType>, shape: Vec<usize>) -> PyResult<Self> {
                Ok(Self {
                    inner: Tensor { value: core::Tensor::<$type>::zeros(shape) },
                })
            }
            #[classmethod]
            pub fn ones(_cls: &Bound<'_, PyType>, shape: Vec<usize>) -> PyResult<Self> {
                Ok(Self {
                    inner: Tensor { value: core::Tensor::<$type>::ones(shape) },
                })
            }
            #[classmethod]
            pub fn full(_cls: &Bound<'_, PyType>, shape: Vec<usize>, fill_value: $type) -> PyResult<Self> {
                Ok(Self {
                    inner: Tensor { value: core::Tensor::<$type>::full(shape, fill_value) },
                })
            }
            #[classmethod]
            #[pyo3(signature = (start, stop, step=NumLike::one()))]
            pub fn arange(_cls: &Bound<'_, PyType>, start: $type, stop: $type, step: $type) -> PyResult<Self> {
                Ok(Self {
                    inner: Tensor { value: core::Tensor::<$type>::arange(start, stop, step) },
                })
            }
            pub fn reprstr(&mut self, spacing_size:usize) -> String { 
                self.inner.value.reprstr(spacing_size)
            }
            pub fn __repr__(&mut self) -> String {
                self.inner.value.repr()
            }
            pub fn __add__(&mut self, other: &mut Self) -> PyResult<Self>{
                Ok(Self {
                    inner: Tensor { value: self.inner.value.add(&mut other.inner.value) },
                })
            }
            pub fn __iadd__(&mut self, other: &mut Self){
                self.inner.value.add_(&mut other.inner.value)
            }
            pub fn __sub__(&mut self, other: &mut Self) -> PyResult<Self>{
                Ok(Self {
                    inner: Tensor { value: self.inner.value.sub(&mut other.inner.value) },
                })
            }
            pub fn __isub__(&mut self, other: &mut Self){
                self.inner.value.sub_(&mut other.inner.value)
            }
            pub fn __mul__(&mut self, other: &mut Self) -> PyResult<Self>{
                Ok(Self {
                    inner: Tensor { value: self.inner.value.mul(&mut other.inner.value) },
                })
            }
            pub fn __imul__(&mut self, other: &mut Self){
                self.inner.value.mul_(&mut other.inner.value)
            }
            pub fn __div__(&mut self, other: &mut Self) -> PyResult<Self>{
                Ok(Self {
                    inner: Tensor { value: self.inner.value.div(&mut other.inner.value) },
                })
            }
            pub fn __idiv__(&mut self, other: &mut Self){
                self.inner.value.div_(&mut other.inner.value)
            }
            pub fn __matmul__(&mut self, other: &mut Self) -> PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.matmul(&mut other.inner.value) },
                })
            }
            pub fn matmul(&mut self, other: &mut Self) -> PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.matmul(&mut other.inner.value) },
                })
            }
            pub fn dot(&mut self, other: &mut Self) -> PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.dot(&mut other.inner.value) },
                })
            }
            #[getter]
            pub fn shape(&self)->Vec<usize>{
                self.inner.value.get_shape().clone()
            }
            #[getter]
            pub fn strides(&self)->Vec<usize>{
                self.inner.value.get_strides().clone()
            }
            #[getter]
            pub fn ndim(&self)->usize{
                *self.inner.value.get_ndim()
            }
            #[getter]
            pub fn size(&self)->usize{
                *self.inner.value.get_size()
            }
            pub fn item(&self)->PyResult<$type>{
                Ok(
                    *self.inner.value.item()
                )
            }
            #[pyo3(signature = (dim=None, keepdims=false))]
            pub fn mean(&mut self, dim: Option<usize>, keepdims: bool)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.mean(dim, keepdims) },
                })
            }
            #[pyo3(signature = (dim=None, keepdims=false))]
            pub fn sum(&mut self, dim: Option<usize>, keepdims: bool)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.sum(dim, keepdims) },
                })
            }
            #[pyo3(signature = (dim=None, keepdims=false))]
            pub fn max(&mut self, dim: Option<usize>, keepdims: bool)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.max(dim, keepdims) },
                })
            }
            #[pyo3(signature = (dim=None, keepdims=false))]
            pub fn min(&mut self, dim: Option<usize>, keepdims: bool)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.min(dim, keepdims) },
                })
            }
            #[pyo3(signature = (dim=None, keepdims=false))]
            pub fn argmax(&mut self, dim: Option<usize>, keepdims: bool)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.argmax(dim, keepdims) },
                })
            }
            #[pyo3(signature = (dim=None, keepdims=false))]
            pub fn argmin(&mut self, dim: Option<usize>, keepdims: bool)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: self.inner.value.argmin(dim, keepdims) },
                })
            }

            pub fn get_by_slicing(&mut self, slice: Vec<[Option<usize>; 3]>)->PyResult<Self>{
                let mut index = Vec::<Slice>::with_capacity(slice.len() as usize);
                for i in slice.iter(){
                    if let Some(start) = i[0] {
                        // index.push(Slice::new(i[0]..))
                        if let Some(end) = i[1] {
                            if let Some(step) = i[2] {
                                index.push(Slice::new(start..end).step(step))
                            }else{
                                index.push(Slice::new(start..end))
                            }
                        }else if let Some(step) = i[2] {
                            index.push(Slice::new(start..).step(step))
                        }else{
                            index.push(Slice::new(start..))
                        }
                    }else if let Some(end) = i[1] {
                        if let Some(step) = i[2] {
                            index.push(Slice::new(..end).step(step))
                        }else{
                            index.push(Slice::new(..end))
                        }
                    }else if let Some(step) = i[2] {
                        index.push(Slice::new(..).step(step))
                    }else{
                        index.push(Slice::new(..))
                    }
                }
                Ok(Self { 
                    inner: Tensor { value: SlicedTensor::from(&mut self.inner.value, &index).to_tensor() },
                })
            }
            pub fn set_by_slicing(&mut self, slice: Vec<[Option<usize>; 3]>, data_flatten: Vec<$type>, shape_for_assert: Vec<usize>){
                let mut index = Vec::<Slice>::with_capacity(slice.len() as usize);
                for i in slice.iter(){
                    if let Some(start) = i[0] {
                        // index.push(Slice::new(i[0]..))
                        if let Some(end) = i[1] {
                            if let Some(step) = i[2] {
                                index.push(Slice::new(start..end).step(step))
                            }else{
                                index.push(Slice::new(start..end))
                            }
                        }else if let Some(step) = i[2] {
                            index.push(Slice::new(start..).step(step))
                        }else{
                            index.push(Slice::new(start..))
                        }
                    }else if let Some(end) = i[1] {
                        if let Some(step) = i[2] {
                            index.push(Slice::new(..end).step(step))
                        }else{
                            index.push(Slice::new(..end))
                        }
                    }else if let Some(step) = i[2] {
                        index.push(Slice::new(..).step(step))
                    }else{
                        index.push(Slice::new(..))
                    }
                }
                let mut mut_tensor = SlicedTensor::from(&mut self.inner.value, &index);

                debug_assert_eq!(*mut_tensor.get_size(), data_flatten.len(), "size mismatch");
                debug_assert_eq!(*mut_tensor.get_ndim(), shape_for_assert.len(), "ndim mismatch");
                debug_assert_eq!(*mut_tensor.get_shape(), shape_for_assert, "shape mismatch");
                

                let mut indices: Vec<usize> = vec![0; *mut_tensor.get_ndim()];
                let ndim = *mut_tensor.get_ndim();
                let shape = mut_tensor.get_shape().clone();
                let mut data_flatten_index = 0;
                loop {
                    *mut_tensor.get_mut_by_indices(&indices) = data_flatten[data_flatten_index];
                    data_flatten_index += 1;
                    // Increment indices
                    let mut i = ndim - 1;
                    loop {
                        indices[i] += 1;
                        if indices[i] < shape[i] {
                            break;
                        }
                        indices[i] = 0;
                        if i == 0 {
                            return;
                        }
                        i -= 1;
                    }
                }
            }
            
            pub fn reshape_(&mut self, new_shape: Vec<usize>){
                self.inner.value.reshape_(new_shape)
            }
            pub fn reshape(&mut self, new_shape: Vec<usize>)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: ReshapeTensor::from(&mut self.inner.value, new_shape).to_tensor() },
                })
            }
            #[getter]
            pub fn T(&mut self)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: TransposeTensor::T(&mut self.inner.value).to_tensor() },
                })
            }
            pub fn transpose(&mut self, axis_order: Vec<usize>)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: TransposeTensor::from(&mut self.inner.value, axis_order).to_tensor() },
                })
            }
            #[classmethod]
            pub fn frombytes(_cls: &Bound<'_, PyType>, buffer: Vec<u8>)->PyResult<Self>{
                Ok(Self { 
                    inner: Tensor { value: core::Tensor::<$type>::from_bytes(buffer) },
                })
            }
            pub fn tobytes(&self)->Vec<u8>{
                self.inner.value.to_bytes()
            }
        }
    };
}

create_interface!(Tensor_f32, f32);
create_interface!(Tensor_f64, f64);
create_interface!(Tensor_i8, i8);
create_interface!(Tensor_i16, i16);
create_interface!(Tensor_i32, i32);
create_interface!(Tensor_i64, i64);
create_interface!(Tensor_u8, u8);
create_interface!(Tensor_u16, u16);
create_interface!(Tensor_u32, u32);
create_interface!(Tensor_u64, u64);

#[pyclass]
struct Tensor_f32_test{
    value: core::Tensor<f32>
}

#[pymethods]
impl Tensor_f32_test {
    #[new]
    pub fn new(data_flatten: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { 
            value:  core::Tensor::<f32>::new(data_flatten, shape)
        }
    }
    #[classmethod]
    #[pyo3(signature = (shape, seed=None))]
    pub fn randn(_cls: &Bound<'_, PyType>, shape: Vec<usize>, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self { value: 
            core::Tensor::<f32>::randn(shape, seed)
         })
    }
    #[classmethod]
    pub fn zeros(_cls: &Bound<'_, PyType>, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self { value: 
            core::Tensor::<f32>::zeros(shape)
         })
    }
    #[classmethod]
    pub fn ones(_cls: &Bound<'_, PyType>, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self { value: 
            core::Tensor::<f32>::ones(shape)
         })
    }
    #[classmethod]
    pub fn full(_cls: &Bound<'_, PyType>, shape: Vec<usize>, fill_value:f32) -> PyResult<Self> {
        Ok(Self { value: 
            core::Tensor::<f32>::full(shape, fill_value)
         })
    }
    #[classmethod]
    #[pyo3(signature = (start, stop, step=NumLike::one()))]
    pub fn arange(_cls: &Bound<'_, PyType>, start: f32, stop: f32, step: f32) -> PyResult<Self> {
        Ok(Self { value: 
            core::Tensor::<f32>::arange(start, stop, step)
        })
    }

    pub fn __repr__(&mut self) -> String {
        self.value.repr()
    }

    pub fn __add__(&mut self, other: &mut Self) -> PyResult<Self>{
        Ok(Self { 
            value: self.value.add(&mut other.value) 
        })
    }
    pub fn __iadd__(&mut self, other: &mut Self){
        self.value.add_(&mut other.value)
    }
    pub fn __sub__(&mut self, other: &mut Self) -> PyResult<Self>{
        Ok(Self { 
            value: self.value.sub(&mut other.value) 
        })
    }
    pub fn __isub__(&mut self, other: &mut Self){
        self.value.sub_(&mut other.value)
    }
    pub fn __mul__(&mut self, other: &mut Self) -> PyResult<Self>{
        Ok(Self { 
            value: self.value.mul(&mut other.value) 
        })
    }
    pub fn __imul__(&mut self, other: &mut Self){
        self.value.mul_(&mut other.value)
    }
    pub fn __div__(&mut self, other: &mut Self) -> PyResult<Self>{
        Ok(Self { 
            value: self.value.div(&mut other.value) 
        })
    }
    pub fn __idiv__(&mut self, other: &mut Self){
        self.value.div_(&mut other.value)
    }
    pub fn __matmul__(&mut self, other: &mut Self) -> PyResult<Self>{
        Ok(Self { 
            value: self.value.matmul(&mut other.value) 
        })
    }
    pub fn matmul(&mut self, other: &mut Self) -> PyResult<Self>{
        Ok(Self { 
            value: self.value.matmul(&mut other.value) 
        })
    }
    pub fn dot(&mut self, other: &mut Self) -> PyResult<Self>{
        Ok(Self { 
            value: self.value.dot(&mut other.value) 
        })
    }
    #[getter]
    pub fn shape(&self)->Vec<usize>{
        self.value.get_shape().clone()
    }
    #[getter]
    pub fn strides(&self)->Vec<usize>{
        self.value.get_strides().clone()
    }
    #[getter]
    pub fn ndim(&self)->usize{
        *self.value.get_ndim()
    }
    pub fn reshape_(&mut self, new_shape: Vec<usize>){
        self.value.reshape_(new_shape)
    }
    pub fn reshape(&mut self, new_shape: Vec<usize>)->PyResult<Self>{
        Ok(Self { 
            value: ReshapeTensor::from(&mut self.value, new_shape).to_tensor()
        })
    }
    pub fn T(&mut self)->PyResult<Self>{
        Ok(Self { 
            value: TransposeTensor::T(&mut self.value).to_tensor()
        })
    }
    pub fn transpose(&mut self, axis_order: Vec<usize>)->PyResult<Self>{
        Ok(Self { 
            value: TransposeTensor::from(&mut self.value, axis_order).to_tensor()
        })
    }
    pub fn get_by_slicing(&mut self, slice: Vec<[Option<usize>; 3]>)->PyResult<Self>{
        let mut index = Vec::<Slice>::with_capacity(slice.len() as usize);
        for i in slice.iter(){
            if let Some(start) = i[0] {
                // index.push(Slice::new(i[0]..))
                if let Some(end) = i[1] {
                    if let Some(step) = i[2] {
                        index.push(Slice::new(start..end).step(step))
                    }else{
                        index.push(Slice::new(start..end))
                    }
                }else if let Some(step) = i[2] {
                    index.push(Slice::new(start..).step(step))
                }else{
                    index.push(Slice::new(start..))
                }
            }else if let Some(end) = i[1] {
                if let Some(step) = i[2] {
                    index.push(Slice::new(..end).step(step))
                }else{
                    index.push(Slice::new(..end))
                }
            }else if let Some(step) = i[2] {
                index.push(Slice::new(..).step(step))
            }else{
                index.push(Slice::new(..))
            }
        }
        
        Ok(Self { 
            value: SlicedTensor::from(&mut self.value, &index).to_tensor()
        })
    }
    pub fn set_by_slicing(&mut self, slice: Vec<[Option<usize>; 3]>, data_flatten: Vec<f32>){
        let mut index = Vec::<Slice>::with_capacity(slice.len() as usize);
        for i in slice.iter(){
            if let Some(start) = i[0] {
                // index.push(Slice::new(i[0]..))
                if let Some(end) = i[1] {
                    if let Some(step) = i[2] {
                        index.push(Slice::new(start..end).step(step))
                    }else{
                        index.push(Slice::new(start..end))
                    }
                }else if let Some(step) = i[2] {
                    index.push(Slice::new(start..).step(step))
                }else{
                    index.push(Slice::new(start..))
                }
            }else if let Some(end) = i[1] {
                if let Some(step) = i[2] {
                    index.push(Slice::new(..end).step(step))
                }else{
                    index.push(Slice::new(..end))
                }
            }else if let Some(step) = i[2] {
                index.push(Slice::new(..).step(step))
            }else{
                index.push(Slice::new(..))
            }
        }
        let mut mut_tensor = SlicedTensor::from(&mut self.value, &index);
        
        let mut indices: Vec<usize> = vec![0; *mut_tensor.get_ndim()];
        let ndim = *mut_tensor.get_ndim();
        let shape = mut_tensor.get_shape().clone();
        let mut data_flatten_index = 0;
        loop {
            *mut_tensor.get_mut_by_indices(&indices) = data_flatten[data_flatten_index];
            data_flatten_index += 1;
            // Increment indices
            let mut i = ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return;
                }
                i -= 1;
            }
        }
    }
}

#[pyclass]
struct TensorType;
#[pymethods]
impl TensorType {
    #[new]
    pub fn new() -> Self {
        Self
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn tftensor_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<Core>()?;

    m.add_class::<Tensor_f32>()?;
    m.add_class::<Tensor_f64>()?;
    m.add_class::<Tensor_i8>()?;
    m.add_class::<Tensor_i16>()?;
    m.add_class::<Tensor_i32>()?;
    m.add_class::<Tensor_i64>()?;
    m.add_class::<Tensor_u8>()?;
    m.add_class::<Tensor_u16>()?;
    m.add_class::<Tensor_u32>()?;
    m.add_class::<Tensor_u64>()?;

    m.add_class::<Tensor_f32_test>()?;

    m.add_class::<TensorType>()?;

    Ok(())
}
