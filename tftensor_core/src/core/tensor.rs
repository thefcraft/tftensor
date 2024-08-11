use super::dtype::{NumLike, DType};
use super::TensorLike;
use super::random::LCG;

pub enum Order {
    c_like,
    f_like
}
pub struct Tensor<T: NumLike>{
    data: Vec<T>,
    strides: Vec<usize>,
    shape: Vec<usize>,
    ndim: usize,
    size: usize,
    // f_contiguous: bool, // Fortran-like order (column-major order). 
    // c_contiguous: bool, // C-like order (row-major order)
    dtype: DType,
    // device: Device
}

impl<T: NumLike> Tensor<T> {
    // pub fn new(data_flatten: Vec<T>, shape: Vec<usize>, order: Order) -> Self 
    pub fn new(data_flatten: Vec<T>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        debug_assert!(size == data_flatten.len(), "shape does not match to the given data length");
        
        
        let ndim = shape.len();
        // let (f_contiguous, c_contiguous, strides) = match order {
        //     Order::f_like => {
        //         panic!("f_contiguous is not supported yet");
        //         let mut strides: Vec<usize> = vec![1; ndim];
        //         for i in 1..ndim{
        //             strides[i] = strides[i - 1] * shape[i-1];
        //         }
        //         (true, false, strides)
        //     },
        //     Order::c_like => {
        //         let mut strides: Vec<usize> = vec![1; ndim];
        //         for i in (0..ndim - 1).rev() {
        //             strides[i] = strides[i + 1] * shape[i + 1];
        //         }
        //         (false, true, strides)
        //     },
        // };
        
        let mut strides: Vec<usize> = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        Self {
            data: data_flatten,
            strides,
            shape,
            ndim,
            size,
            // f_contiguous,
            // c_contiguous,           
            // device: gpu.into(),
            dtype: T::dtype(), // DType::from::<T>()
        }
    }
    pub fn from(data_flatten: Vec<T>, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        debug_assert!(size == data_flatten.len(), "shape does not match to the given data length");
        let ndim = shape.len();
        
        let mut result = Self {
            data: data_flatten,
            strides,
            shape,
            ndim,
            size,
            // f_contiguous,
            // c_contiguous: false,           
            // device: gpu.into(),
            dtype: T::dtype(), // DType::from::<T>()
        };
        // result.c_contiguous = result.is_c_contiguous();
        result
    }
    pub fn ones(shape: Vec<usize>) -> Self {
        let size:usize = shape.iter().product();
        Self::new(vec![T::one(); size], shape) 
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![T::zero(); size], shape)
    }
    pub fn full(shape: Vec<usize>, fill_value:T) -> Self {
        let size = shape.iter().product();
        Self::new(vec![fill_value; size], shape)
    }
    pub fn randn(shape: Vec<usize>, seed: Option<u64>) -> Self {
        let mut random = LCG::new();
        if let Some(seed) = seed {
            random.set_seed(seed);
        }
        let size: usize = shape.iter().product();
        let data: Vec<T> = (0..size).map(|_| T::random(&mut random)).collect();
        Self::new(data, shape)
    }
    pub fn arange(mut start: T, stop: T, step:T)->Self{
        let mut data_flatten = Vec::<T>::new();
        while start < stop {
            data_flatten.push(start.clone());
            start += step.clone();
        }
        let size = data_flatten.len();
        Self::new(data_flatten, vec![size])
    }
    pub fn reshape_(&mut self, new_shape: Vec<usize>){
        let new_size: usize = new_shape.iter().product();
        debug_assert_eq!(new_size, *self.get_size(), "New shape must have the same total number of elements as the original shape");
        let ndim = new_shape.len();
        
        let mut strides: Vec<usize> = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * new_shape[i + 1];
        }
        self.strides = strides;
        self.ndim = ndim;
        self.shape = new_shape;
        self.size = new_size;
    }
    pub fn item(&self) -> &T{
        debug_assert!(self.data.len() == 1, "Item must be called on scalar data");
        &self.data[0]
    }
    pub fn get_data(&self) -> &Vec<T>{
        &self.data
    }
    pub fn get_mut_data(&mut self) -> &mut Vec<T>{
        &mut self.data
    }
    pub fn from_num_like(value: T)->Self{
        Self::new(vec![value; 1], vec![1])
    }
}


impl<T: NumLike> TensorLike<T> for Tensor<T> {
    fn is_mutable(&self) -> bool{true}
    fn get_mut_by_indices(&mut self, indices:&Vec<usize>)->&mut T{
        debug_assert!(indices.len() == self.ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");

        let index:usize = indices.iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
        &mut self.data[index]
    }
    fn get_by_indices(&mut self, indices:&Vec<usize>)-> &T{
        debug_assert!(indices.len() == self.ndim, "Provided full indices");
        // println!("{:?} {:?}", self.shape, indices);
        debug_assert!(indices.iter().zip(self.shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");

        let index:usize = indices.iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
        &self.data[index]
    }
    fn get_strides(&self)->&Vec<usize> {
        &self.strides   
    }
    fn get_shape(&self)->&Vec<usize>{
        &self.shape
    }
    fn get_ndim(&self)->&usize{
        &self.ndim
    }
    fn get_size(&self)->&usize{
        &self.size
    }
    // fn get_f_contiguous(&self)->&bool{
    //     &self.f_contiguous
    // }
    // fn get_c_contiguous(&self)->&bool{
    //     &self.c_contiguous
    // }
    fn get_dtype(&self)->&DType{
        &self.dtype
    }
}

impl<T: NumLike> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor::new(vec![value; 1], vec![1])
    }
}