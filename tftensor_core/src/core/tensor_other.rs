use super::{Tensor, TensorLike};
use super::dtype::{NumLike, DType};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

pub struct TransposeTensor<'a, T, U>
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub tensor: &'a mut U,
    pub new_strides: Vec<usize>,
    pub new_shape: Vec<usize>,
    pub new_size: usize,
    pub new_ndim: usize,
    pub phantom: PhantomData<T>,  // PhantomData to indicate association with T
    
    pub __temp_indices: Vec<usize>,
    axis_order: Vec<usize>, 
    __nochange: bool,
}
// TODO check __nochange in case of traspose as it have bugs
impl<'a, T, U> TransposeTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub fn from(tensor: &'a mut U, axis_order: Vec<usize>) -> Self{
        // Validate axis_orde
        let ndim = *tensor.get_ndim();
        if axis_order.len() != ndim {
            panic!("Axis order length must match tensor dimension length.");
        }
        let mut seen = vec![false; ndim];
        for &axis in axis_order.iter() {
            if axis >= ndim || seen[axis] {
                panic!("Invalid axis order. Axis order must be a permutation of tensor dimensions.");
            }
            seen[axis] = true;
        }
                
        // Compute new shape and strides
        let mut new_shape = vec![0; ndim];
        let mut new_strides = vec![0; ndim];

        // Reorder shape and strides according to axis_order
        for (new_axis, &old_axis) in axis_order.iter().enumerate() {
            new_shape[new_axis] = tensor.get_shape()[old_axis];
            new_strides[new_axis] = tensor.get_strides()[old_axis];
        }
        let __nochange = *tensor.get_strides() == new_strides;
        let new_size = *tensor.get_size();
        // println!("{:?}", new_shape);
        // println!("{:?}", new_strides);
        Self { 
            tensor, 
            new_strides, 
            new_shape, 
            new_size, 
            new_ndim: ndim, 
            phantom: PhantomData, 
            __temp_indices: vec![0; ndim],
            axis_order,
            __nochange
        }
    }
    pub fn T(tensor: &'a mut U)->Self{
        let mut new_shape = tensor.get_shape().clone();
        let mut new_strides = tensor.get_strides().clone();
        new_shape.reverse();
        new_strides.reverse();
        let __nochange = *tensor.get_strides() == new_strides;
        let new_size = *tensor.get_size();
        let new_ndim = *tensor.get_ndim();
        let mut axis_order: Vec<usize> = (0..new_ndim).into_iter().map(|x| x).collect();
        axis_order.reverse();
        Self { 
            tensor, 
            new_strides, 
            new_shape, 
            new_size, 
            new_ndim, 
            phantom: PhantomData, 
            __temp_indices: vec![0; new_ndim],
            axis_order,
            __nochange
        }
    }
}

impl<'a, T, U>  TensorLike<T> for TransposeTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    fn is_mutable(&self) -> bool{self.tensor.is_mutable()}
    fn get_mut_by_indices(&mut self, indices:&Vec<usize>)->&mut T{
        // if self.__nochange {return self.tensor.get_mut_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        // Reorder shape and strides according to axis_order
        for (&axis, &idx) in self.axis_order.iter().zip(indices.iter()) {
            self.__temp_indices[axis] = idx;
        }
        return self.tensor.get_mut_by_indices(&self.__temp_indices);
    }
    fn get_by_indices(&mut self, indices:&Vec<usize>)-> &T{
        // if self.__nochange {return self.tensor.get_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        // Reorder shape and strides according to axis_order
        for (&axis, &idx) in self.axis_order.iter().zip(indices.iter()) {
            self.__temp_indices[axis] = idx;
        }
        return self.tensor.get_by_indices(&self.__temp_indices);
    }
    fn get_strides(&self)->&Vec<usize> {
        &self.new_strides   
    }
    fn get_shape(&self)->&Vec<usize>{
        &self.new_shape
    }
    fn get_ndim(&self)->&usize{
        &self.new_ndim
    }
    fn get_size(&self)->&usize{
        &self.new_size
    }
    // fn get_c_contiguous(&self)->&bool{
    //     self.tensor.get_c_contiguous()
    // }
    fn get_dtype(&self)->&DType{
        self.tensor.get_dtype()
    }
}



pub struct ReshapeTensor<'a, T, U>
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub tensor: &'a mut U,
    pub new_strides: Vec<usize>,
    pub new_shape: Vec<usize>,
    pub new_size: usize,
    pub new_ndim: usize,
    pub phantom: PhantomData<T>,  // PhantomData to indicate association with T

    pub __temp_indices: Vec<usize>,

    __nochange: bool,
}

impl<'a, T, U> ReshapeTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub fn from(tensor: &'a mut U, new_shape: Vec<usize>) -> Self{
        let new_size: usize = new_shape.iter().product();
        debug_assert_eq!(new_size, *tensor.get_size(), "New shape must have the same total number of elements as the original shape");
        
        let new_ndim = new_shape.len();
        let old_ndim = *tensor.get_ndim();

        let mut new_strides = vec![1; new_ndim];
        for i in (0..new_ndim - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
        let __nochange = new_shape == *tensor.get_shape();
        Self{
            tensor,
            new_strides,
            new_shape,
            new_size,
            new_ndim,
            phantom: PhantomData,

            __temp_indices: vec![0; old_ndim],

            __nochange,
        }
    }
}

impl<'a, T, U>  TensorLike<T> for ReshapeTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    fn is_mutable(&self) -> bool{self.tensor.is_mutable()}
    fn get_mut_by_indices(&mut self, indices:&Vec<usize>)->&mut T{
        if self.__nochange {return self.tensor.get_mut_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        // Convert the reshaped index to a flat index
        let mut flat_index:usize = indices.iter()
            .zip(self.get_strides().iter())
            .map(|(&i, &s)| i * s)
            .sum();
        
        // Calculate the number of elements in each block of the original array
        for (i, &block_size) in self.tensor.get_strides().iter().enumerate(){
            self.__temp_indices[i] = flat_index / block_size;
            flat_index %= block_size;
        }
        return self.tensor.get_mut_by_indices(&self.__temp_indices);
    }
    fn get_by_indices(&mut self, indices:&Vec<usize>)-> &T{
        if self.__nochange {return self.tensor.get_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        // Convert the reshaped index to a flat index
        let mut flat_index:usize = indices.iter()
            .zip(self.get_strides().iter())
            .map(|(&i, &s)| i * s)
            .sum();
        
        // Calculate the number of elements in each block of the original array
        for (i, &block_size) in self.tensor.get_strides().iter().enumerate(){
            self.__temp_indices[i] = flat_index / block_size;
            flat_index %= block_size;
        }

        self.tensor.get_by_indices(&self.__temp_indices)
    }
    fn get_strides(&self)->&Vec<usize> {
        &self.new_strides   
    }
    fn get_shape(&self)->&Vec<usize>{
        &self.new_shape
    }
    fn get_ndim(&self)->&usize{
        &self.new_ndim
    }
    fn get_size(&self)->&usize{
        &self.new_size
    }
    // fn get_c_contiguous(&self)->&bool{
    //     self.tensor.get_c_contiguous()
    // }
    fn get_dtype(&self)->&DType{
        self.tensor.get_dtype()
    }
}


pub struct BroadCastTensor<'a, T, U>
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub tensor: &'a mut U,
    pub new_strides: Vec<usize>,
    pub new_shape: Vec<usize>,
    pub new_size: usize,
    pub new_ndim: usize,

    pub phantom: PhantomData<T>,  // PhantomData to indicate association with T

    pub old_ndim: usize, // new_indices[-old_ndim:]
    pub old_indices_mask: Vec<usize>,    // old_shape = [1, 3], new_shape = [1, 3, 3] => new_indices [0, 1, 2] => [1, 2] => 

    pub __temp_indices: Vec<usize>,

    __nochange: bool,
}

impl<'a, T, U> BroadCastTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub fn from(tensor: &'a mut U, target_shape: &Vec<usize>) -> Self{
        let mut new_shape = tensor.get_shape().clone();
        let mut new_strides = tensor.get_strides().clone();

        let mut old_indices_mask: Vec<usize> = vec![0; *tensor.get_ndim()];

        // Reverse the shapes and strides for easier processing
        new_shape.reverse();
        new_strides.reverse();

        // Pad shape and strides with 1s from the right if necessary
        while new_shape.len() < target_shape.len() {
            new_shape.push(1);
            new_strides.push(0);
        }
        // Check if broadcasting is possible and adjust strides
        for (i, (&target_dim, &current_dim)) in target_shape.iter().rev().zip(new_shape.clone().iter()).enumerate() {
            if current_dim == target_dim {
                continue;
            } else if current_dim == 1 {
                new_shape[i] = target_dim;
                new_strides[i] = 0;
                if i < *tensor.get_ndim(){
                    old_indices_mask[i] = 1;
                }
            } else {
                panic!("broadcasting is not possible for shape {:?} to {:?}", tensor.get_shape(), target_shape)
                // return None; // Broadcasting not possible
            }
        }
        
        // Reverse back the shapes and strides
        new_shape.reverse();
        new_strides.reverse();
        old_indices_mask.reverse();
        let new_size:usize = new_shape.iter().product();
        let new_ndim = new_shape.len();
        let old_ndim = tensor.get_shape().len();
        let __nochange = new_shape == *tensor.get_shape();
        Self { 
            tensor, 
            new_strides, 
            new_shape, 
            new_size, 
            new_ndim, 
            old_ndim, 
            old_indices_mask,
            phantom: PhantomData,
            __temp_indices: vec![0; old_ndim],
            __nochange
        }
    }
    
    #[inline]
    pub fn new_shape<V>(tensor_a: &'a U, tensor_b: &'a V) -> Vec<usize>
        where V: TensorLike<T>
    {
        BroadCastTensor::<T, U>::new_shape_from_shape(tensor_a.get_shape(), tensor_b.get_shape())
    }
    pub fn new_shape_from_shape(shape_a: &Vec<usize>, shape_b: &Vec<usize>) -> Vec<usize>{
        // Determine the length of the new shape (the max length of both input shapes)
        let len_a = shape_a.len();
        let len_b = shape_b.len();
        let max_len = usize::max(len_a, len_b);
        // Pad the shorter shape with 1s on the left
        let padded_shape_a: Vec<usize> = vec![1; max_len - len_a].into_iter().chain(shape_a.iter().cloned()).collect();
        let padded_shape_b: Vec<usize> = vec![1; max_len - len_b].into_iter().chain(shape_b.iter().cloned()).collect();

        // Compute the broadcasted shape
        let mut broadcasted_shape: Vec<usize> = Vec::with_capacity(max_len);
        for (dim_a, dim_b) in padded_shape_a.iter().zip(padded_shape_b.iter()) {
            match (dim_a, dim_b) {
                (1, b) => broadcasted_shape.push(*b),
                (a, 1) => broadcasted_shape.push(*a),
                (a, b) if a == b => broadcasted_shape.push(*a),
                _ => panic!("Broadcasting is not possible for shape {:?} and {:?}", shape_a, shape_b), // Broadcasting is not possible
            }
        }
        broadcasted_shape
    }
}

impl<'a, T, U>  TensorLike<T> for BroadCastTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    fn is_mutable(&self) -> bool{ false }
    fn get_mut_by_indices(&mut self, indices:&Vec<usize>)->&mut T{
        if self.__nochange {return self.tensor.get_mut_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        
        // Calculate the starting index for the last n elements
        let start = self.new_ndim.saturating_sub(self.old_ndim);
        let new_indices = &mut self.__temp_indices;
        for (i, &mask) in (start..indices.len()).zip(self.old_indices_mask.iter()){
            if mask == 1{
                new_indices[i-start] = 0;
            }else {
                new_indices[i-start] = indices[i];
            }
        }
        return self.tensor.get_mut_by_indices(&self.__temp_indices);
    }
    fn get_by_indices(&mut self, indices:&Vec<usize>)-> &T{
        if self.__nochange {return self.tensor.get_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        
        // Calculate the starting index for the last n elements
        let start = self.new_ndim.saturating_sub(self.old_ndim);
        let new_indices = &mut self.__temp_indices;
        for (i, &mask) in (start..indices.len()).zip(self.old_indices_mask.iter()){
            if mask == 1{
                new_indices[i-start] = 0;
            }else {
                new_indices[i-start] = indices[i];
            }
        }
        self.tensor.get_by_indices(&new_indices)
    }
    fn get_strides(&self)->&Vec<usize> {
        &self.new_strides   
    }
    fn get_shape(&self)->&Vec<usize>{
        &self.new_shape
    }
    fn get_ndim(&self)->&usize{
        &self.new_ndim
    }
    fn get_size(&self)->&usize{
        &self.new_size
    }
    // fn get_c_contiguous(&self)->&bool{
    //     self.tensor.get_c_contiguous()
    // }
    fn get_dtype(&self)->&DType{
        self.tensor.get_dtype()
    }
}
#[derive(Clone, Copy)]
pub struct Slice {
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub step: Option<usize>,
}

impl Slice {
    pub fn new<R>(range: R) -> Self 
    where R : RangeBounds<usize>{
        let start = match range.start_bound() {
            Bound::Included(&n) => Some(n),
            Bound::Excluded(&n) => Some(n + 1),
            Bound::Unbounded => None,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => Some(n + 1),
            Bound::Excluded(&n) => Some(n),
            Bound::Unbounded => None,
        };
        Self { start, end, step: None }
    }
    pub fn step(&self, step: usize) -> Self{
        Self { start: self.start, end: self.end, step: Some(step) }
    }
}

pub struct SlicedTensor<'a, T, U>
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub tensor: &'a mut U,
    pub new_strides: Vec<usize>,
    pub new_shape: Vec<usize>,
    pub new_size: usize,
    pub new_ndim: usize,

    pub phantom: PhantomData<T>,  // PhantomData to indicate association with T

    // pub temp_indices: Vec<usize>, // to remove unnecessary allocation of indices in get_indices
    pub new_shape_with_zeros: Vec<usize>, // new shape with same ndim as old one
    pub old_indices_mask: Vec<usize>, // 

    pub indices_jump: Vec<usize>,
    
    pub __temp_indices: Vec<usize>,

    __nochange: bool,
}

impl<'a, T, U> SlicedTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    pub fn from(tensor: &'a mut U, index: &Vec<Slice>) -> Self{
        debug_assert!(index.len()>0, "Provided full index");
        debug_assert!(index.len() <= *tensor.get_ndim(), "index is larger then shape");
        let old_ndim = *tensor.get_ndim();
        let mut new_shape = tensor.get_shape().clone();
        let mut old_indices_mask = vec![0; *tensor.get_ndim()];
        let mut jump = vec![0; index.len()];
        for (idx, slice) in index.iter().enumerate(){
            let start = if let Some(start) = slice.start{start}
                               else{0};
            let new_dim = if let Some(end) = slice.end{end-start}
                                 else{new_shape[idx]-start};
            let step = if let Some(step) = slice.step{step}
                            else {1};
            debug_assert!(new_dim <= new_shape[idx]+1, "dim out of range");
            old_indices_mask[idx] = start;
            
            new_shape[idx] = (new_dim+step-1) / step; // todo
            
            jump[idx] = step;
        }

        let mut new_shape_with_zeros = Vec::<usize>::new();
        for &i in new_shape.iter(){
            if i != 0{
                new_shape_with_zeros.push(i);
            }
        }
        let (new_shape_with_zeros, new_shape ) = (new_shape, new_shape_with_zeros);
        
        // SlicedTensor_native::new(self, new_shape, new_shape_with_zeros, old_indices_mask)
        let new_size:usize = new_shape.iter().map(|&x| std::cmp::max(1, x)).product();
        let mut new_strides = vec![1; new_shape_with_zeros.len()];
        for i in (0..new_shape_with_zeros.len() - 1).rev() {
            if new_shape_with_zeros[i + 1] == 0{
                new_strides[i] = new_strides[i + 1];
            }else{
                new_strides[i] = new_strides[i + 1] * new_shape_with_zeros[i + 1];
            }
        }

        let new_ndim = new_shape.len();
        // let temp_indices = new_shape_with_zeros.clone();
        // let new_size:usize = new_shape.iter().product();
        // println!("new_shape_with_zeros : {:?}", new_shape_with_zeros);
        // println!("new_shape : {:?}", new_shape);
        // println!("new_strides : {:?}", new_strides);
        // println!("old_indices_mask : {:?}", old_indices_mask);
        // println!("indices_jump : {:?}", jump);
        let __nochange = new_shape == *tensor.get_shape();
        Self { 
            tensor, 
            new_shape, 
            new_size,
            new_ndim,
            new_strides, // TODO handle zero case here... //BUG

            new_shape_with_zeros,
            old_indices_mask,
            indices_jump: jump,
            phantom: PhantomData,
            __temp_indices: vec![0; old_ndim],
            
            __nochange
        }
    }
}

impl<'a, T, U>  TensorLike<T> for SlicedTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{
    fn is_mutable(&self) -> bool{self.tensor.is_mutable()}
    fn get_mut_by_indices(&mut self, indices:&Vec<usize>)->&mut T{
        if self.__nochange {return self.tensor.get_mut_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        
        let new_indices = &mut self.__temp_indices;
        
        let mut idx = 0;
        for i in 0..self.new_shape_with_zeros.len() {
            if self.new_shape_with_zeros[i] != 0{
                new_indices[i] = self.old_indices_mask[i] + indices[idx] + (indices[idx] * (self.indices_jump[i]-1));
                idx+=1;
            }else{
                new_indices[i] = self.old_indices_mask[i];
            }
        }
        
        return self.tensor.get_mut_by_indices(&self.__temp_indices);
    }
    fn get_by_indices(&mut self, indices:&Vec<usize>)-> &T{
        if self.__nochange {return self.tensor.get_by_indices(&indices);}
        debug_assert!(indices.len() == self.new_ndim, "Provided full indices");
        debug_assert!(indices.iter().zip(self.new_shape.iter()).all(|(&index, &dim)| index < dim), "Check indices");
        
        let new_indices = &mut self.__temp_indices;
        
        let mut idx = 0;
        for i in 0..self.new_shape_with_zeros.len() {
            if self.new_shape_with_zeros[i] != 0{
                new_indices[i] = self.old_indices_mask[i] + indices[idx] + (indices[idx] * (self.indices_jump[i]-1));
                idx+=1;
            }else{
                new_indices[i] = self.old_indices_mask[i];
            }
        }
        self.tensor.get_by_indices(new_indices)
    }
    fn get_strides(&self)->&Vec<usize> {
        &self.new_strides   
    }
    fn get_shape(&self)->&Vec<usize>{
        &self.new_shape
    }
    fn get_ndim(&self)->&usize{
        &self.new_ndim
    }
    fn get_size(&self)->&usize{
        &self.new_size
    }
    // fn get_c_contiguous(&self)->&bool{
    //     self.tensor.get_c_contiguous()
    // }
    fn get_dtype(&self)->&DType{
        self.tensor.get_dtype()
    }
}

