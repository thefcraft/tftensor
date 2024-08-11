use super::PrintHelper;
use super::{NumLike, dtype::DType, Tensor, BroadCastTensor, ReshapeTensor, SlicedTensor, Slice};

pub trait TensorLike<T: NumLike>{
    fn get_by_indices(&mut self, indices:&Vec<usize>)->&T;
    fn get_mut_by_indices(&mut self, indices:&Vec<usize>)->&mut T;
    fn get_strides(&self)->&Vec<usize>;
    fn get_shape(&self)->&Vec<usize>;
    fn get_ndim(&self)->&usize;
    fn get_size(&self)->&usize;
    // fn get_f_contiguous(&self)->&bool;
    // fn get_c_contiguous(&self)->&bool;
    fn get_dtype(&self)->&DType;
    // fn get_device(&self)->&Device;

    fn to_tensor(&mut self) -> Tensor<T>{
        let mut result_flatten = Vec::<T>::with_capacity(*self.get_size());
        let mut indices: Vec<usize> = vec![0; *self.get_ndim()];
        let ndim = *self.get_ndim();
        let shape = self.get_shape().clone();
        loop {
            // println!("{:?}", indices);
            result_flatten.push(
                *self.get_by_indices(&indices)
            );
            // Increment indices
            let mut i = ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return Tensor::new(result_flatten, shape);
                }
                i -= 1;
            }
        }
    }

    fn is_f_contiguous(&self)->bool{
        let mut strides: Vec<usize> = vec![1; *self.get_ndim()];
        let shape = self.get_shape();
        for i in 1..*self.get_ndim(){
            strides[i] = strides[i - 1] * shape[i-1];
        };
        strides == *self.get_strides()
    }
    fn is_c_contiguous(&self)->bool{
        let mut strides: Vec<usize> = vec![1; *self.get_ndim()];
        let shape = self.get_shape();
        for i in (0..self.get_ndim() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides == *self.get_strides()
    }

    fn is_mutable(&self) -> bool;

    
    fn add<U: TensorLike<T>>(&mut self, other: &mut U) -> Tensor<T>
        where Self:Sized 
    {
        let new_shape = BroadCastTensor::new_shape(self, other);
        let new_size: usize = new_shape.iter().product();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut broadcasted_self = BroadCastTensor::from(self, &new_shape);
        let mut result_flatten = Vec::<T>::with_capacity(new_size);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        
        
        loop {
            result_flatten.push(
                broadcasted_self.get_by_indices(&indices).safe_add(broadcasted_other.get_by_indices(&indices))
            );
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return Tensor::new(result_flatten, new_shape);
                }
                i -= 1;
            }
        }
    }
    fn sub<U: TensorLike<T>>(&mut self, other: &mut U) -> Tensor<T>
        where Self:Sized 
    {
        let new_shape = BroadCastTensor::new_shape(self, other);
        let new_size: usize = new_shape.iter().product();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut broadcasted_self = BroadCastTensor::from(self, &new_shape);
        let mut result_flatten = Vec::<T>::with_capacity(new_size);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        
        loop {
            result_flatten.push(
                broadcasted_self.get_by_indices(&indices).safe_sub(broadcasted_other.get_by_indices(&indices))
            );
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return Tensor::new(result_flatten, new_shape);
                }
                i -= 1;
            }
        }
    }
    fn mul<U: TensorLike<T>>(&mut self, other: &mut U) -> Tensor<T>
        where Self:Sized 
    {
        let new_shape = BroadCastTensor::new_shape(self, other);
        let new_size: usize = new_shape.iter().product();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut broadcasted_self = BroadCastTensor::from(self, &new_shape);
        let mut result_flatten = Vec::<T>::with_capacity(new_size);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        
        loop {
            result_flatten.push(
                broadcasted_self.get_by_indices(&indices).safe_mul(broadcasted_other.get_by_indices(&indices))
            );
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return Tensor::new(result_flatten, new_shape);
                }
                i -= 1;
            }
        }
    }
    fn div<U: TensorLike<T>>(&mut self, other: &mut U) -> Tensor<T>
        where Self:Sized 
    {
        let new_shape = BroadCastTensor::new_shape(self, other);
        let new_size: usize = new_shape.iter().product();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut broadcasted_self = BroadCastTensor::from(self, &new_shape);
        let mut result_flatten = Vec::<T>::with_capacity(new_size);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        
        loop {
            result_flatten.push(
                broadcasted_self.get_by_indices(&indices).safe_div(broadcasted_other.get_by_indices(&indices))
            );
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return Tensor::new(result_flatten, new_shape);
                }
                i -= 1;
            }
        }
    }

    //TODO: implement this function for single dimensions tensors 
    fn matmul<U: TensorLike<T>>(&mut self, other: &mut U) -> Tensor<T>
        where Self:Sized 
    {
        // np.allclose(np.einsum('ijk,ikl->ijl', a,b), a@b)        # True
        // performs batched matrix multiplication
        debug_assert!(*self.get_ndim() >= 2 && *other.get_ndim() >= 2, "Tensor must have at least 2 dimensions for matmul");
        debug_assert_eq!(self.get_shape()[self.get_ndim() - 1], other.get_shape()[other.get_ndim() - 2], "(..m,n) @ (..n,i) only check dimensions once again");
        
        let mut other_shape_temp = other.get_shape().clone();
        other_shape_temp[other.get_ndim() - 1] = self.get_shape()[self.get_ndim() - 1];
        other_shape_temp[other.get_ndim() - 2] = self.get_shape()[self.get_ndim() - 2];
        let mut new_shape_temp = BroadCastTensor::<T, U>::new_shape_from_shape(self.get_shape(), &other_shape_temp);
        let mut broadcasted_self = BroadCastTensor::from(self, &new_shape_temp);
        new_shape_temp[broadcasted_self.new_ndim - 1] = other.get_shape()[other.get_ndim() - 1];
        new_shape_temp[broadcasted_self.new_ndim - 2] = other.get_shape()[other.get_ndim() - 2];
        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape_temp);
        // self,  (1,2,4,2) => (1,2,4,2) => (3,2,4,2)
        // other, (3,1,2,8) => (3,1,4,2) => (3,2,4,2)
        
        let nj = broadcasted_self.get_shape()[broadcasted_self.get_ndim() - 2];
        let nk = broadcasted_other.get_shape()[broadcasted_other.get_ndim() - 1];
        
        let mut shape = broadcasted_self.get_shape().clone();
        shape[broadcasted_self.get_ndim() - 1] = broadcasted_other.get_shape()[broadcasted_other.get_ndim() - 1];
        let shape = shape; // non mutable shape
        let new_size: usize = shape.iter().product();
    
            
        // let mut result = Tensor::zeros(shape.clone());
        let mut result_flatten = Vec::<T>::with_capacity(new_size);
        // if self.shape.len() > 2
        let mut indices: Vec<usize> = vec![0; broadcasted_self.get_ndim()-2];
        // let mut ptr = 0;
        let broadcasted_self_ndim = *broadcasted_self.get_ndim();
        loop {
            // println!("{:?}", indices);
            let mut slice_index = vec![Slice::new(..); broadcasted_self_ndim];
            for (j, &i) in indices.iter().enumerate() {
                slice_index[j] = Slice::new(i..i);
            }
            for j in 0..nj{
                for k in 0..nk{
                    slice_index[broadcasted_self_ndim-1] = Slice::new(k..k);
                    let mut k_tensor = SlicedTensor::from(&mut broadcasted_other, &slice_index);
                    // println!("{}", k_tensor);
                    // println!("{:?}", slice_index);
                    slice_index[broadcasted_self_ndim-1] = Slice::new(..);
                    slice_index[broadcasted_self_ndim-2] = Slice::new(j..j);
                    
                    let mut j_tensor = SlicedTensor::from(&mut broadcasted_self, &slice_index);
                    // println!("{}", j_tensor);
                    // println!("{:?}", slice_index);
                    slice_index[broadcasted_self_ndim-2] = Slice::new(..);
                    
                    let s = j_tensor.mul(&mut k_tensor).sum(None, false);
                    let s = *s.item();
                    result_flatten.push(s);

                    // println!("{}", s);
                }
            }
            if indices.len() == 0{
                return Tensor::new(result_flatten, shape);
            }
            let mut i = indices.len() - 1;
            loop {
                indices[i] += 1;
                if indices[i] < shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return Tensor::new(result_flatten, shape);
                }
                i -= 1;
            }
        }
    }
    //TODO: implement this function for single dimensions tensors 
    fn dot<U: TensorLike<T>>(&mut self, other: &mut U) -> Tensor<T>
        where Self:Sized 
    {   
        // np.allclose(np.einsum('ijk,lkm->ijlm',a,b), a.dot(b))   # True
        // performs matrix multiplication
        debug_assert!(*self.get_ndim() >= 2 && *other.get_ndim() >= 2, "Tensor must have at least 2 dimensions for dot");
        debug_assert_eq!(self.get_shape()[self.get_ndim() - 1], other.get_shape()[other.get_ndim() - 2], "(..m,n) dot (..n,i) only. check dimensions once again");

        let mut new_shape: Vec<usize> = vec![0; other.get_ndim()+self.get_ndim()-2];
        new_shape[other.get_ndim()+self.get_ndim()-3] = other.get_shape()[other.get_ndim() - 1];
        for i in 0..(*self.get_ndim()-1){
            new_shape[i] = self.get_shape()[i];
        }
        for i in 0..(*other.get_ndim()-2){
            new_shape[(*self.get_ndim() - 1) + i] = other.get_shape()[i];
        }
        let new_ndim = new_shape.len();
        let new_size: usize = new_shape.iter().product();
        let ndim_self = *self.get_ndim();
        let ndim_other = *other.get_ndim();

        let mut result_flatten = Vec::<T>::with_capacity(new_size);
        let mut slice_index_self = vec![Slice::new(..); ndim_self];
        let mut slice_index_other = vec![Slice::new(..); ndim_other];
        let mut indices: Vec<usize> = vec![0; new_ndim];
        loop {
            for i in 0..ndim_self-1 {
                slice_index_self[i] = Slice::new(indices[i]..indices[i]);
            }
            // slice_index_self[ndim_self - 1] = Slice::new(..);
            for i in 0..ndim_other-2 {
                slice_index_other[i] = Slice::new(indices[ndim_self+i-1]..indices[ndim_self+i-1]);
            }
            slice_index_other[ndim_other-1] = Slice::new(indices[new_ndim-1]..indices[new_ndim-1]);
            // slice_index_self[ndim_other - 2] = Slice::new(..);
            
            let mut s = SlicedTensor::from(self, &slice_index_self);
            let mut o = SlicedTensor::from(other, &slice_index_other);
            
            result_flatten.push(*s.mul(&mut o).sum(None, false).item());
            

            if indices.len() == 0{
                return Tensor::new(result_flatten, new_shape);
            }
            let mut i = indices.len() - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return Tensor::new(result_flatten, new_shape);
                }
                i -= 1;
            }
        }
    }

    fn mean(&mut self, dim: Option<usize>, keepdims: bool) -> Tensor<T>
        where Self:Sized 
    {
        if let Some(dim) = dim {
            assert!(dim<*self.get_ndim(), "dim must be less than ndim");
            let mut slice_index = vec![Slice::new(..); *self.get_ndim()];
            slice_index[dim] = Slice::new(0..1);
            let mut result = SlicedTensor::from(self, &slice_index).to_tensor();
            for i in 1..self.get_shape()[dim]{
                slice_index[dim] = Slice::new(i..=i);
                result.add_(&mut SlicedTensor::from(self, &slice_index));
            }
            
            let mut n = Tensor::from_num_like(T::from_f64(self.get_shape()[dim] as f64));
            result.div_(&mut n);

            if keepdims{
                result
            }else {
                let mut new_shape = Vec::<usize>::with_capacity(*self.get_ndim());
                for i in 0..*self.get_ndim() {
                    if i == dim{ continue; }
                    new_shape.push(self.get_shape()[i]);
                }
                if new_shape.len() == 0{new_shape.push(1);}

                result.reshape_(new_shape);// todo
                result
            }
        }else {
            let mut sum = T::zero();
            let mut indices: Vec<usize> = vec![0; *self.get_ndim()];
            loop {
                sum = sum.safe_add(self.get_by_indices(&indices));

                // Increment indices
                let mut i = indices.len() - 1;
                loop {
                    indices[i] += 1;
                    if indices[i] < self.get_shape()[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == 0 {
                        return if keepdims{
                            let new_shape = vec![1; *self.get_ndim()];
                            sum = sum.safe_div(&T::from_f64(*self.get_size() as f64));
                            Tensor::new(vec![sum; 1], new_shape)
                        }else {
                            sum = sum.safe_div(&T::from_f64(*self.get_size() as f64));
                            Tensor::from_num_like(sum)
                        }
                    }
                    i -= 1;
                }
            }
        }
    }
        
    fn sum(&mut self, dim: Option<usize>, keepdims: bool) -> Tensor<T>
        where Self:Sized 
    {
        if let Some(dim) = dim {
            assert!(dim<*self.get_ndim(), "dim must be less than ndim");
            let mut slice_index = vec![Slice::new(..); *self.get_ndim()];
            slice_index[dim] = Slice::new(0..1);
            let mut result = SlicedTensor::from(self, &slice_index).to_tensor();
            for i in 1..self.get_shape()[dim]{
                slice_index[dim] = Slice::new(i..=i);
                result.add_(&mut SlicedTensor::from(self, &slice_index));
            }
            if keepdims{
                result
            }else {
                let mut new_shape = Vec::<usize>::with_capacity(self.get_ndim()-1);
                for i in 0..*self.get_ndim() {
                    if i == dim{ continue; }
                    new_shape.push(self.get_shape()[i]);
                }
                result.reshape_(new_shape);
                result
            }
        }else {
            let mut sum = T::zero();
            let mut indices: Vec<usize> = vec![0; *self.get_ndim()];
            loop {
                sum = sum.safe_add(self.get_by_indices(&indices));

                // Increment indices
                let mut i = indices.len() - 1;
                loop {
                    indices[i] += 1;
                    if indices[i] < self.get_shape()[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == 0 {
                        return if keepdims{
                            let new_shape = vec![1; *self.get_ndim()];
                            Tensor::new(vec![sum], new_shape)
                        }else {
                            Tensor::from_num_like(sum)
                        }
                    }
                    i -= 1;
                }
            }
            
        }
    }

    fn max(&mut self, dim: Option<usize>, keepdims: bool) -> Tensor<T>
        where Self:Sized 
    {
        if let Some(dim) = dim {
            assert!(dim<*self.get_ndim(), "dim must be less than shape len");
            let mut slice_index = vec![Slice::new(..); *self.get_ndim()];
            slice_index[dim] = Slice::new(0..1);
            let mut result = SlicedTensor::from(self, &slice_index).to_tensor();
            
            for i in 1..self.get_shape()[dim]{
                slice_index[dim] = Slice::new(i..=i);
                let mut next = SlicedTensor::from(self, &slice_index);
                
                
                let mut indices: Vec<usize> = vec![0; *result.get_ndim()];
                'outer: loop {
                    if result.get_by_indices(&indices) < next.get_by_indices(&indices){
                        result.update_by_indices(&indices, next.get_by_indices(&indices));
                    }
                    // Increment indices
                    let mut i = indices.len() - 1;
                    loop {
                        indices[i] += 1;
                        if indices[i] < result.get_shape()[i] {
                            break;
                        }
                        indices[i] = 0;
                        if i == 0 {
                            break 'outer;
                        }
                        i -= 1;
                    }
                }
            }
            
            if keepdims{
                result
            }else {
                let mut new_shape = Vec::<usize>::with_capacity(self.get_ndim()-1);
                for i in 0..*self.get_ndim() {
                    if i == dim{ continue; }
                    new_shape.push(self.get_shape()[i]);
                }
                result.reshape_(new_shape);
                result
            }
        }else{
            let mut indices: Vec<usize> = vec![0; *self.get_ndim()];
            let mut max_value = *self.get_by_indices(&indices);
            loop {
                if *self.get_by_indices(&indices) > max_value {
                    max_value = *self.get_by_indices(&indices);
                }

                // Increment indices
                let mut i = indices.len() - 1;
                loop {
                    indices[i] += 1;
                    if indices[i] < self.get_shape()[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == 0 {
                        return if keepdims{
                            let new_shape = vec![1; *self.get_ndim()];
                            let mut result = Tensor::from_num_like(max_value);
                            result.reshape_(new_shape);
                            result
                        }else {
                            max_value.into()
                        }
                    }
                    i -= 1;
                }
            }
        }
    }
    
    fn min(&mut self, dim: Option<usize>, keepdims: bool) -> Tensor<T>
        where Self:Sized 
    {
        if let Some(dim) = dim {
            assert!(dim<*self.get_ndim(), "dim must be less than shape len");
            let mut slice_index = vec![Slice::new(..); *self.get_ndim()];
            slice_index[dim] = Slice::new(0..1);
            let mut result = SlicedTensor::from(self, &slice_index).to_tensor();
            
            for i in 1..self.get_shape()[dim]{
                slice_index[dim] = Slice::new(i..=i);
                let mut next = SlicedTensor::from(self, &slice_index);
                
                let mut indices: Vec<usize> = vec![0; *result.get_ndim()];
                'outer: loop {
                    if result.get_by_indices(&indices) > next.get_by_indices(&indices){
                        result.update_by_indices(&indices, next.get_by_indices(&indices));
                    }
                    // Increment indices
                    let mut i = indices.len() - 1;
                    loop {
                        indices[i] += 1;
                        if indices[i] < result.get_shape()[i] {
                            break;
                        }
                        indices[i] = 0;
                        if i == 0 {
                            break 'outer;
                        }
                        i -= 1;
                    }
                }
            }
            
            if keepdims{
                result
            }else {
                let mut new_shape = Vec::<usize>::with_capacity(self.get_ndim()-1);
                for i in 0..*self.get_ndim() {
                    if i == dim{ continue; }
                    new_shape.push(self.get_shape()[i]);
                }
                result.reshape_(new_shape);
                result
            }
        }else{
            let mut indices: Vec<usize> = vec![0; *self.get_ndim()];
            let mut min_value = *self.get_by_indices(&indices);
            loop {
                if *self.get_by_indices(&indices) < min_value {
                    min_value = *self.get_by_indices(&indices);
                }

                // Increment indices
                let mut i = indices.len() - 1;
                loop {
                    indices[i] += 1;
                    if indices[i] < self.get_shape()[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == 0 {
                        return if keepdims{
                            let new_shape = vec![1; *self.get_ndim()];
                            let mut result = Tensor::from_num_like(min_value);
                            result.reshape_(new_shape);
                            result
                        }else {
                            min_value.into()
                        }
                    }
                    i -= 1;
                }
            }
        }
    }
    
    fn argmax(&mut self, dim: Option<usize>, keepdims: bool) -> Tensor<T>
        where Self:Sized 
    {
        if let Some(dim) = dim {
            assert!(dim<*self.get_ndim(), "dim must be less than shape len");
            let mut slice_index = vec![Slice::new(..); *self.get_ndim()];
            slice_index[dim] = Slice::new(0..1);
            let mut result = SlicedTensor::from(self, &slice_index).to_tensor();
            let mut result_idx = Tensor::zeros(result.get_shape().clone());
            
            for i in 1..self.get_shape()[dim]{
                slice_index[dim] = Slice::new(i..=i);
                let mut next = SlicedTensor::from(self, &slice_index);

                let mut indices: Vec<usize> = vec![0; *result.get_ndim()];
                'outer: loop {
                    if result.get_by_indices(&indices) < next.get_by_indices(&indices){
                        result.update_by_indices(&indices, next.get_by_indices(&indices));
                        result_idx.update_by_indices(&indices, &T::from_f64(i as f64));
                    }
                    // Increment indices
                    let mut i = indices.len() - 1;
                    loop {
                        indices[i] += 1;
                        if indices[i] < result.get_shape()[i] {
                            break;
                        }
                        indices[i] = 0;
                        if i == 0 {
                            break 'outer;
                        }
                        i -= 1;
                    }
                }
            }
            
            if keepdims{
                result_idx
            }else {
                let mut new_shape = Vec::<usize>::with_capacity(self.get_ndim()-1);
                for i in 0..*self.get_ndim() {
                    if i == dim{ continue; }
                    new_shape.push(self.get_shape()[i]);
                }
                result_idx.reshape_(new_shape);
                result_idx
            }
        }else{
            let mut indices: Vec<usize> = vec![0; *self.get_ndim()];
            let mut max_value = *self.get_by_indices(&indices);
            let mut max_idx = T::zero();
            let mut idx = 0;
            loop {
                if *self.get_by_indices(&indices) > max_value {
                    max_value = *self.get_by_indices(&indices);
                    max_idx = T::from_f64(idx as f64);
                }
                idx += 1;
                // Increment indices
                let mut i = indices.len() - 1;
                loop {
                    indices[i] += 1;
                    if indices[i] < self.get_shape()[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == 0 {
                        return if keepdims{
                            let new_shape = vec![1; *self.get_ndim()];
                            let mut result = Tensor::from_num_like(max_idx);
                            result.reshape_(new_shape);
                            result
                        }else {
                            Tensor::from_num_like(max_idx)
                        }
                    }
                    i -= 1;
                }
            }
        }
    }
    
    fn argmin(&mut self, dim: Option<usize>, keepdims: bool) -> Tensor<T>
        where Self:Sized 
    {
        if let Some(dim) = dim {
            assert!(dim<*self.get_ndim(), "dim must be less than shape len");
            let mut slice_index = vec![Slice::new(..); *self.get_ndim()];
            slice_index[dim] = Slice::new(0..1);
            let mut result = SlicedTensor::from(self, &slice_index).to_tensor();
            let mut result_idx = Tensor::zeros(result.get_shape().clone()).to_tensor();
            
            for i in 1..self.get_shape()[dim]{
                slice_index[dim] = Slice::new(i..=i);
                let mut next = SlicedTensor::from(self, &slice_index);

                let mut indices: Vec<usize> = vec![0; *result.get_ndim()];
                'outer: loop {
                    if result.get_by_indices(&indices) > next.get_by_indices(&indices){
                        result.update_by_indices(&indices, next.get_by_indices(&indices));
                        result_idx.update_by_indices(&indices, &T::from_f64(i as f64));
                    }
                    // Increment indices
                    let mut i = indices.len() - 1;
                    loop {
                        indices[i] += 1;
                        if indices[i] < result.get_shape()[i] {
                            break;
                        }
                        indices[i] = 0;
                        if i == 0 {
                            break 'outer;
                        }
                        i -= 1;
                    }
                }
            }
            
            if keepdims{
                result_idx
            }else {
                let mut new_shape = Vec::<usize>::with_capacity(self.get_ndim()-1);
                for i in 0..*self.get_ndim() {
                    if i == dim{ continue; }
                    new_shape.push(self.get_shape()[i]);
                }
                result_idx.reshape_(new_shape);
                result_idx
            }
        }else{
            let mut indices: Vec<usize> = vec![0; *self.get_ndim()];
            let mut min_value = *self.get_by_indices(&indices);
            let mut min_idx = T::zero();
            let mut idx = 0;
            loop {
                if *self.get_by_indices(&indices) < min_value {
                    min_value = *self.get_by_indices(&indices);
                    min_idx = T::from_f64(idx as f64);
                }
                idx += 1;
                // Increment indices
                let mut i = indices.len() - 1;
                loop {
                    indices[i] += 1;
                    if indices[i] < self.get_shape()[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == 0 {
                        return if keepdims{
                            let new_shape = vec![1; *self.get_ndim()];
                            let mut result = Tensor::from_num_like(min_idx);
                            result.reshape_(new_shape);
                            result
                        }else {
                            Tensor::from_num_like(min_idx)
                        }
                    }
                    i -= 1;
                }
            }
        }
    }

    fn update_by_indices(&mut self, indices:&Vec<usize>, value: &T){
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        *self.get_mut_by_indices(indices) = *value;
    }
    fn add_by_indices(&mut self, indices:&Vec<usize>, value: &T){
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let mut ptr = self.get_mut_by_indices(indices);
        *ptr = ptr.safe_add(&value);
    }
    fn sub_by_indices(&mut self, indices:&Vec<usize>, value: &T){
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let mut ptr = self.get_mut_by_indices(indices);
        *ptr = ptr.safe_sub(&value);
    }
    fn mul_by_indices(&mut self, indices:&Vec<usize>, value: &T){
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let mut ptr = self.get_mut_by_indices(indices);
        *ptr = ptr.safe_mul(&value);
    }
    fn div_by_indices(&mut self, indices:&Vec<usize>, value: &T){
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let mut ptr = self.get_mut_by_indices(indices);
        *ptr = ptr.safe_div(&value);
    }
    fn pow_by_indices(&mut self, indices:&Vec<usize>, value: f32){
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        
        let mut ptr = self.get_mut_by_indices(indices);
        *ptr = ptr.safe_powf(value);
    }
    fn add_<U: TensorLike<T>>(&mut self, other: &mut U)
        where Self:Sized 
    {
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let new_shape = self.get_shape().clone();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        

        loop {
            self.add_by_indices(&indices, broadcasted_other.get_by_indices(&indices));
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
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

    fn sub_<U: TensorLike<T>>(&mut self, other: &mut U)
        where Self:Sized 
    {
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let new_shape = self.get_shape().clone();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        

        loop {
            self.sub_by_indices(&indices, broadcasted_other.get_by_indices(&indices));
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
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
    
    fn mul_<U: TensorLike<T>>(&mut self, other: &mut U)
        where Self:Sized 
    {
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let new_shape = self.get_shape().clone();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        

        loop {
            self.mul_by_indices(&indices, broadcasted_other.get_by_indices(&indices));
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
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
    
    fn div_<U: TensorLike<T>>(&mut self, other: &mut U)
        where Self:Sized 
    {
        debug_assert!(self.is_mutable(), "cannot update the values of broadcasted tensors as they have not one one mapping");
        let new_shape = self.get_shape().clone();
        let new_ndim = new_shape.len();

        let mut broadcasted_other = BroadCastTensor::from(other, &new_shape);
        let mut indices: Vec<usize> = vec![0; new_ndim];
        

        loop {
            self.div_by_indices(&indices, broadcasted_other.get_by_indices(&indices));
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
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
    fn pow_scalar(&mut self, value: f32) -> Tensor<T>
        where Self:Sized 
    {
        let mut result = self.to_tensor();
        result.pow_scalar_(value);
        result
    }

    fn pow_scalar_(&mut self, value: f32){
        let new_shape = self.get_shape().clone();
        let new_ndim = new_shape.len();
        let mut indices: Vec<usize> = vec![0; new_ndim];
        
        loop {
            self.pow_by_indices(&indices, value);
            // Increment indices
            let mut i = new_ndim - 1;
            loop {
                indices[i] += 1;
                if indices[i] < new_shape[i] {
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