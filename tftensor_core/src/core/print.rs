use core::fmt;

use super::{dtype::{DType, NumLike}, TensorLike, Tensor};
use super::tensor_other::{BroadCastTensor, SlicedTensor, ReshapeTensor, Slice, TransposeTensor};
const PRINT_THRESHOLD: usize = 5; // must be more then 7 ...
const PRINT_MIN: usize = 2; // 6 / 2
// assert PRINT_THRESHOLD > 2*PRINT_MIN

const PRINT_LEN: usize = 7;

fn format_number<T: NumLike>(num: &T)->String{
    match T::dtype(){
        DType::F32 | DType::F64 => {
            if num.positive() {
                format!(" {:.width$}", num, width = PRINT_LEN)
            } else {
                format!("{:+.width$}", num, width = PRINT_LEN)
            }
        },
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            let normal = format!("{}", num);
            if normal.len() <= PRINT_LEN {
                if num.positive() {
                    format!(" {:>width$}", num, width=PRINT_LEN)
                }else{
                    format!("{:>width$}", num, width=PRINT_LEN+1)
                }
            } else {
                // Format the number in scientific notation
                let scientific_str = format!("{:e}", num.to_f64());
                let e_pos = scientific_str.find('e').unwrap();
                let (coeff, exponent) = scientific_str.split_at(e_pos); // 'e+00' or 'e-00'
                let coeff = coeff.to_string();
                
                if num.positive() {
                    let coefficient_len = PRINT_LEN - exponent.len();
                    let coeff = if coeff.len() > coefficient_len {
                        coeff[..coefficient_len].to_string()
                    } else {
                        // Otherwise, pad it to the left with zeros if needed
                        format!("{:0>width$}", coeff, width = coefficient_len)
                    };
                    format!(" {}{}", coeff, exponent)
                } else {
                    let coefficient_len = PRINT_LEN + 1 - exponent.len();
                    let coeff = if coeff.len() > coefficient_len {
                        coeff[..coefficient_len].to_string()
                    } else {
                        // Otherwise, pad it to the left with zeros if needed
                        format!("{:0>width$}", coeff, width = coefficient_len)
                    };
                    format!("{}{}", coeff, exponent)
                }
            }
        }
        _ => {
            let normal = format!("{}", num);
            if normal.len() <= PRINT_LEN {
                format!("{:>width$}", num, width=PRINT_LEN)
            }else{
                // Format the number in scientific notation
                let scientific_str = format!("{:e}", num.to_f64());
                let e_pos = scientific_str.find('e').unwrap();
                let (coeff, exponent) = scientific_str.split_at(e_pos); // 'e+00' or 'e-00'
                let coeff = coeff.to_string();
                let coefficient_len = PRINT_LEN - exponent.len();
                let coeff = if coeff.len() > coefficient_len {
                    coeff[..coefficient_len].to_string()
                } else {
                    // Otherwise, pad it to the left with zeros if needed
                    format!("{:0>width$}", coeff, width = coefficient_len)
                };

                format!("{}{}", coeff, exponent)
            }
        }
    }
}
pub trait PrintHelper<T: NumLike>
where 
    Self : TensorLike<T>
{
    fn print_helper(&mut self, indices: &mut Vec<usize>, depth: usize, spacing_size:usize)->String{
        
        let mut result: String = String::new();
        if depth == self.get_shape().len() - 1 {
            result += "[";
            if self.get_shape()[depth] > PRINT_THRESHOLD {
                for i in 0..PRINT_MIN {
                    indices[depth] = i;
                    result+=format!("{}, ", format_number(self.get_by_indices(indices))).as_str();
                }
                result += "..., ";
                for i in (self.get_shape()[depth]-PRINT_MIN)..self.get_shape()[depth] {
                    indices[depth] = i;
                    if i < self.get_shape()[depth] - 1 {
                        result+=format!("{}, ", format_number(self.get_by_indices(indices))).as_str();
                    } else {
                        result+=format!("{}", format_number(self.get_by_indices(indices))).as_str();
                    }
                }
            } else {
                for i in 0..self.get_shape()[depth] {
                    indices[depth] = i;
                    if i < self.get_shape()[depth] - 1 {
                        result+=format!("{}, ", format_number(self.get_by_indices(indices))).as_str();
                    } else {
                        result+=format!("{}", format_number(self.get_by_indices(indices))).as_str();
                    }
                }
            }
            result += "]";
            result
        } else {
            result += "[";
            if self.get_shape()[depth] > PRINT_THRESHOLD {
                for i in 0..PRINT_MIN {
                    indices[depth] = i;
                    result += self.print_helper(indices, depth + 1, spacing_size).as_str();
                    result += ",";
                    for _ in 0..self.get_shape().len() - depth - 1 { result += "\n"; }
                    for _ in 0..depth+spacing_size+1 { result += " "; }
                }
                result += "...,";
                for _ in 0..self.get_shape().len() - depth - 1 { result += "\n"; }
                for _ in 0..depth+spacing_size+1 { result += " "; }
                for i in (self.get_shape()[depth]-PRINT_MIN)..self.get_shape()[depth] {
                    indices[depth] = i;
                    result += self.print_helper(indices, depth + 1, spacing_size).as_str();
                    if i < self.get_shape()[depth] - 1 {
                        result += ",";
                        for _ in 0..self.get_shape().len() - depth - 1 { result += "\n"; }
                        for _ in 0..depth+spacing_size+1 { result += " "; }
                    }
                }
            } else {
                for i in 0..self.get_shape()[depth] {
                    indices[depth] = i;
                    result += self.print_helper(indices, depth + 1, spacing_size).as_str();
                    if i < self.get_shape()[depth] - 1 {
                        result += ",";
                        for _ in 0..self.get_shape().len() - depth - 1 { result += "\n"; }
                        for _ in 0..depth+spacing_size+1 { result += " "; }
                    }
                }
            }
            result += "]";
            result
        }
    }
    fn repr(&mut self) -> String {
        let mut result = String::new();
        result+="Tensor(";
        let mut indices = vec![0; self.get_shape().len()];
        result+=self.print_helper(&mut indices, 0, 7).as_str();
        result+=", shape=(";
        for i in 0..self.get_shape().len()-1 {
            result+=format!("{}, ", self.get_shape()[i]).as_str();
        }
        result+=format!("{})", self.get_shape()[self.get_shape().len()-1]).as_str();
        
        format!("{}, size={}, dtype={})", result, self.get_size(), self.get_dtype())
    }
    fn reprstr(&mut self, spacing_size:usize) -> String {
        let mut indices = vec![0; self.get_shape().len()];
        self.print_helper(&mut indices, 0, spacing_size)
    }
}


impl<T: NumLike> PrintHelper<T> for Tensor<T>{}
impl<'a, T, U> PrintHelper<T> for TransposeTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{}
impl<'a, T, U> PrintHelper<T> for ReshapeTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{}
impl<'a, T, U> PrintHelper<T> for BroadCastTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{}
impl<'a, T, U> PrintHelper<T> for SlicedTensor<'a, T, U> 
where 
    T: NumLike,
    U: TensorLike<T>
{}

impl fmt::Display for DType{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::U8  => write!(f, "u8"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::I8  => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
        }
    }
}

impl std::fmt::Display for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(step) = self.step{
            if let Some(start) = self.start {
                if let Some(end) = self.end {
                    write!(f, "{}:{}:{}", start, end, step)
                }else{
                    write!(f, "{}::{}", start,step)
                }           
            }else{
                if let Some(end) = self.end {
                    write!(f, ":{}:{}", end, step)
                }else{
                    write!(f, "::{}", step)
                }           
            }
        }else{
            if let Some(start) = self.start {
                if let Some(end) = self.end {
                    if start == end{
                        write!(f, "{}", start)
                    }else {
                        write!(f, "{}:{}", start, end)
                    }
                }else{
                    write!(f, "{}:", start)
                }           
            }else{
                if let Some(end) = self.end {
                    write!(f, ":{}", end)
                }else{
                    write!(f, ":")
                }           
            }
        }
    }
}
impl std::fmt::Debug for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!("{}", self))
    }
}