use std::clone::Clone;
use std::fmt::Display;
use super::random::LCG;

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg};

// pub enum Device {
//     CPU,
//     GPU,
// }

#[derive(PartialEq)]
pub enum DType {
    F32, F64,
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    // BIT1_5
}
impl DType{
    pub fn from<T: NumLike>()->Self{
        T::dtype()
    }    
    pub fn byte_size(&self) -> usize{
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4, 
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }
    pub fn to_le_bytes(&self) -> [u8; 8]{
        match self {
            DType::F32 => 1_usize.to_le_bytes(), 
            DType::F64 => 2_usize.to_le_bytes(),
            DType::U8 => 3_usize.to_le_bytes(),
            DType::U16 => 4_usize.to_le_bytes(),
            DType::U32 => 5_usize.to_le_bytes(), 
            DType::U64 => 6_usize.to_le_bytes(),
            DType::I8 => 7_usize.to_le_bytes(),
            DType::I16 => 8_usize.to_le_bytes(),
            DType::I32 => 9_usize.to_le_bytes(), 
            DType::I64 => 10_usize.to_le_bytes(),
        }
    }
    pub fn from_le_bytes(data: [u8; 8])->Self{
        match usize::from_le_bytes(data) {
            1 => DType::F32,
            2 => DType::F64,
            3 => DType::U8 ,
            4 => DType::U16,
            5 => DType::U32,
            6 => DType::U64,
            7 => DType::I8 ,
            8 => DType::I16,
            9 => DType::I32,
            10 => DType::I64,
            _ => panic!("Wrong Dtype value"),
        }
    }
    
}
// Custom type `bit1_5`
// pub struct Bit1_5 {
//     data: Option<bool>,
// }
pub trait NumLike: Clone + Copy + Display + std::cmp::PartialOrd + 
    AddAssign + SubAssign + MulAssign + DivAssign +
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>{
    fn zero() -> Self;
    fn one() -> Self;
    fn dtype() -> DType;
    fn random(lgc: &mut LCG) -> Self;
    fn positive(&self) -> bool;
    fn to_f64(&self) -> f64;
    fn from_f64(value: f64) -> Self;
    fn bytes(&self) -> Vec<u8>;
    fn from_le_bytes(data: Vec<u8>) -> Self;

    fn safe_add(&self, other: &Self) -> Self;
    fn safe_sub(&self, other: &Self) -> Self;
    fn safe_mul(&self, other: &Self) -> Self;
    fn safe_div(&self, other: &Self) -> Self;
    fn safe_powf(&self, v: f32) -> Self;
}

impl NumLike for f32 {
    fn zero() -> Self {0.0}
    fn one() -> Self {1.0}
    fn dtype() -> DType {DType::F32}
    fn random(lgc: &mut LCG) -> Self {lgc.random() as f32}
    fn positive(&self) -> bool {*self >= 0.0}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as f32}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }

    fn safe_add(&self, other: &Self) -> Self{self.add(other)}
    fn safe_sub(&self, other: &Self) -> Self{self.sub(other)}
    fn safe_mul(&self, other: &Self) -> Self{self.mul(other)}
    fn safe_div(&self, other: &Self) -> Self{self.div(other)}
    fn safe_powf(&self, v: f32)->Self{self.powf(v)}
}
impl NumLike for f64 {
    fn zero() -> Self {0.0}
    fn one() -> Self {1.0}
    fn random(lgc: &mut LCG) -> Self {lgc.random()}
    fn dtype() -> DType {DType::F64}
    fn positive(&self) -> bool {*self >= 0.0}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.add(other)}
    fn safe_sub(&self, other: &Self) -> Self{self.sub(other)}
    fn safe_mul(&self, other: &Self) -> Self{self.mul(other)}
    fn safe_div(&self, other: &Self) -> Self{self.div(other)}
    fn safe_powf(&self, v: f32)->Self{self.powf(v as f64)}
}

impl NumLike for i8 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {lgc.randint(i8::MIN as i64, i8::MAX as i64) as i8}
    fn dtype() -> DType {DType::I8}
    fn positive(&self) -> bool {*self >= 0}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as i8}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}
impl NumLike for i16 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {lgc.randint(i16::MIN as i64, i16::MAX as i64) as i16}
    fn dtype() -> DType {DType::I16}
    fn positive(&self) -> bool {*self >= 0}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as i16}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}
impl NumLike for i32 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {lgc.randint(i32::MIN as i64, i32::MAX as i64) as i32}
    fn dtype() -> DType {DType::I32}
    fn positive(&self) -> bool {*self >= 0}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as i32}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}
impl NumLike for i64 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {(lgc.randint(0, i32::MAX as i64 - 1) as i64) * 4 + lgc.randint(0, 4) as i64}
    fn dtype() -> DType {DType::I64}
    fn positive(&self) -> bool {*self >= 0}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as i64}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}

impl NumLike for u8 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {lgc.randint(0, u8::MAX as i64) as u8}
    fn dtype() -> DType {DType::U8}
    fn positive(&self) -> bool {true}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as u8}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}
impl NumLike for u16 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {lgc.randint(0, u16::MAX as i64) as u16}
    fn dtype() -> DType {DType::U16}
    fn positive(&self) -> bool {true}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as u16}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}
impl NumLike for u32 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {lgc.randint(0, u32::MAX as i64) as u32}
    fn dtype() -> DType {DType::U32}
    fn positive(&self) -> bool {true}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as u32}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}
impl NumLike for u64 {
    fn zero() -> Self {0}
    fn one() -> Self {1}
    fn random(lgc: &mut LCG) -> Self {(lgc.randint(0, i32::MAX as i64 - 1) as u64) * 4 + lgc.randint(0, 4) as u64}
    fn dtype() -> DType {DType::U64}
    fn positive(&self) -> bool {true}
    fn to_f64(&self) -> f64 { *self as f64}
    fn from_f64(value: f64) -> Self {value as u64}
    fn bytes(&self) -> Vec<u8>{
        self.to_le_bytes().to_vec()
    }
    fn from_le_bytes(data: Vec<u8>) -> Self{
        Self::from_le_bytes(data.try_into().expect("Error while converting to le bytes NumLike."))
    }
    fn safe_add(&self, other: &Self) -> Self{self.wrapping_add(*other)}
    fn safe_sub(&self, other: &Self) -> Self{self.wrapping_sub(*other)}
    fn safe_mul(&self, other: &Self) -> Self{self.wrapping_mul(*other)}
    fn safe_div(&self, other: &Self) -> Self{self.wrapping_div(*other)}
    fn safe_powf(&self, v: f32)->Self{self.wrapping_pow(v as u32)}
}