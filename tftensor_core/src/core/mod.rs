mod tensor;
mod dtype;
mod random;
pub mod print;
mod utils;
mod ops;
mod tensor_other;
// use tensor::TensorLike;
pub use ops::TensorLike;
pub use tensor::Tensor;
pub use tensor::Order;

// pub use print::PrintHelper; 
pub use print::PrintHelper;

pub use random::LCG;

pub use tensor_other::TransposeTensor;
pub use tensor_other::ReshapeTensor;
pub use tensor_other::BroadCastTensor;
pub use tensor_other::SlicedTensor;
pub use tensor_other::Slice;

pub use dtype::NumLike;

