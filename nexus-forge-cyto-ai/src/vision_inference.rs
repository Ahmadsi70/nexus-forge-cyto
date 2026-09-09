use tract_onnx::prelude::*;
use std::path::Path;

/// A struct that encapsulates the ONNX runtime via `tract`.
/// This is used to run native Vision models (like HoVer-Net or MedSAM) 
/// directly inside the Rust engine without relying on Python.
pub struct VisionEngine {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl VisionEngine {
    /// Loads an ONNX model from the specified file path.
    pub fn new<P: AsRef<Path>>(model_path: P) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            // Define input signature if needed, or let tract infer it
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { model })
    }

    /// Runs inference on an image tensor.
    /// In a real pipeline, the input is a Normalized RGB Image Tensor (e.g., [1, 3, 256, 256]).
    /// Returns the raw output tensor (e.g., probability masks).
    pub fn infer(&self, image_tensor: Tensor) -> TractResult<Tensor> {
        // Run the model on the input tensor
        let mut result = self.model.run(tvec!(image_tensor.into()))?;
        
        // Return the first output tensor
        Ok(result.remove(0).into_tensor())
    }
}
