use crate::spatial::SpatialNode;
use crate::vision_inference::VisionEngine;
use image::{DynamicImage, GenericImageView, Pixel};
use ndarray::{Array2, Array4};
use std::path::Path;

pub fn extract_visual_features(
    image_path: &Path,
    nodes: &[SpatialNode],
) -> Result<Array2<f64>, String> {
    if nodes.is_empty() {
        return Ok(Array2::zeros((0, 16)));
    }

    let img = image::open(image_path).map_err(|e| format!("Failed to open image: {}", e))?;
    let (width, height) = img.dimensions();

    let patch_size = 32;
    let half_patch = patch_size / 2;

    let mut batch_tensor = Array4::<f32>::zeros((nodes.len(), 3, patch_size as usize, patch_size as usize));

    for (i, node) in nodes.iter().enumerate() {
        let cx = node.centroid_x as i32;
        let cy = node.centroid_y as i32;

        let start_x = cx - half_patch as i32;
        let start_y = cy - half_patch as i32;

        for py in 0..patch_size {
            for px in 0..patch_size {
                let ix = start_x + px as i32;
                let iy = start_y + py as i32;

                if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
                    let pixel = img.get_pixel(ix as u32, iy as u32).to_rgb();
                    batch_tensor[[i, 0, py as usize, px as usize]] = pixel[0] as f32 / 255.0;
                    batch_tensor[[i, 1, py as usize, px as usize]] = pixel[1] as f32 / 255.0;
                    batch_tensor[[i, 2, py as usize, px as usize]] = pixel[2] as f32 / 255.0;
                }
            }
        }
    }

    // Load vision model (for now assuming it's in the current dir)
    let engine = VisionEngine::new("vision_model.onnx")
        .map_err(|e| format!("Failed to load vision_model.onnx: {:?}", e))?;

    let shape = &[nodes.len(), 3, patch_size as usize, patch_size as usize];
    let tract_tensor = tract_onnx::prelude::Tensor::from_shape(shape, batch_tensor.into_raw_vec().as_slice())
        .map_err(|e| format!("Failed to create Tensor: {:?}", e))?;
    
    let output = engine.infer(tract_tensor).map_err(|e| format!("Vision inference failed: {:?}", e))?;

    let output_view = output.to_array_view::<f32>().map_err(|e| format!("Failed to parse output view: {:?}", e))?;
    let out_shape = output_view.shape();
    
    if out_shape.len() != 2 || out_shape[1] != 16 {
        return Err(format!("Expected shape (N, 16) from vision model, got {:?}", out_shape));
    }

    let mut final_features = Array2::<f64>::zeros((nodes.len(), 16));
    for r in 0..nodes.len() {
        for c in 0..16 {
            final_features[[r, c]] = output_view[[r, c]] as f64;
        }
    }

    Ok(final_features)
}
