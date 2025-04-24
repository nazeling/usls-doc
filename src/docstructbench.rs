use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;

use crate::{models::YOLO, Annotator, DataLoader, ModelChoice, Options, Bbox};

#[derive(Serialize)]
pub struct Detection {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Serialize)]
pub struct DetectionOutput {
    pub bboxes: Vec<Detection>,
}

/// Detects document elements in a batch of images using a YOLO model selected by `model_choice`.
///
/// # Arguments
/// * `device` - The device to run the model on (e.g., "auto", "cuda:0", "cpu:0").
/// * `image_paths` - A slice of paths to the input images.
/// * `model_choice` - The YOLO model to use (`DocSynth` or `DocStructBench`).
///
/// # Returns
/// A vector of tuples, each containing an image path and its corresponding detections in JSON-compatible format.
///
/// # Notes
/// - Annotated images are saved to the "doclayout-yolo" directory.
/// - Detections with label "abandon" and confidence < 0.50 are filtered out (applies only to `DocStructBench`).
/// - The output detections are in `xywh` format with `i32` coordinates to match the expectations of `json_bbox::create_mask_from_json`.
pub fn detect_documents(
    device: &str,
    image_paths: &[PathBuf],
    model_choice: ModelChoice,
) -> Result<Vec<(PathBuf, DetectionOutput)>> {
    println!("Starting detect_documents with device: {}, model: {:?}", device, model_choice);
    println!("Input image paths: {:?}", image_paths);

    // Load images from the provided paths
    let image_paths_str: Vec<_> = image_paths.iter().map(|p| p.to_str().unwrap()).collect();
    println!("Converted image paths to strings: {:?}", image_paths_str);
    let images = DataLoader::try_read_batch(&image_paths_str)?;
    println!("Loaded {} images", images.len());

    // Configure and initialize the YOLO model based on model_choice
    let yolo_config = match model_choice {
        ModelChoice::DocStructBench => Options::doclayout_yolo_docstructbench(),
        ModelChoice::DocSynth => Options::doclayout_yolo_docsynth(),
    }
    .with_model_device(device.try_into()?)
    .commit()?;
    println!("YOLO configuration committed: {:?}", yolo_config);

    let mut yolo_model = YOLO::new(yolo_config)?;
    println!("YOLO model initialized");

    // Run YOLO detection on the batch of images
    let yolo_results = yolo_model.forward(&images)?;
    println!("YOLO detection completed, results: {} images", yolo_results.len());

    // Annotate images (saves annotated versions to disk)
    let yolo_annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout("doclayout-yolo");
    yolo_annotator.annotate(&images, &yolo_results);
    println!("Annotation completed, images saved to 'doclayout-yolo'");

    // Process YOLO results into DetectionOutput structs
    let detections_per_image: Vec<(PathBuf, DetectionOutput)> = image_paths
        .iter()
        .zip(yolo_results.iter())
        .map(|(path, result)| {
            let detections: Vec<Detection> = result
                .bboxes()
                .map(|bboxes| {
                    println!("Processing bboxes for {}: {} detections", path.display(), bboxes.len());
                    bboxes
                        .iter()
                        .filter_map(|bbox| {
                            let keep = !(bbox
                                .name()
                                .as_ref()
                                .map_or(false, |n| n.to_lowercase() == "abandon")
                                && bbox.confidence() < 0.50);
                            if !keep {
                                println!(
                                    "Filtered out bbox: name={:?}, confidence={}",
                                    bbox.name(), bbox.confidence()
                                );
                            }
                            if keep {
                                let (x, y, w, h) = bbox.xywh();
                                Some(Detection {
                                    x: x.round() as i32,
                                    y: y.round() as i32,
                                    width: w.round() as i32,
                                    height: h.round() as i32,
                                })
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            println!("Final detections for {}: {} bboxes", path.display(), detections.len());
            (path.clone(), DetectionOutput { bboxes: detections })
        })
        .collect();

    println!("Returning {} detection results", detections_per_image.len());
    Ok(detections_per_image)
}