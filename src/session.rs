use crate::{model::Model, DataLoader, Options};
use anyhow::Result;
use image::DynamicImage;

pub struct Session {
    model: Model,
    options: Options,
}

impl Session {
    pub fn new(model_path: &str, device: &str, options: Options) -> Result<Self> {
        let model = Model::new(model_path, device)?;
        model.debug_summary();
        Ok(Self { model, options })
    }

    pub fn process_folder(&self, folder: &str) -> Result<Vec<Vec<ort::value::Value>>> {
        let dataloader = DataLoader::new(folder)?.build()?;
        let mut results = Vec::new();

        for (images, _paths) in dataloader {
            let outputs = self.process_batch(&images)?;
            results.push(outputs);
        }

        Ok(results)
    }

    pub fn process_batch(&self, images: &[DynamicImage]) -> Result<Vec<ort::value::Value>> {
        let mut processor = self.options.to_processor()?;
        let _input_tensor = processor.process_images(images)?; // Keep for side effects (e.g., setting scale factors)

        let rgb_images: Vec<_> = images.iter().map(|img| img.to_rgb8()).collect();
        let tensors = crate::DataLoader::try_from_rgb8(&rgb_images)?;

        self.model.run(tensors)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn options(&self) -> &Options {
        &self.options
    }
}
