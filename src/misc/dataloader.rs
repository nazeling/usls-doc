use image::DynamicImage;
use indicatif::ProgressBar;
use log::{info, warn};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use rayon::prelude::*; // For parallel processing
use pdfium_render::prelude::*; // Assuming this crate for PDFium bindings
use crate::{build_progress_bar, Location, MediaType};
use ort::value::Value;
use ndarray::Array;
use anyhow::{anyhow, Result};

type TempReturnType = (Vec<DynamicImage>, Vec<PathBuf>);

pub struct DataLoaderIterator {
    receiver: mpsc::Receiver<TempReturnType>,
    progress_bar: Option<ProgressBar>,
    batch_size: u64,
}

impl Iterator for DataLoaderIterator {
    type Item = TempReturnType;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.progress_bar {
            None => self.receiver.recv().ok(),
            Some(progress_bar) => {
                match self.receiver.recv().ok() {
                    Some(item) => {
                        progress_bar.inc(self.batch_size);
                        Some(item)
                    }
                    None => {
                        progress_bar.set_prefix("Iterated");
                        progress_bar.set_style(
                            match indicatif::ProgressStyle::with_template(crate::PROGRESS_BAR_STYLE_FINISH_2) {
                                Ok(x) => x,
                                Err(err) => panic!("Failed to set style for progressbar: {}", err),
                            },
                        );
                        progress_bar.finish();
                        None
                    }
                }
            }
        }
    }
}

impl IntoIterator for DataLoader {
    type Item = TempReturnType;
    type IntoIter = DataLoaderIterator;

    fn into_iter(self) -> Self::IntoIter {
        let progress_bar = if self.with_pb {
            build_progress_bar(
                self.nf,
                "Iterating",
                Some("Images"),
                crate::PROGRESS_BAR_STYLE_CYAN_2,
            ).ok()
        } else {
            None
        };

        DataLoaderIterator {
            receiver: self.receiver,
            progress_bar,
            batch_size: self.batch_size as _,
        }
    }
}

/// Loads and manages image data for document processing, optimized for batches from folders or PDFs.
/// Supports single images (CPU-only), folders of images, remote URLs, and PDFs via pdfium.
pub struct DataLoader {
    paths: Option<VecDeque<PathBuf>>,    // Queue of image paths
    media_type: MediaType,               // Image or PDF
    batch_size: usize,                   // Batch size for iteration
    bound: usize,                        // Channel buffer size
    receiver: mpsc::Receiver<TempReturnType>,
    nf: u64,                             // Number of images
    with_pb: bool,                       // Progress bar flag
}

impl TryFrom<&str> for DataLoader {
    type Error = anyhow::Error;

    fn try_from(str: &str) -> Result<Self, Self::Error> {
        Self::new(str)
    }
}

impl DataLoader {
    pub fn new(source: &str) -> Result<Self> {
        let source_path = Path::new(source);
        let (paths, media_type, nf) = if source.starts_with("http://") || source.starts_with("https://") {
            // Remote image URL (single image)
            (
                Some(VecDeque::from([source_path.to_path_buf()])),
                MediaType::Image(Location::Remote),
                1
            )
        } else if source_path.exists() {
            if source_path.is_file() {
                if source_path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                    // PDF file
                    let images = Self::pdf_to_images(source_path)?;
                    let nf = images.len() as u64;
                    (
                        Some(VecDeque::from(images)),
                        MediaType::Image(Location::Local),
                        nf
                    )
                } else {
                    // Single image file
                    (
                        Some(VecDeque::from([source_path.to_path_buf()])),
                        MediaType::Image(Location::Local),
                        1
                    )
                }
            } else if source_path.is_dir() {
                // Directory of images (batch case)
                let paths_sorted = Self::load_from_folder(source_path)?;
                let nf = paths_sorted.len() as u64;
                (
                    Some(VecDeque::from(paths_sorted)),
                    MediaType::Image(Location::Local),
                    nf
                )
            } else {
                anyhow::bail!("Invalid source: {:?}", source_path);
            }
        } else {
            anyhow::bail!("Source not found: {:?}", source_path);
        };

        info!("Found {:?} with {} images", media_type, nf);

        Ok(DataLoader {
            paths,
            media_type,
            bound: 50,
            receiver: mpsc::sync_channel(1).1,
            batch_size: if nf == 1 { 1 } else { 8 }, // Default batch size 8 for multi-image
            nf,
            with_pb: true,
        })
    }

    pub fn with_bound(mut self, x: usize) -> Self {
        self.bound = x;
        self
    }

    pub fn with_batch(mut self, x: usize) -> Self {
        self.batch_size = x;
        self
    }

    pub fn with_progress_bar(mut self, x: bool) -> Self {
        self.with_pb = x;
        self
    }

    pub fn build(mut self) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<TempReturnType>(self.bound);
        self.receiver = receiver;
        let batch_size = self.batch_size;
        let data = self.paths.take().unwrap_or_default();
        let media_type = self.media_type.clone();

        std::thread::spawn(move || {
            Self::producer_thread(sender, data, batch_size, media_type);
        });

        Ok(self)
    }

    fn producer_thread(
        sender: mpsc::SyncSender<TempReturnType>,
        mut data: VecDeque<PathBuf>,
        batch_size: usize,
        media_type: MediaType,
    ) {
        let mut yis: Vec<DynamicImage> = Vec::with_capacity(batch_size);
        let mut yps: Vec<PathBuf> = Vec::with_capacity(batch_size);

        if let MediaType::Image(_) = media_type {
            while let Some(path) = data.pop_front() {
                match Self::try_read(&path) {
                    Err(err) => {
                        warn!("{:?} | {:?}", path, err);
                        continue;
                    }
                    Ok(img) => {
                        yis.push(img);
                        yps.push(path);
                    }
                }
                if yis.len() == batch_size
                    && sender.send((std::mem::take(&mut yis), std::mem::take(&mut yps))).is_err()
                {
                    break;
                }
            }
            if !yis.is_empty() && sender.send((yis, yps)).is_err() {
                info!("Receiver dropped, stopping production");
            }
        }
    }

    pub fn load_from_folder<P: AsRef<Path>>(path: P) -> Result<Vec<PathBuf>> {
        let entries: Vec<_> = std::fs::read_dir(&path)?
            .filter_map(|entry| entry.ok())
            .collect();

        let paths: Vec<PathBuf> = entries
            .par_iter()
            .filter_map(|entry| {
                let p = entry.path();
                if p.is_file() && Self::is_image_file(&p) {
                    Some(p)
                } else {
                    None
                }
            })
            .collect();

        if paths.is_empty() {
            anyhow::bail!("No valid images found in directory: {:?}", path.as_ref());
        }
        Ok(paths)
    }

    fn is_image_file(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp"))
            .unwrap_or(false)
    }

    pub fn pdf_to_images<P: AsRef<Path>>(path: P) -> Result<Vec<PathBuf>> {
        let pdfium = Pdfium::new(Pdfium::bind_to_system_library()?);
        let doc = pdfium.load_pdf_from_file(path.as_ref(), None)?;
        let temp_dir = std::env::temp_dir();
        let mut image_paths = Vec::new();

        for (i, page) in doc.pages().iter().enumerate() {
            let output_path = temp_dir.join(format!("page_{}.png", i));
            let bitmap = page.render_with_config(
                &PdfRenderConfig::new().set_target_width(2480),
            )?;
            let image = bitmap.as_image();
            image.save(&output_path)?;
            image_paths.push(output_path);
        }
        Ok(image_paths)
    }

    pub fn try_read<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        let path = path.as_ref();
        let img = image::ImageReader::open(path)
            .map_err(|err| anyhow!("Failed to open image at {:?}: {:?}", path, err))?
            .decode()
            .map_err(|err| anyhow!("Failed to decode image at {:?}: {:?}", path, err))?;
        Ok(img)
    }

    pub fn try_read_batch<P: AsRef<Path> + std::fmt::Debug + Sync>(
        paths: &[P],
    ) -> Result<Vec<DynamicImage>> {
        let images: Vec<DynamicImage> = paths.par_iter()
            .filter_map(|path| Self::try_read(path).ok())
            .collect();
        if images.is_empty() {
            anyhow::bail!("Failed to read any images from batch");
        }
        Ok(images)
    }
	pub fn try_from_rgb8(images: &[image::RgbImage]) -> Result<Vec<Value>> {
    let mut tensors = Vec::with_capacity(images.len());
    for img in images {
        let (width, height) = img.dimensions();
        let shape = [1, 3, height as usize, width as usize];
        let data: Vec<f32> = img
            .pixels()
            .flat_map(|p| p.0.iter().map(|v| *v as f32 / 255.0))
            .collect();

        // Create the ndarray from owned data (pass tensor_array by value, not by reference)
        let tensor_array = Array::from_shape_vec(shape, data)
            .map_err(|e| anyhow!("Failed to create tensor: {}", e))?;

        // Create a tensor. Value::from_array returns a Value specialized on TensorValueType<f32>
        let tensor_specialized = Value::from_array(tensor_array)
            .map_err(|e| anyhow!("Failed to create ONNX tensor: {}", e))?;
        
        // Convert the specialized tensor into a dynamic type if required
        tensors.push(tensor_specialized.into());
    }
    Ok(tensors)
}
}