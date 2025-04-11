use crate::string_random;
use crate::{Bbox, LogitsSampler, Options, Polygon, Text, Xs, Ys, X, Y};
use anyhow::Result;
use dyn_clone::DynClone;
use fast_image_resize::{
    images::{CroppedImageMut, Image},
    pixels::PixelType,
    FilterType, ResizeAlg, ResizeOptions, Resizer,
};
use image::imageops;
use image::{DynamicImage, GenericImageView, ImageBuffer, RgbImage};
use ndarray::{s, Array, Axis};
use rayon::prelude::*;
use serde_json::Value;
use std::fmt;
use tokenizers::{Encoding, Tokenizer};

//---------------------------------------------------------------------------
// Define the global TokenizerTrait with DynClone for cloning trait objects.
pub trait TokenizerTrait: Send + Sync + DynClone + std::fmt::Debug {
    fn encode(&self, text: &str, skip_special_tokens: bool) -> Result<Encoding>;
    fn encode_batch(&self, texts: Vec<String>, skip_special_tokens: bool) -> Result<Vec<Encoding>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn decode_batch(&self, ids: &[&[u32]], skip_special_tokens: bool) -> Result<Vec<String>>;
}

dyn_clone::clone_trait_object!(TokenizerTrait);

#[derive(Clone)]
pub struct MyTokenizer(pub Tokenizer);

impl fmt::Debug for MyTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MyTokenizer(<tokenizers::Tokenizer>)")
    }
}

impl TokenizerTrait for MyTokenizer {
    fn encode(&self, text: &str, skip_special_tokens: bool) -> Result<Encoding> {
        self.0
            .encode(text, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!(e))
    }
    fn encode_batch(&self, texts: Vec<String>, skip_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.0
            .encode_batch(texts, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!(e))
    }
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.0
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!(e))
    }
    fn decode_batch(&self, ids: &[&[u32]], skip_special_tokens: bool) -> Result<Vec<String>> {
        self.0
            .decode_batch(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

//---------------------------------------------------------------------------
// Define structs for returning resize and process results.
pub struct ResizedImage {
    pub buffer: Vec<u8>,
    pub original_size: (u32, u32),
    pub scale_factors: Vec<f32>,
}

pub struct ProcessedImages {
    pub x: X,
    pub original_sizes: Vec<(u32, u32)>,
    pub scale_factors: Vec<Vec<f32>>,
}

pub struct CroppedImage {
    pub image: DynamicImage,
    pub bbox: Bbox,
}

impl ProcessedImages {
    /// Converts ProcessedImages to an Xs instance with a single X tensor.
    pub fn to_xs(self, key: Option<&str>) -> Xs {
        let mut xs = Xs::default();
        let key = key
            .map(|k| k.to_string())
            .unwrap_or_else(|| string_random(5));
        xs.push_kv(&key, self.x).expect("Failed to insert into Xs");
        xs
    }
}

//---------------------------------------------------------------------------
// Processor definition
#[derive(aksr::Builder, Clone, Debug)]
pub struct Processor {
    image_width: u32,
    image_height: u32,
    resize_mode: ResizeMode,
    resize_filter: &'static str,
    padding_value: u8,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    nchw: bool,
    tokenizer: Option<Box<dyn TokenizerTrait>>,
    vocab: Vec<String>,
    unsigned: bool,
    logits_sampler: Option<LogitsSampler>,
    options: Options,
    config: Option<Value>,
    image_sizes: Vec<(u32, u32)>,
    scale_factors: Vec<[f32; 2]>,
}

impl Processor {
    /// Creates a new Processor instance, loading the configuration once.
    pub fn new(options: Options) -> Result<Self> {
        let config = options
            .generation_config_file
            .as_ref()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|s| serde_json::from_str(&s).ok());
        Ok(Self {
            image_width: 0,
            image_height: 0,
            resize_mode: ResizeMode::FitAdaptive,
            resize_filter: "Bilinear",
            padding_value: 114,
            do_normalize: true,
            image_mean: vec![],
            image_std: vec![],
            nchw: true,
            tokenizer: None,
            vocab: vec![],
            unsigned: false,
            logits_sampler: None,
            options,
            config,
            image_sizes: vec![],
            scale_factors: vec![],
        })
    }

    pub fn get_options(&self) -> &Options {
        &self.options
    }

    /// Converts an RGB image to grayscale efficiently using imageops.
    pub fn rgb_to_grayscale(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let rgb = image.to_rgb8();
        let gray = imageops::grayscale(&rgb);
        Ok(DynamicImage::ImageLuma8(gray))
    }

    /// Converts a grayscale image to RGB by duplicating the channel.
    pub fn grayscale_to_rgb(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let gray = image.to_luma8();
        let (width, height) = gray.dimensions();
        let rgb: RgbImage = ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = gray.get_pixel(x, y);
            image::Rgb([pixel[0], pixel[0], pixel[0]])
        });
        Ok(DynamicImage::ImageRgb8(rgb))
    }

    /// Ensures the image is in RGB format.
    pub fn ensure_rgb(&self, image: DynamicImage) -> DynamicImage {
        match image {
            DynamicImage::ImageRgb8(_) => image,
            DynamicImage::ImageRgba8(img) => {
                DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(img).to_rgb8())
            }
            _ => self
                .grayscale_to_rgb(&image)
                .unwrap_or_else(|_| image.to_rgb8().into()),
        }
    }

    /// Processes a batch of images, returning processed tensor and metadata.
    pub fn process_images(&mut self, xs: &[DynamicImage]) -> Result<ProcessedImages> {
        if xs.is_empty() {
            anyhow::bail!("No images provided for processing");
        }
        let rgb_images: Vec<DynamicImage> =
            xs.iter().map(|img| self.ensure_rgb(img.clone())).collect();
        let resized_images: Vec<ResizedImage> = rgb_images
            .par_iter()
            .map(|img| self.resize(img))
            .collect::<Result<Vec<_>>>()?;

        // Ensure all resized images have the same size for stacking
        let first_len = resized_images[0].buffer.len();
        if !resized_images.iter().all(|r| r.buffer.len() == first_len) {
            anyhow::bail!(
                "Resized images have different sizes; batch processing requires uniform size"
            );
        }

        let buffers: Vec<Vec<u8>> = resized_images.iter().map(|r| r.buffer.clone()).collect();
        let original_sizes: Vec<(u32, u32)> =
            resized_images.iter().map(|r| r.original_size).collect();
        let scale_factors: Vec<Vec<f32>> = resized_images
            .iter()
            .map(|r| r.scale_factors.clone())
            .collect();

        // Store in Processor
        self.image_sizes = original_sizes.clone();
        self.scale_factors = scale_factors
            .iter()
            .map(|sf| [sf[0], sf[1]])
            .collect();

        let shape = (self.image_height as usize, self.image_width as usize, 3);
        let arrays: Vec<_> = buffers
            .iter()
            .map(|buffer| {
                Array::from_shape_vec(shape, buffer.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to create array from buffer: {}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut x = X(ndarray::stack(
            Axis(0),
            &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?
        .mapv(|x| x as f32)
        .into_dyn());
        if self.do_normalize {
            x = x.normalize(0., 255.)?;
        }
        if !self.image_std.is_empty() && !self.image_mean.is_empty() {
            x = x.standardize(&self.image_mean, &self.image_std, 3)?;
        }
        if self.nchw {
            x = x.nhwc2nchw()?;
        }
        if self.unsigned {
            x = x.unsigned();
        }
        eprintln!("Processor::process_images output shape: {:?}", x.0.shape());
        Ok(ProcessedImages {
            x,
            original_sizes,
            scale_factors,
        })
    }

    /// Processes grayscale images by converting them first.
    pub fn process_grayscale_images(&mut self, xs: &[DynamicImage]) -> Result<ProcessedImages> {
        let gray_images: Vec<DynamicImage> = xs
            .iter()
            .map(|img| self.rgb_to_grayscale(img).unwrap_or_else(|_| img.clone()))
            .collect();
        self.process_images(&gray_images)
    }

    /// Encodes text with proper error handling.
    pub fn encode_text(&self, x: &str, skip_special_tokens: bool) -> Result<Encoding> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer specified"))
            .and_then(|tk| tk.encode(x, skip_special_tokens))
    }

    pub fn encode_texts(&self, xs: &[&str], skip_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer specified"))
            .and_then(|tk| {
                tk.encode_batch(
                    xs.iter().map(|s| s.to_string()).collect(),
                    skip_special_tokens,
                )
            })
    }

    pub fn encode_text_ids(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<f32>> {
        let ids: Vec<f32> = if x.is_empty() {
            vec![0.0]
        } else {
            self.encode_text(x, skip_special_tokens)?
                .get_ids()
                .iter()
                .map(|x| *x as f32)
                .collect()
        };
        Ok(ids)
    }

    pub fn encode_texts_ids(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let ids: Vec<Vec<f32>> = if xs.is_empty() {
            vec![vec![0.0]]
        } else {
            self.encode_texts(xs, skip_special_tokens)?
                .into_iter()
                .map(|encoding| encoding.get_ids().iter().map(|x| *x as f32).collect())
                .collect()
        };
        Ok(ids)
    }

    pub fn encode_text_tokens(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<Text>> {
        Ok(self
            .encode_text(x, skip_special_tokens)?
            .get_tokens()
            .iter()
            .map(|s| Text::from(s.as_str()))
            .collect())
    }

    pub fn encode_texts_tokens(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<Vec<Text>>> {
        Ok(self
            .encode_texts(xs, skip_special_tokens)?
            .into_iter()
            .map(|encoding| {
                encoding
                    .get_tokens()
                    .iter()
                    .map(|s| Text::from(s.as_str()))
                    .collect()
            })
            .collect())
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<Text> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer specified"))
            .and_then(|tk| tk.decode(ids, skip_special_tokens))
            .map(Text::from)
    }

    pub fn decode_tokens_batch(
        &self,
        ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<Text>> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer specified"))
            .and_then(|tk| {
                tk.decode_batch(
                    &ids.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
                    skip_special_tokens,
                )
            })
            .map(|strings| strings.into_iter().map(Text::from).collect())
    }

    pub fn par_generate(
        &self,
        logits: &X,
        token_ids: &mut [Vec<u32>],
        eos_token_id: u32,
    ) -> Result<(bool, Vec<f32>)> {
        let batch = token_ids.len();
        let mut finished = vec![false; batch];
        let mut last_tokens = vec![0.0; batch];
        for (i, logit) in logits.axis_iter(Axis(0)).enumerate() {
            if !finished[i] {
                let token_id = self
                    .logits_sampler
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No `LogitsSampler` specified"))?
                    .decode(
                        &logit
                            .slice(s![-1, ..])
                            .into_owned()
                            .into_raw_vec_and_offset()
                            .0,
                    )?;
                if token_id == eos_token_id {
                    finished[i] = true;
                } else {
                    token_ids[i].push(token_id);
                }
                last_tokens[i] = token_id as f32;
            }
        }
        Ok((finished.iter().all(|&x| x), last_tokens))
    }

    /// Decodes bounding box logits into a list of Bbox instances.
    fn decode_bboxes(&self, bbox_logits: &X, confidence_threshold: f32) -> Result<Vec<Bbox>> {
        let shape = bbox_logits.0.shape();
        if shape.len() < 3 || shape[2] < 5 {
            anyhow::bail!("Invalid bbox_logits shape: {:?}", shape);
        }
        let num_boxes = shape[1];
        let _num_classes = shape[2] - 5; // [x, y, w, h, conf, class probs...]
        let mut bboxes = Vec::new();

        // Process each box in the batch (assuming batch size 1 for simplicity)
        for i in 0..num_boxes {
            let box_data = bbox_logits.0.slice(s![0, i, ..]).to_vec();
            let confidence = box_data[4];
            if confidence >= confidence_threshold {
                let x = box_data[0];
                let y = box_data[1];
                let w = box_data[2];
                let h = box_data[3];
                // Find max class probability and corresponding class ID
                let class_probs = &box_data[5..];
                let (class_id, max_prob) = class_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, &p)| (i as isize, p))
                    .unwrap_or((0, 0.0));
                let name = if class_id < self.vocab.len() as isize {
                    Some(self.vocab[class_id as usize].clone())
                } else {
                    None
                };
                let bbox = Bbox {
                    x,
                    y,
                    w,
                    h,
                    id: class_id,
                    id_born: -1,
                    confidence: confidence * max_prob,
                    name,
                };
                bboxes.push(bbox);
            }
        }
        Ok(bboxes)
    }

    /// Generates tokens and bounding boxes, returning a Ys instance.
    pub fn generate_to_ys(
        &self,
        token_logits: &X,
        token_ids: &mut [Vec<u32>],
        eos_token_id: u32,
        skip_special_tokens: bool,
        bbox_logits: Option<&X>,
        confidence_threshold: Option<f32>,
    ) -> Result<Ys> {
        let (_finished, _last_tokens) = self.par_generate(token_logits, token_ids, eos_token_id)?;
        let texts: Vec<Text> = token_ids
            .iter()
            .map(|ids| self.decode_tokens(ids, skip_special_tokens))
            .collect::<Result<Vec<_>>>()?;

        let bboxes = if let Some(bbox_logits) = bbox_logits {
            let threshold = confidence_threshold.unwrap_or(0.5);
            Some(self.decode_bboxes(bbox_logits, threshold)?)
        } else {
            None
        };

        let y = Y {
            texts: Some(texts),
            bboxes,
            ..Default::default()
        };
        Ok(Ys(vec![y]))
    }

    pub fn build_resizer_filter(ty: &str) -> Result<(Resizer, ResizeOptions)> {
        let ty = match ty.to_lowercase().as_str() {
            "box" => FilterType::Box,
            "bilinear" => FilterType::Bilinear,
            "hamming" => FilterType::Hamming,
            "catmullrom" => FilterType::CatmullRom,
            "mitchell" => FilterType::Mitchell,
            "gaussian" => FilterType::Gaussian,
            "lanczos3" => FilterType::Lanczos3,
            x => anyhow::bail!("Unsupported resizer's filter type: {}", x),
        };
        Ok((
            Resizer::new(),
            ResizeOptions::new().resize_alg(ResizeAlg::Convolution(ty)),
        ))
    }

    /// Resizes an image and returns the buffer with metadata.
    pub fn resize(&self, x: &DynamicImage) -> Result<ResizedImage> {
        let max_width = self
            .config
            .as_ref()
            .and_then(|json| json.get("max_width").and_then(|v| v.as_u64()))
            .unwrap_or(self.image_width as u64) as u32;
        let max_height = self
            .config
            .as_ref()
            .and_then(|json| json.get("max_height").and_then(|v| v.as_u64()))
            .unwrap_or(self.image_height as u64) as u32;
        let target_width = self.image_width.min(max_width);
        let target_height = self.image_height.min(max_height);

        if target_width == 0 || target_height == 0 {
            anyhow::bail!(
                "Invalid target dimensions: width={}, height={}",
                target_width,
                target_height
            );
        }

        let rgb_img = self.ensure_rgb(x.clone());
        let (w0, h0) = rgb_img.dimensions();
        let original_size = (w0, h0);

        let effective_mode = match &self.resize_mode {
            ResizeMode::FitExact => ResizeMode::FitExact,
            ResizeMode::Letterbox => ResizeMode::Letterbox,
            ResizeMode::FitAdaptive => ResizeMode::FitAdaptive,
            #[allow(deprecated)]
            ResizeMode::FitWidth | ResizeMode::FitHeight => ResizeMode::FitAdaptive,
        };

        let (buffer, scale_factors) = match effective_mode {
            ResizeMode::FitExact => {
                let mut dst = Image::new(target_width, target_height, PixelType::U8x3);
                let (mut resizer, options) = Self::build_resizer_filter(self.resize_filter)?;
                resizer.resize(&rgb_img, &mut dst, &options)?;
                let scale_factors = vec![
                    target_width as f32 / w0 as f32,
                    target_height as f32 / h0 as f32,
                ];
                (dst.into_vec(), scale_factors)
            }
            ResizeMode::Letterbox | ResizeMode::FitAdaptive => {
                let r = (target_width as f32 / w0 as f32).min(target_height as f32 / h0 as f32);
                let scale_factors = vec![r, r];
                let (w, h) = (
                    (w0 as f32 * r).round() as u32,
                    (h0 as f32 * r).round() as u32,
                );
                let mut dst = Image::from_vec_u8(
                    target_width,
                    target_height,
                    vec![self.padding_value; 3 * target_width as usize * target_height as usize],
                    PixelType::U8x3,
                )?;
                let (l, t) = if let ResizeMode::Letterbox = effective_mode {
                    if w == target_width {
                        (0, (target_height - h) / 2)
                    } else {
                        ((target_width - w) / 2, 0)
                    }
                } else {
                    (0, 0)
                };
                let mut dst_cropped = CroppedImageMut::new(&mut dst, l, t, w, h)?;
                let (mut resizer, options) = Self::build_resizer_filter(self.resize_filter)?;
                resizer.resize(&rgb_img, &mut dst_cropped, &options)?;
                (dst.into_vec(), scale_factors)
            }
            ResizeMode::FitWidth | ResizeMode::FitHeight => {
                unreachable!("Handled by effective_mode")
            }
        };

        Ok(ResizedImage {
            buffer,
            original_size,
            scale_factors,
        })
    }

    pub fn process_with_layout(
        &mut self,
        images: &[DynamicImage],
        polygons: &[Vec<Polygon>],
    ) -> Result<Vec<Vec<CroppedImage>>> {
        let mut cropped_images = Vec::new();
        for (image, polys) in images.iter().zip(polygons.iter()) {
            let mut crops = Vec::new();
            for poly in polys {
                if let Some(bbox) = poly.bbox() {
                    let x_min = bbox.xmin() as u32;
                    let y_min = bbox.ymin() as u32;
                    let width = (bbox.xmax() - bbox.xmin()) as u32;
                    let height = (bbox.ymax() - bbox.ymin()) as u32;
                    if width > 0
                        && height > 0
                        && x_min + width <= image.width()
                        && y_min + height <= image.height()
                    {
                        let crop = DynamicImage::ImageRgba8(
                            image.view(x_min, y_min, width, height).to_image(),
                        );
                        let rgb_crop = self.ensure_rgb(crop);
                        let resized = self.resize(&rgb_crop)?;
                        let bbox =
                            Bbox::from((x_min as f32, y_min as f32, width as f32, height as f32));
                        crops.push(CroppedImage {
                            image: DynamicImage::ImageRgb8(
                                RgbImage::from_raw(
                                    self.image_width,
                                    self.image_height,
                                    resized.buffer,
                                )
                                .ok_or_else(|| {
                                    anyhow::anyhow!("Failed to create RgbImage from raw data")
                                })?,
                            ),
                            bbox,
                        });
                    }
                }
            }
            cropped_images.push(crops);
        }
        Ok(cropped_images)
    }
}

#[derive(Debug, Clone)]
pub enum ResizeMode {
    FitExact,
    #[deprecated(note = "FitWidth produces variable sizes; use FitAdaptive instead")]
    FitWidth,
    #[deprecated(note = "FitHeight produces variable sizes; use FitAdaptive instead")]
    FitHeight,
    FitAdaptive,
    Letterbox,
}

impl Processor {
    pub fn generation_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_length: 64,
            conf_threshold: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub conf_threshold: f32,
}