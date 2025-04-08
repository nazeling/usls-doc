use anyhow::Result;
use fast_image_resize::{images::{CroppedImageMut, Image}, pixels::PixelType, FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{s, Array, Axis};
use rayon::prelude::*;
use std::sync::Mutex;
use tokenizers::{Encoding, Tokenizer};
use serde_json::Value;

use crate::{LogitsSampler, X, Polygon, Options};

#[derive(Debug, Clone)]
pub enum ResizeMode {
    FitExact,
    FitWidth,
    FitHeight,
    FitAdaptive,
    Letterbox,
}

#[derive(aksr::Builder, Debug, Clone)]
pub struct Processor {
    pub image_width: u32,
    pub image_height: u32,
    pub image0s_size: Vec<(u32, u32)>,
    pub scale_factors_hw: Vec<Vec<f32>>,
    pub resize_mode: ResizeMode,
    pub resize_filter: &'static str,
    pub padding_value: u8,
    pub do_normalize: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub nchw: bool,
    pub tokenizer: Option<Tokenizer>,
    pub vocab: Vec<String>,
    pub unsigned: bool,
    pub logits_sampler: Option<LogitsSampler>,
    options: Options,
}

impl Default for Processor {
    fn default() -> Self {
        Self {
            image0s_size: vec![],
            image_width: 0,
            image_height: 0,
            scale_factors_hw: vec![],
            resize_mode: ResizeMode::FitAdaptive,
            resize_filter: "Bilinear",
            padding_value: 114,
            do_normalize: true,
            image_mean: vec![],
            image_std: vec![],
            nchw: true,
            tokenizer: Default::default(),
            vocab: vec![],
            unsigned: false,
            logits_sampler: None,
            options: Options::default(),
        }
    }
}

impl Processor {
    pub fn reset_image0_status(&mut self) {
        self.scale_factors_hw.clear();
        self.image0s_size.clear();
    }

    pub fn get_options(&self) -> &Options {
        &self.options
    }

    pub fn generation_config(&self) -> Option<Value> {
        self.options
            .generation_config_file
            .as_ref()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
    }

    pub fn process_images(&mut self, xs: &[DynamicImage]) -> Result<X> {
        let (mut x, image0s_size, scale_factors_hw) = self.par_resize(xs)?;
        self.image0s_size = image0s_size;
        self.scale_factors_hw = scale_factors_hw;
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
        Ok(x)
    }

    pub fn encode_text(&self, x: &str, skip_special_tokens: bool) -> Result<Encoding> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .encode(x, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer encode error: {}", err))
    }

    pub fn encode_texts(&self, xs: &[&str], skip_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .encode_batch(xs.to_vec(), skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer encode_batch error: {}", err))
    }

    pub fn encode_text_ids(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<f32>> {
        let ids: Vec<f32> = if x.is_empty() {
            vec![0.0f32]
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
            vec![vec![0.0f32]]
        } else {
            self.encode_texts(xs, skip_special_tokens)?
                .into_iter()
                .map(|encoding| encoding.get_ids().iter().map(|x| *x as f32).collect())
                .collect()
        };
        Ok(ids)
    }

    pub fn encode_text_tokens(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<String>> {
        Ok(self
            .encode_text(x, skip_special_tokens)?
            .get_tokens()
            .to_vec())
    }

    pub fn encode_texts_tokens(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<Vec<String>>> {
        Ok(self
            .encode_texts(xs, skip_special_tokens)?
            .into_iter()
            .map(|encoding| encoding.get_tokens().to_vec())
            .collect())
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .decode(ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer decode error: {}", err))
    }

    pub fn decode_tokens_batch2(
        &self,
        ids: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .decode_batch(ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer decode_batch error: {}", err))
    }

    pub fn decode_tokens_batch(
        &self,
        ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .decode_batch(
                &ids.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
                skip_special_tokens,
            )
            .map_err(|err| anyhow::anyhow!("Tokenizer decode_batch error: {}", err))
    }

    pub fn par_generate(
        &self,
        logits: &X,
        token_ids: &mut [Vec<u32>],
        eos_token_id: u32,
    ) -> Result<(bool, Vec<f32>)> {
        let batch = token_ids.len();
        let mut finished = vec![false; batch];
        let mut last_tokens: Vec<f32> = vec![0.; batch];
        for (i, logit) in logits.axis_iter(Axis(0)).enumerate() {
            if !finished[i] {
                let token_id = self
                    .logits_sampler
                    .as_ref()
                    .expect("No `LogitsSampler` specified!")
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

    pub fn resize(&mut self, x: &DynamicImage) -> Result<Vec<u8>> {
        let max_width = self.generation_config()
            .and_then(|json| json.get("max_width").and_then(|v| v.as_u64()))
            .unwrap_or(self.image_width as u64) as u32;
        let max_height = self.generation_config()
            .and_then(|json| json.get("max_height").and_then(|v| v.as_u64()))
            .unwrap_or(self.image_height as u64) as u32;

        let target_width = self.image_width.min(max_width);
        let target_height = self.image_height.min(max_height);

        if target_width + target_height == 0 {
            anyhow::bail!(
                "Invalid target height: {} or width: {}.",
                target_height,
                target_width
            );
        }

        let buffer = match x.dimensions() {
            (w, h) if (w, h) == (target_height, target_width) => {
                self.image0s_size.push((h, w));
                self.scale_factors_hw.push(vec![1., 1.]);
                x.to_rgb8().into_raw()
            }
            (w0, h0) => {
                self.image0s_size.push((h0, w0));
                let (mut resizer, options) = Self::build_resizer_filter(self.resize_filter)?;

                if let ResizeMode::FitExact = self.resize_mode {
                    let mut dst = Image::new(target_width, target_height, PixelType::U8x3);
                    resizer.resize(x, &mut dst, &options)?;
                    self.scale_factors_hw.push(vec![
                        (target_height as f32 / h0 as f32),
                        (target_width as f32 / w0 as f32),
                    ]);
                    dst.into_vec()
                } else {
                    let (w, h) = match self.resize_mode {
                        ResizeMode::Letterbox | ResizeMode::FitAdaptive => {
                            let r = (target_width as f32 / w0 as f32)
                                .min(target_height as f32 / h0 as f32);
                            self.scale_factors_hw.push(vec![r, r]);
                            ((w0 as f32 * r).round() as u32, (h0 as f32 * r).round() as u32)
                        }
                        ResizeMode::FitHeight => {
                            let r = target_height as f32 / h0 as f32;
                            self.scale_factors_hw.push(vec![1.0, r]);
                            ((r * w0 as f32).round() as u32, target_height)
                        }
                        ResizeMode::FitWidth => {
                            let r = target_width as f32 / w0 as f32;
                            self.scale_factors_hw.push(vec![r, 1.0]);
                            (target_width, (r * h0 as f32).round() as u32)
                        }
                        _ => unreachable!(),
                    };
                    let mut dst = Image::from_vec_u8(
                        target_width,
                        target_height,
                        vec![self.padding_value; 3 * target_height as usize * target_width as usize],
                        PixelType::U8x3,
                    )?;
                    let (l, t) = if let ResizeMode::Letterbox = self.resize_mode {
                        if w == target_width {
                            (0, (target_height - h) / 2)
                        } else {
                            ((target_width - w) / 2, 0)
                        }
                    } else {
                        (0, 0)
                    };
                    let mut dst_cropped = CroppedImageMut::new(&mut dst, l, t, w, h)?;
                    resizer.resize(x, &mut dst_cropped, &options)?;
                    dst.into_vec()
                }
            }
        };
        Ok(buffer)
    }

    pub fn process_with_layout(
        &mut self,
        images: &[DynamicImage],
        polygons: &[Vec<Polygon>],
    ) -> Result<Vec<Vec<DynamicImage>>> {
        let mut cropped_images = Vec::new();
        for (image, polys) in images.iter().zip(polygons.iter()) {
            let mut crops = Vec::new();
            for poly in polys {
                if let Some(bbox) = poly.bbox() {
                    let x_min = bbox.xmin() as u32;
                    let y_min = bbox.ymin() as u32;
                    let width = (bbox.xmax() - bbox.xmin()) as u32;
                    let height = (bbox.ymax() - bbox.ymin()) as u32;
                    if width > 0 && height > 0 && x_min + width <= image.width() && y_min + height <= image.height() {
                        let crop = DynamicImage::ImageRgba8(image.view(x_min, y_min, width, height).to_image()).to_rgb8();
                        let processed_crop = self.resize(&DynamicImage::ImageRgb8(crop))?;
                        crops.push(DynamicImage::ImageRgb8(
                            RgbImage::from_raw(self.image_width, self.image_height, processed_crop)
                                .ok_or_else(|| anyhow::anyhow!("Failed to create RgbImage from raw data"))?
                        ));
                    }
                }
            }
            cropped_images.push(crops);
        }
        Ok(cropped_images)
    }

    #[allow(clippy::type_complexity)]
    pub fn resize2(&self, x: &DynamicImage) -> Result<(X, (u32, u32), Vec<f32>)> {
        if self.image_width + self.image_height == 0 {
            anyhow::bail!(
                "Invalid target height: {} or width: {}.",
                self.image_height,
                self.image_width
            );
        }
        let image0s_size: (u32, u32);
        let scale_factors_hw: Vec<f32>;
        let buffer = match x.dimensions() {
            (w, h) if (w, h) == (self.image_height, self.image_width) => {
                image0s_size = (h, w);
                scale_factors_hw = vec![1., 1.];
                x.to_rgb8().into_raw()
            }
            (w0, h0) => {
                image0s_size = (h0, w0);
                let (mut resizer, options) = Self::build_resizer_filter(self.resize_filter)?;
                if let ResizeMode::FitExact = self.resize_mode {
                    let mut dst = Image::new(self.image_width, self.image_height, PixelType::U8x3);
                    resizer.resize(x, &mut dst, &options)?;
                    scale_factors_hw = vec![
                        (self.image_height as f32 / h0 as f32),
                        (self.image_width as f32 / w0 as f32),
                    ];
                    dst.into_vec()
                } else {
                    let (w, h) = match self.resize_mode {
                        ResizeMode::Letterbox | ResizeMode::FitAdaptive => {
                            let r = (self.image_width as f32 / w0 as f32)
                                .min(self.image_height as f32 / h0 as f32);
                            scale_factors_hw = vec![r, r];
                            ((w0 as f32 * r).round() as u32, (h0 as f32 * r).round() as u32)
                        }
                        ResizeMode::FitHeight => {
                            let r = self.image_height as f32 / h0 as f32;
                            scale_factors_hw = vec![1.0, r];
                            ((r * w0 as f32).round() as u32, self.image_height)
                        }
                        ResizeMode::FitWidth => {
                            let r = self.image_width as f32 / w0 as f32;
                            scale_factors_hw = vec![r, 1.0];
                            (self.image_width, (r * h0 as f32).round() as u32)
                        }
                        _ => unreachable!(),
                    };
                    let mut dst = Image::from_vec_u8(
                        self.image_width,
                        self.image_height,
                        vec![self.padding_value; 3 * self.image_height as usize * self.image_width as usize],
                        PixelType::U8x3,
                    )?;
                    let (l, t) = if let ResizeMode::Letterbox = self.resize_mode {
                        if w == self.image_width {
                            (0, (self.image_height - h) / 2)
                        } else {
                            ((self.image_width - w) / 2, 0)
                        }
                    } else {
                        (0, 0)
                    };
                    let mut dst_cropped = CroppedImageMut::new(&mut dst, l, t, w, h)?;
                    resizer.resize(x, &mut dst_cropped, &options)?;
                    dst.into_vec()
                }
            }
        };
        let y = Array::from_shape_vec(
            (self.image_height as usize, self.image_width as usize, 3),
            buffer.clone(),
        )?
        .mapv(|x| x as f32)
        .into_dyn();
        Ok((y.into(), image0s_size, scale_factors_hw))
    }

    #[allow(clippy::type_complexity)]
    pub fn par_resize(&mut self, xs: &[DynamicImage]) -> Result<(X, Vec<(u32, u32)>, Vec<Vec<f32>>)> {
        match xs.len() {
            0 => anyhow::bail!("Found no input images."),
            1 => {
                let (y, image0_size, scale_factors) = self.resize2(&xs[0])?;
                Ok((y.insert_axis(0)?, vec![image0_size], vec![scale_factors]))
            }
            _ => {
                let ys = Mutex::new(
                    Array::zeros((
                        xs.len(),
                        self.image_height as usize,
                        self.image_width as usize,
                        3,
                    ))
                    .into_dyn(),
                );
                let results: Result<Vec<((u32, u32), Vec<f32>)>> = xs
                    .par_iter()
                    .enumerate()
                    .map(|(idx, x)| {
                        let (y, image0_size, scale_factors) = self.resize2(x)?;
                        {
                            let mut ys_guard = ys
                                .lock()
                                .map_err(|e| anyhow::anyhow!("Mutex lock error: {e}"))?;
                            ys_guard.slice_mut(s![idx, .., .., ..]).assign(&y);
                        }
                        Ok((image0_size, scale_factors))
                    })
                    .collect();
                let (image0s_size, scale_factors_hw) = results?.into_iter().unzip();
                let ys_inner = ys
                    .into_inner()
                    .map_err(|e| anyhow::anyhow!("Mutex into_inner error: {e}"))?;
                Ok((ys_inner.into(), image0s_size, scale_factors_hw))
            }
        }
    }

    pub fn get_original_sizes(&self) -> &[(u32, u32)] {
        &self.image0s_size
    }

    pub fn get_scale_factors(&self) -> &[Vec<f32>] {
        &self.scale_factors_hw
    }
}