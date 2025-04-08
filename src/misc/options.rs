use aksr::Builder;
use anyhow::Result;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use std::path::Path;
use crate::{ DType, Device, Engine, Hub, Iiix, Kind, LogitsSampler, MinOptMax, Processor, ResizeMode, Scale,
    Task, Version,
};

#[derive(Builder, Debug, Clone)]
pub struct Options {
    // Model configs
    pub model_file: String,
    pub model_name: &'static str,
    pub model_device: Device,
    pub model_dtype: DType,
    pub model_version: Option<Version>,
    pub model_task: Option<Task>,
    pub model_scale: Option<Scale>,
    pub model_kind: Option<Kind>,
    pub model_iiixs: Vec<Iiix>,
    pub model_spec: String,
    pub model_num_dry_run: usize,
    pub trt_fp16: bool,
    pub profile: bool,
    
    // Processor configs
    #[args(setter = false)]
    pub image_width: u32,
    #[args(setter = false)]
    pub image_height: u32,
    pub resize_mode: ResizeMode,
    pub resize_filter: &'static str,
    pub padding_value: u8,
    pub letterbox_center: bool,
    pub normalize: bool,
    pub image_std: Vec<f32>,
    pub image_mean: Vec<f32>,
    pub nchw: bool,
    pub unsigned: bool,

    // Names
    pub class_names: Option<Vec<String>>,
    pub class_names_2: Option<Vec<String>>,
    pub class_names_3: Option<Vec<String>>,
    pub keypoint_names: Option<Vec<String>>,
    pub keypoint_names_2: Option<Vec<String>>,
    pub keypoint_names_3: Option<Vec<String>>,
    pub text_names: Option<Vec<String>>,
    pub text_names_2: Option<Vec<String>>,
    pub text_names_3: Option<Vec<String>>,
    pub category_names: Option<Vec<String>>,
    pub category_names_2: Option<Vec<String>>,
    pub category_names_3: Option<Vec<String>>,

    // Confs
    pub class_confs: Vec<f32>,
    pub class_confs_2: Vec<f32>,
    pub class_confs_3: Vec<f32>,
    pub keypoint_confs: Vec<f32>,
    pub keypoint_confs_2: Vec<f32>,
    pub keypoint_confs_3: Vec<f32>,
    pub text_confs: Vec<f32>,
    pub text_confs_2: Vec<f32>,
    pub text_confs_3: Vec<f32>,

    // Detection
    pub num_classes: Option<usize>,
    pub num_keypoints: Option<usize>,
    pub num_masks: Option<usize>,
    pub iou: Option<f32>,
    pub iou_2: Option<f32>,
    pub iou_3: Option<f32>,
    pub apply_nms: Option<bool>,
    pub find_contours: bool,
    pub yolo_preds_format: Option<YOLOPredsFormat>,
    pub classes_excluded: Vec<usize>,
    pub classes_retained: Vec<usize>,
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,

    // Language/OCR configs
    pub model_max_length: Option<u64>,
    pub tokenizer_file: Option<String>,
    pub config_file: Option<String>,
    pub special_tokens_map_file: Option<String>,
    pub tokenizer_config_file: Option<String>,
    pub generation_config_file: Option<String>,
    pub vocab_file: Option<String>,
    pub vocab_txt: Option<String>,
    pub temperature: f32,
    pub topp: f32,

    // DB-specific
    pub unclip_ratio: Option<f32>,
    pub binary_thresh: Option<f32>,

    // SAM-specific
    pub low_res_mask: Option<bool>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model_file: String::new(),
            model_name: "",
            model_device: Device::Cpu(0),
            model_dtype: DType::Auto,
            model_version: None,
            model_task: None,
            model_scale: None,
            model_kind: None,
            model_iiixs: Vec::new(),
            model_spec: String::new(),
            model_num_dry_run: 3,
            trt_fp16: true,
            profile: false,
            image_width: 640,
            image_height: 640,
            resize_mode: ResizeMode::FitExact,
            resize_filter: "Bilinear",
            padding_value: 114,
            letterbox_center: false,
            normalize: true,
            image_mean: vec![],
            image_std: vec![],
            nchw: true,
            unsigned: false,
            class_names: None,
            class_names_2: None,
            class_names_3: None,
            keypoint_names: None,
            keypoint_names_2: None,
            keypoint_names_3: None,
            text_names: None,
            text_names_2: None,
            text_names_3: None,
            category_names: None,
            category_names_2: None,
            category_names_3: None,
            class_confs: vec![0.3],
            class_confs_2: vec![0.3],
            class_confs_3: vec![0.3],
            keypoint_confs: vec![0.3],
            keypoint_confs_2: vec![0.5],
            keypoint_confs_3: vec![0.5],
            text_confs: vec![0.4],
            text_confs_2: vec![0.4],
            text_confs_3: vec![0.4],
            num_classes: None,
            num_keypoints: None,
            num_masks: None,
            iou: None,
            iou_2: None,
            iou_3: None,
            apply_nms: None,
            find_contours: false,
            yolo_preds_format: None,
            classes_excluded: vec![],
            classes_retained: vec![],
            min_width: None,
            min_height: None,
            model_max_length: None,
            tokenizer_file: None,
            config_file: None,
            special_tokens_map_file: None,
            tokenizer_config_file: None,
            generation_config_file: None,
            vocab_file: None,
            vocab_txt: None,
            temperature: 1.0,
            topp: 0.0,
            unclip_ratio: Some(1.5),
            binary_thresh: Some(0.2),
            low_res_mask: None,
        }
    }
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model_path(mut self, path: &str) -> Result<Self> {
        self.model_file = path.to_string();
        Ok(self)
    }

    pub fn with_generation_config<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        self.generation_config_file = Some(path.as_ref().to_string_lossy().into_owned());
        Ok(self)
    }

    pub fn with_special_tokens_map<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        self.special_tokens_map_file = Some(path.as_ref().to_string_lossy().into_owned());
        Ok(self)
    }

    pub fn to_engine(&self) -> Result<Engine> {
        Engine {
            file: self.model_file.clone(),
            spec: self.model_spec.clone(),
            device: self.model_device,
            trt_fp16: self.trt_fp16,
            iiixs: self.model_iiixs.clone(),
            num_dry_run: self.model_num_dry_run,
            ..Default::default()
        }
        .build()
    }

    pub fn to_processor(&self) -> Result<Processor> {
    let logits_sampler = LogitsSampler::new()
        .with_temperature(self.temperature)
        .with_topp(self.topp);

    let tokenizer = match self.model_kind {
        Some(Kind::Language) | Some(Kind::VisionLanguage) => self.try_build_tokenizer()?,
        _ => return Err(anyhow::anyhow!("Tokenizer required for language models")),
    };

    let vocab: Vec<String> = match &self.vocab_txt {
        Some(x) => {
            let file = if !std::path::PathBuf::from(&x).exists() {
                Hub::default().try_fetch(&format!("{}/{}", self.model_name, x))?
            } else {
                x.to_string()
            };
            std::fs::read_to_string(file)?
                .lines()
                .map(|line| line.to_string())
                .collect()
        }
        None => vec![],
    };

    // Convert Vec<String> to Vec<&str>
    let vocab_refs: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();

    Ok(Processor::default()
        .with_image_width(self.image_width)
        .with_image_height(self.image_height)
        .with_image0s_size(&[]) // Empty slice
        .with_scale_factors_hw(&[]) // Empty slice
        .with_resize_mode(self.resize_mode.clone())
        .with_resize_filter(self.resize_filter)
        .with_padding_value(self.padding_value)
        .with_do_normalize(self.normalize)
        .with_image_mean(&self.image_mean) // Borrow the Vec
        .with_image_std(&self.image_std) // Borrow the Vec
        .with_nchw(self.nchw)
        .with_tokenizer(tokenizer) // Unwrapped tokenizer
        .with_vocab(&vocab_refs) // Convert to slice of &str
        .with_unsigned(self.unsigned)
        .with_logits_sampler(logits_sampler) // Unwrapped sampler
        .with_options(self.clone()))
}

    pub fn commit(mut self) -> Result<Self> {
        if std::path::PathBuf::from(&self.model_file).exists() {
            self.model_spec = format!(
                "{}/{}",
                self.model_name,
                crate::try_fetch_stem(&self.model_file)?
            );
        } else {
            if self.model_file.is_empty() && self.model_name.is_empty() {
                anyhow::bail!("Neither `model_name` nor `model_file` were specified. Failed to fetch model from remote.")
            }

            match Hub::is_valid_github_release_url(&self.model_file) {
                Some((owner, repo, tag, _file_name)) => {
                    let stem = crate::try_fetch_stem(&self.model_file)?;
                    self.model_spec =
                        format!("{}/{}-{}-{}-{}", self.model_name, owner, repo, tag, stem);
                    self.model_file = Hub::default().try_fetch(&self.model_file)?;
                }
                None => {
                    if self.model_file.is_empty() && self.model_name == "yolo" {
                        let mut y = String::new();
                        if let Some(x) = self.model_version {
                            y.push_str(&x.to_string());
                        }
                        if let Some(x) = self.model_scale {
                            y.push_str(&format!("-{}", x));
                        }
                        if let Some(ref x) = self.model_task {
                            y.push_str(&format!("-{}", x.yolo_str()));
                        }
                        y.push_str(".onnx");
                        self.model_file = y;
                    }

                    match self.model_dtype {
                        d @ (DType::Auto | DType::Fp32) => {
                            if self.model_file.is_empty() {
                                self.model_file = format!("{}.onnx", d);
                            }
                        }
                        dtype => {
                            if self.model_file.is_empty() {
                                self.model_file = format!("{}.onnx", dtype);
                            } else {
                                let pos = self.model_file.len() - 5;
                                let suffix = self.model_file.split_off(pos);
                                self.model_file =
                                    format!("{}-{}{}", self.model_file, dtype, suffix);
                            }
                        }
                    }

                    let stem = crate::try_fetch_stem(&self.model_file)?;
                    self.model_spec = format!("{}/{}", self.model_name, stem);
                    self.model_file = Hub::default()
                        .try_fetch(&format!("{}/{}", self.model_name, self.model_file))?;
                }
            }
        }

        Ok(self)
    }

    pub fn with_batch_size(mut self, x: usize) -> Self {
        self.model_iiixs.push(Iiix::from((0, 0, x.into())));
        self
    }

    pub fn with_image_height(mut self, x: u32) -> Self {
        self.image_height = x;
        self.model_iiixs.push(Iiix::from((0, 2, x.into())));
        self
    }

    pub fn with_image_width(mut self, x: u32) -> Self {
        self.image_width = x;
        self.model_iiixs.push(Iiix::from((0, 3, x.into())));
        self
    }

    pub fn with_model_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.model_iiixs.push(Iiix::from((i, ii, x)));
        self
    }

    pub fn exclude_classes(mut self, xs: &[usize]) -> Self {
        self.classes_retained.clear();
        self.classes_excluded.extend_from_slice(xs);
        self
    }

    pub fn retain_classes(mut self, xs: &[usize]) -> Self {
        self.classes_excluded.clear();
        self.classes_retained.extend_from_slice(xs);
        self
    }

    pub fn try_build_tokenizer(&self) -> Result<Tokenizer> {
        let mut hub = Hub::default();
        let pad_id = match hub.try_fetch(
            self.tokenizer_config_file
                .as_ref()
                .unwrap_or(&format!("{}/config.json", self.model_name)),
        ) {
            Ok(x) => {
                let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(x)?)?;
                config["pad_token_id"].as_u64().unwrap_or(0) as u32
            }
            Err(_err) => 0u32,
        };

        let mut max_length = None;
        let mut pad_token = String::from("[PAD]");
        match hub.try_fetch(
            self.tokenizer_config_file
                .as_ref()
                .unwrap_or(&format!("{}/tokenizer_config.json", self.model_name)),
        ) {
            Err(_) => {}
            Ok(x) => {
                let tokenizer_config: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(x)?)?;
                max_length = tokenizer_config["model_max_length"].as_u64();
                pad_token = tokenizer_config["pad_token"]
                    .as_str()
                    .unwrap_or("[PAD]")
                    .to_string();
            }
        };

        let mut tokenizer: Tokenizer = Tokenizer::from_file(
            hub.try_fetch(
                self.tokenizer_file
                    .as_ref()
                    .unwrap_or(&format!("{}/tokenizer.json", self.model_name)),
            )?,
        )
        .map_err(|err| anyhow::anyhow!("Failed to build tokenizer: {err}"))?;

        let tokenizer = match self.model_max_length {
            Some(n) => {
                let n = match max_length {
                    None => n,
                    Some(x) => x.min(n),
                };
                tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::Fixed(n as _),
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .clone()
            }
            None => {
                let modified = match max_length {
                    Some(n) => tokenizer
                        .with_padding(Some(PaddingParams {
                            strategy: PaddingStrategy::BatchLongest,
                            pad_token,
                            pad_id,
                            ..Default::default()
                        }))
                        .with_truncation(Some(TruncationParams {
                            max_length: n as _,
                            ..Default::default()
                        }))
                        .map_err(|err| anyhow::anyhow!("Failed to set truncation: {err}"))?,
                    None => &mut tokenizer,
                };
                modified.clone()
            }
        };
        Ok(tokenizer.into())
    }

    pub fn nc(&self) -> Option<usize> {
        self.num_classes.or_else(|| self.class_names.as_ref().map(|v| v.len()))
    }

    pub fn nk(&self) -> Option<usize> {
        self.num_keypoints.or_else(|| self.keypoint_names.as_ref().map(|v| v.len()))
    }
}