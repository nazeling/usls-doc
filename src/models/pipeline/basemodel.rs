use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;

use crate::{
    elapsed, DType, Device, Engine, Kind, Options, Processor, Scale, Task, Ts, Version, Xs, X,
};

#[derive(Debug, Builder)]
pub struct BaseModelVisual {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    ts: Ts,
    spec: String,
    name: &'static str,
    device: Device,
    dtype: DType,
    task: Option<Task>,
    scale: Option<Scale>,
    kind: Option<Kind>,
    version: Option<Version>,
    #[args(setter = false, default)]
    force_rgb_flag: bool, // Flag to indicate whether to force images to RGB.
}

impl BaseModelVisual {
    pub fn summary(&self) {
        self.ts.summary();
    }

    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let err_msg = "You need to specify the image height and image width for visual model.";
        let (batch, height, width, ts, spec) = (
            engine.batch().opt(),
            engine.try_height().expect(err_msg).opt(),
            engine.try_width().expect(err_msg).opt(),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(options.image_width)
            .with_image_height(options.image_height);

        let device = options.model_device;
        let task = options.model_task;
        let scale = options.model_scale;
        let dtype = options.model_dtype;
        let kind = options.model_kind;
        let name = options.model_name;
        let version = options.model_version;

        Ok(Self {
            engine,
            height,
            width,
            batch,
            processor,
            ts,
            spec,
            dtype,
            task,
            scale,
            kind,
            device,
            version,
            name,
            force_rgb_flag: false,
        })
    }

    pub fn force_rgb(&mut self, enable: bool) {
        self.force_rgb_flag = enable; // No-op, Processor handles RGB automatically
    }

    pub fn processor_mut(&mut self) -> &mut Processor {
        &mut self.processor
    }

    pub fn encode_with_processed(&mut self, x: &X) -> Result<X> {
        let mut input = x.clone();
        if self.processor.nchw() && input.0.shape()[1] != 3 {
            input = input.nhwc2nchw()?;
        }
        println!("CUDA input shape: {:?}", input.0.shape()); // Updated label
        let xs = Xs::from(vec![input]);
        let outputs = self.engine.run(xs)?;
        Ok(outputs[0].clone())
    }

    pub fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;
        self.batch = xs.len();
        println!("Preprocessed shape: {:?}", x.x.0.shape()); // changed from x.images[0]
        Ok(Xs::from(vec![x.x])) // changed from xs::Xs::from(x.images)
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn encode(&mut self, xs: &[DynamicImage]) -> Result<X> {
        let xs = elapsed!("visual-preprocess", self.ts, { self.preprocess(xs)? });
        let xs = elapsed!("visual-inference", self.ts, { self.inference(xs)? });
        Ok(xs[0].to_owned())
    }
}

#[derive(Debug, Builder)]
pub struct BaseModelTextual {
    engine: Engine,
    batch: usize,
    processor: Processor,
    ts: Ts,
    spec: String,
    name: &'static str,
    device: Device,
    dtype: DType,
    task: Option<Task>,
    scale: Option<Scale>,
    kind: Option<Kind>,
    version: Option<Version>,
}

impl BaseModelTextual {
    pub fn summary(&self) {
        self.ts.summary();
    }

    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, ts, spec) = (
            engine.batch().opt(),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );
        let processor = options.to_processor()?;
        let device = options.model_device;
        let task = options.model_task;
        let scale = options.model_scale;
        let dtype = options.model_dtype;
        let kind = options.model_kind;
        let name = options.model_name;
        let version = options.model_version;

        Ok(Self {
            engine,
            batch,
            processor,
            ts,
            spec,
            dtype,
            task,
            scale,
            kind,
            device,
            version,
            name,
        })
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }
}
