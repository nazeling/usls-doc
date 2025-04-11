use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{elapsed, DynConf, Engine, Options, Processor, Ts, Xs, Ys, Y};

#[derive(Builder, Debug)]
pub struct SVTR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs: DynConf,
    spec: String,
    ts: Ts,
    pub processor: Processor,
}

impl SVTR {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, height, width, ts) = (
            options.model_iiixs.get(0).map_or(1, |i| i.x.opt()),
            options.model_iiixs.get(2).map_or(48, |i| i.x.opt()),
            options.model_iiixs.get(3).map_or(960, |i| i.x.opt()),
            engine.ts.clone(),
        );
        let spec = options.model_spec().to_string();
        let confs = DynConf::new(options.class_confs(), 1);
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);
        if processor.vocab().is_empty() {
            anyhow::bail!("No vocab file found")
        }
        log::info!("Vocab size: {}", processor.vocab().len());

        Ok(Self {
            engine,
            height,
            width,
            batch,
            confs,
            processor,
            spec,
            ts,
        })
    }

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        // ProcessedImages struct has .x not .images
        Ok(self.processor.process_images(xs)?.to_xs(None))
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[DynamicImage]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });
        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }

    pub fn postprocess(&self, xs: Xs) -> Result<Ys> {
        // Use values directly from the ProcessedImages' config or processor fields
        let config = self.processor.generation_config();
        let max_length = config.max_length;
        let conf_threshold = config.conf_threshold;

        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|preds| {
                let mut preds: Vec<_> = preds
                    .axis_iter(Axis(0))
                    .filter_map(|x| x.into_iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)))
                    .collect();
                preds.truncate(max_length);
                let text: String = preds
                    .into_iter()
                    .filter(|(id, &conf)| *id != 0 && conf >= conf_threshold)
                    .map(|(id, _)| self.processor.vocab()[id].clone())
                    .collect();
                Y::default().with_texts(&[text.into()])
            })
            .collect();
        Ok(Ys(ys)) // Wrap with Ys
    }
}
