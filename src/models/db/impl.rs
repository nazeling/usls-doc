use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{elapsed, Bbox, DynConf, Engine, Mbr, Ops, Options, Polygon, Processor, Ts, Xs, Ys, Y};

#[derive(Debug, Builder)]
pub struct DB {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs: DynConf,
    unclip_ratio: f32,
    binary_thresh: f32,
    min_width: f32,
    min_height: f32,
    spec: String,
    ts: Ts,
    processor: Processor,
}

impl DB {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, height, width, ts, spec) = (
            engine.batch().opt(),
            engine.try_height().map_or(960, |h| h.opt()),
            engine.try_width().map_or(960, |w| w.opt()),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );
        let unclip_ratio = options.unclip_ratio.unwrap_or(1.5);
        let binary_thresh = options.binary_thresh.unwrap_or(0.2);
        let confs = DynConf::new(options.class_confs(), 1);
        let processor = options
            .to_processor()?
            .with_image_width(width as u32)
            .with_image_height(height as u32);
        Ok(Self {
            engine,
            height,
            width,
            batch,
            unclip_ratio,
            binary_thresh,
            confs,
            processor,
            spec,
            ts,
            min_height: options.min_height.unwrap_or(0.0),
            min_width: options.min_width.unwrap_or(0.0),
        })
    }

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
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

    pub fn postprocess(&mut self, xs: Xs) -> Result<Ys> {
    let ys: Vec<Y> = xs[0]
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .filter_map(|(idx, luma)| {
            let (image_height, image_width) = if !self.processor.image_sizes().is_empty() {
                self.processor.image_sizes()[idx]
            } else {
                println!("Warning: image_sizes empty, using default 960x960");
                (127, 1192) // Use actual image size (1192x127)
            };
            let ratio = if !self.processor.scale_factors().is_empty() {
                self.processor.scale_factors()[idx][0]
            } else {
                // Calculate scale: target / original
                let scale_h = self.height as f32 / image_height as f32; // 960 / 127
                let scale_w = self.width as f32 / image_width as f32;   // 960 / 1192
                let ratio = scale_h.min(scale_w); // Use the smaller scale to fit
                println!("Warning: scale_factors empty, calculated ratio: {}", ratio);
                ratio
            };
            let v = luma
                .as_slice()
                .unwrap()
                .par_iter()
                .map(|x| {
                    if x <= &self.binary_thresh {
                        0u8
                    } else {
                        (*x * 255.0) as u8
                    }
                })
                .collect::<Vec<_>>();

            let luma = Ops::resize_luma8_u8(
                &v,
                self.width as _,
                self.height as _,
                image_width as _,
                image_height as _,
                true,
                "Bilinear",
            )
            .ok()?;
            let mask_im: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                image::ImageBuffer::from_raw(image_width as _, image_height as _, luma)?;

            let contours: Vec<imageproc::contours::Contour<i32>> =
                imageproc::contours::find_contours_with_threshold(&mask_im, 1);

            let (y_polygons, y_bbox, y_mbrs): (Vec<Polygon>, Vec<Bbox>, Vec<Mbr>) = contours
                .par_iter()
                .filter_map(|contour| {
                    if contour.border_type == imageproc::contours::BorderType::Hole
                        && contour.points.len() <= 2
                    {
                        return None;
                    }

                    let polygon = Polygon::default()
                        .with_points_imageproc(&contour.points)
                        .with_id(0);
                    let delta =
                        polygon.area() * ratio.round() as f64 * self.unclip_ratio as f64
                            / polygon.perimeter();

                    let polygon = polygon
                        .unclip(delta, image_width as f64, image_height as f64)
                        .resample(50)
                        .convex_hull()
                        .verify();

                    polygon.bbox().and_then(|bbox| {
                        if bbox.height() < self.min_height || bbox.width() < self.min_width {
                            return None;
                        }
                        let confidence = polygon.area() as f32 / bbox.area();
                        if confidence < self.confs[0] {
                            return None;
                        }
                        let bbox = bbox.with_confidence(confidence).with_id(0);
                        let mbr = polygon
                            .mbr()
                            .map(|mbr| mbr.with_confidence(confidence).with_id(0));

                        Some((polygon, bbox, mbr))
                    })
                })
                .fold(
                    || (Vec::new(), Vec::new(), Vec::new()),
                    |mut acc, (polygon, bbox, mbr)| {
                        acc.0.push(polygon);
                        acc.1.push(bbox);
                        if let Some(mbr) = mbr {
                            acc.2.push(mbr);
                        }
                        acc
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new(), Vec::new()),
                    |mut acc, (polygons, bboxes, mbrs)| {
                        acc.0.extend(polygons);
                        acc.1.extend(bboxes);
                        acc.2.extend(mbrs);
                        acc
                    },
                );

            Some(
                Y::default()
                    .with_bboxes(&y_bbox)
                    .with_polygons(&y_polygons)
                    .with_mbrs(&y_mbrs),
            )
        })
        .collect();

    Ok(Ys(ys))

    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }
}