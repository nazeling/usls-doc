use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use log::{error, info};
use ndarray::{s, Array, Axis};
use rayon::prelude::*;

pub use ort::value::Tensor;

use crate::{
    elapsed,
    yolo::{BoxType, YOLOPredsFormat},
    Bbox, DynConf, Engine, Keypoint, Mask, Mbr, Ops, Options, Polygon, Prob, Processor, Task, Ts,
    Version, Xs, Ys, Y,
};

#[derive(Debug, Builder)]
pub struct YOLO {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    layout: YOLOPredsFormat,
    task: Task,
    version: Option<Version>,
    names: Vec<String>,
    names_kpt: Vec<String>,
    nc: usize,
    nk: usize,
    confs: DynConf,
    kconfs: DynConf,
    iou: f32,
    find_contours: bool,
    processor: Processor,
    ts: Ts,
    spec: String,
    classes_excluded: Vec<usize>,
    classes_retained: Vec<usize>,
}

impl TryFrom<Options> for YOLO {
    type Error = anyhow::Error;

    fn try_from(options: Options) -> Result<Self, Self::Error> {
        Self::new(options)
    }
}

impl YOLO {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        // Dynamic sizing: get batch, height, and width from the engine (with fallbacks)
        let (batch, height, width, ts, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let task: Option<Task> = match &options.model_task {
            Some(task) => Some(task.clone()),
            None => match engine.try_fetch("task") {
                Some(x) => match x.as_str() {
                    "classify" => Some(Task::ImageClassification),
                    "detect" => Some(Task::ObjectDetection),
                    "pose" => Some(Task::KeypointsDetection),
                    "segment" => Some(Task::InstanceSegmentation),
                    "obb" => Some(Task::OrientedObjectDetection),
                    x => {
                        error!("Unsupported YOLO Task: {}", x);
                        None
                    }
                },
                None => None,
            },
        };

        // Determine task and layout based on provided options and engine metadata
        let version = options.model_version;
        let (layout, task) = match &options.yolo_preds_format {
            // Customized layout provided by the user
            Some(layout) => {
                let task_parsed = layout.task();
                let task = match task {
                    Some(ref t) => {
                        if t != &task_parsed {
                            anyhow::bail!(
                                "Task specified: {:?} is inconsistent with parsed from yolo_preds_format: {:?}",
                                t,
                                task_parsed
                            );
                        }
                        task_parsed
                    }
                    None => task_parsed,
                };
                (layout.clone(), task)
            }
            // Determine layout based on version and task
            None => match (task, version) {
                (Some(task), Some(version)) => {
                    let layout = match (task.clone(), version) {
                        (Task::ImageClassification, Version(5, 0)) => {
                            YOLOPredsFormat::n_clss().apply_softmax(true)
                        }
                        (Task::ImageClassification, Version(8, 0) | Version(11, 0)) => {
                            YOLOPredsFormat::n_clss()
                        }
                        (Task::ObjectDetection, Version(5, 0) | Version(6, 0) | Version(7, 0)) => {
                            YOLOPredsFormat::n_a_cxcywh_confclss()
                        }
                        (
                            Task::ObjectDetection,
                            Version(8, 0) | Version(9, 0) | Version(11, 0) | Version(12, 0),
                        ) => YOLOPredsFormat::n_cxcywh_clss_a(),
                        (Task::ObjectDetection, Version(10, 0)) => {
                            YOLOPredsFormat::n_a_xyxy_confcls().apply_nms(false)
                        }
                        (Task::KeypointsDetection, Version(8, 0) | Version(11, 0)) => {
                            YOLOPredsFormat::n_cxcywh_clss_xycs_a()
                        }
                        (Task::InstanceSegmentation, Version(5, 0)) => {
                            YOLOPredsFormat::n_a_cxcywh_confclss_coefs()
                        }
                        (Task::InstanceSegmentation, Version(8, 0) | Version(11, 0)) => {
                            YOLOPredsFormat::n_cxcywh_clss_coefs_a()
                        }
                        (Task::OrientedObjectDetection, Version(8, 0) | Version(11, 0)) => {
                            YOLOPredsFormat::n_cxcywh_clss_r_a()
                        }
                        (t, v) => {
                            anyhow::bail!("Task: {:?} is unsupported for Version: {:?}. Try using `.with_yolo_preds()` for customization.", t, v)
                        }
                    };
                    (layout, task)
                }
                (None, Some(version)) => {
                    let layout = match version {
                        Version(6, 0) | Version(7, 0) => YOLOPredsFormat::n_a_cxcywh_confclss(),
                        Version(9, 0) | Version(12, 0) => YOLOPredsFormat::n_cxcywh_clss_a(),
                        Version(10, 0) => YOLOPredsFormat::n_a_xyxy_confcls().apply_nms(false),
                        _ => {
                            anyhow::bail!(
                                "No clear YOLO Task specified for Version: {:?}.",
                                version
                            )
                        }
                    };
                    (layout, Task::ObjectDetection)
                }
                (Some(_), None) => {
                    anyhow::bail!("No clear YOLO Version specified for Task.")
                }
                (None, None) => {
                    anyhow::bail!("No clear YOLO Task and Version specified.")
                }
            },
        };

        // Fetch class names and determine number of classes
        let names: Option<Vec<String>> = match Self::fetch_names_from_onnx(&engine) {
            Some(names_parsed) => match &options.class_names {
                Some(names) => {
                    if names.len() == names_parsed.len() {
                        Some(names.clone())
                    } else {
                        anyhow::bail!(
                            "The lengths of parsed class names: {} and user-defined class names: {} do not match.",
                            names_parsed.len(),
                            names.len(),
                        )
                    }
                }
                None => Some(names_parsed),
            },
            None => options.class_names.clone(),
        };

        let (nc, names) = match (options.nc(), names) {
            (_, Some(names)) => (names.len(), names.to_vec()),
            (Some(nc), None) => (nc, Self::n2s(nc)),
            (None, None) => {
                anyhow::bail!(
                    "Neither class names nor the number of classes were specified. \
                    Consider specifying them with Options::default().with_nc() or Options::default().with_class_names()"
                );
            }
        };

        // Determine keypoint names and count if task is keypoint detection
        let (nk, names_kpt) = if let Task::KeypointsDetection = task {
            let nk = Self::fetch_nk_from_onnx(&engine).or(options.nk());
            match (&options.keypoint_names, nk) {
                (Some(names), Some(nk)) => {
                    if names.len() != nk {
                        anyhow::bail!(
                            "The lengths of user-defined keypoint names: {} and nk parsed: {} do not match.",
                            names.len(),
                            nk,
                        );
                    }
                    (nk, names.clone())
                }
                (Some(names), None) => (names.len(), names.clone()),
                (None, Some(nk)) => (nk, Self::n2s(nk)),
                (None, None) => anyhow::bail!(
                    "Neither keypoint names nor the number of keypoints were specified when performing KeypointsDetection. \
                    Consider specifying them with Options::default().with_nk() or Options::default().with_keypoint_names()"
                ),
            }
        } else {
            (0, vec![])
        };

        let confs = DynConf::new(options.class_confs(), nc);
        let kconfs = DynConf::new(options.keypoint_confs(), nk);
        let iou = options.iou().unwrap_or(0.45);
        let classes_excluded = options.classes_excluded().to_vec();
        let classes_retained = options.classes_retained().to_vec();
        let find_contours = options.find_contours();
        info!(
            "YOLO Version: {}, Task: {:?}, Category Count: {}, Keypoint Count: {}",
            version.map_or("Unknown".into(), |x| x.to_string()),
            task,
            nc,
            nk,
        );

        Ok(Self {
            engine,
            height,
            width,
            batch,
            task,
            version,
            spec,
            layout,
            names,
            names_kpt,
            confs,
            kconfs,
            iou,
            nc,
            nk,
            find_contours,
            classes_excluded,
            classes_retained,
            processor,
            ts,
        })
    }

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;
        Ok(x.to_xs(None))
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
    fn fetch_names_from_onnx(_engine: &Engine) -> Option<Vec<String>> {
        None
    }
    fn n2s(n: usize) -> Vec<String> {
        (0..n).map(|i| i.to_string()).collect()
    }
    fn fetch_nk_from_onnx(_engine: &Engine) -> Option<usize> {
        None
    }
    pub fn summary(&mut self) {
        self.ts.summary();
    }

    fn postprocess(&self, xs: Xs) -> Result<Ys> {
        let protos = if xs.len() == 2 { Some(&xs[1]) } else { None };
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, preds)| {
                let mut y = Y::default();

                // Parse predictions from layout
                let (
                    slice_bboxes,
                    slice_id,
                    slice_clss,
                    slice_confs,
                    slice_kpts,
                    slice_coefs,
                    slice_radians,
                ) = self.layout.parse_preds(preds, self.nc);

                if let Task::ImageClassification = self.task {
                    let x = if self.layout.apply_softmax {
                        let exps = slice_clss.mapv(|x| x.exp());
                        let stds = exps.sum_axis(Axis(0));
                        exps / stds
                    } else {
                        slice_clss.into_owned()
                    };
                    let probs = Prob::default()
                        .with_probs(&x.into_raw_vec_and_offset().0)
                        .with_names(&self.names.iter().map(|x| x.as_str()).collect::<Vec<_>>());
                    return Some(y.with_probs(probs));
                }

                let (image_height, image_width) = self.processor.image_sizes()[idx];
                let ratio = self.processor.scale_factors()[idx][0];

                let (y_bboxes, y_mbrs) = slice_bboxes?
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, bbox)| {
                        let (class_id, confidence) = match &slice_id {
                            Some(ids) => (ids[[i, 0]] as _, slice_clss[[i, 0]] as _),
                            None => {
                                let (class_id, &confidence) = slice_clss
                                    .slice(s![i, ..])
                                    .into_iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.total_cmp(b.1))?;
                                match &slice_confs {
                                    None => (class_id, confidence),
                                    Some(slice_confs) => {
                                        (class_id, confidence * slice_confs[[i, 0]])
                                    }
                                }
                            }
                        };

                        if (!self.classes_excluded.is_empty()
                            && self.classes_excluded.contains(&class_id))
                            || (!self.classes_retained.is_empty()
                                && !self.classes_retained.contains(&class_id))
                            || (confidence < self.confs[class_id])
                        {
                            return None;
                        }

                        let bbox = bbox.mapv(|x| x / ratio);
                        let bbox = if self.layout.is_bbox_normalized {
                            (
                                bbox[0] * self.width as f32,
                                bbox[1] * self.height as f32,
                                bbox[2] * self.width as f32,
                                bbox[3] * self.height as f32,
                            )
                        } else {
                            (bbox[0], bbox[1], bbox[2], bbox[3])
                        };
                        let (cx, cy, x, y, w, h) = match self.layout.box_type()? {
                            BoxType::Cxcywh => {
                                let (cx, cy, w, h) = bbox;
                                let x = (cx - w / 2.).max(0.);
                                let y = (cy - h / 2.).max(0.);
                                (cx, cy, x, y, w, h)
                            }
                            BoxType::Xyxy => {
                                let (x, y, x2, y2) = bbox;
                                let (w, h) = (x2 - x, y2 - y);
                                ((x + x2) / 2., (y + y2) / 2., x, y, w, h)
                            }
                            BoxType::Xywh => {
                                let (x, y, w, h) = bbox;
                                ((x + w / 2.), (y + h / 2.), x, y, w, h)
                            }
                            BoxType::Cxcyxy => {
                                let (cx, cy, x2, y2) = bbox;
                                let w = (x2 - cx) * 2.;
                                let h = (y2 - cy) * 2.;
                                (cx, cy, (x2 - w).max(0.), (y2 - h).max(0.), w, h)
                            }
                            BoxType::XyCxcy => {
                                let (x, y, cx, cy) = bbox;
                                let w = (cx - x) * 2.;
                                let h = (cy - y) * 2.;
                                (cx, cy, x, y, w, h)
                            }
                        };

                        let (y_bbox, y_mbr) = match &slice_radians {
                            Some(slice_radians) => {
                                let radians = slice_radians[[i, 0]];
                                let (w, h, radians) = if w > h {
                                    (w, h, radians)
                                } else {
                                    (h, w, radians + std::f32::consts::PI / 2.)
                                };
                                let radians = radians % std::f32::consts::PI;
                                let mbr = Mbr::from_cxcywhr(
                                    cx as f64,
                                    cy as f64,
                                    w as f64,
                                    h as f64,
                                    radians as f64,
                                )
                                .with_confidence(confidence)
                                .with_id(class_id as isize)
                                .with_name(&self.names[class_id]);
                                (None, Some(mbr))
                            }
                            None => {
                                let bbox = Bbox::default()
                                    .with_xywh(x, y, w, h)
                                    .with_confidence(confidence)
                                    .with_id(class_id as isize)
                                    .with_id_born(i as isize)
                                    .with_name(&self.names[class_id]);
                                (Some(bbox), None)
                            }
                        };

                        Some((y_bbox, y_mbr))
                    })
                    .collect::<(Vec<_>, Vec<_>)>();

                let y_bboxes: Vec<Bbox> = y_bboxes.into_iter().flatten().collect();
                let y_mbrs: Vec<Mbr> = y_mbrs.into_iter().flatten().collect();

                if !y_mbrs.is_empty() {
                    y = y.with_mbrs(&y_mbrs);
                    if self.layout.apply_nms {
                        y = y.apply_nms(self.iou);
                    }
                    return Some(y);
                }

                if !y_bboxes.is_empty() {
                    y = y.with_bboxes(&y_bboxes);
                    if self.layout.apply_nms {
                        y = y.apply_nms(self.iou);
                    }
                }

                // Process keypoints if present
                if let Some(pred_kpts) = slice_kpts {
                    let kpt_step = self.layout.kpt_step().unwrap_or(3);
                    if let Some(bboxes) = y.bboxes() {
                        let y_kpts = bboxes
                            .into_par_iter()
                            .filter_map(|bbox| {
                                let pred = pred_kpts.slice(s![bbox.id_born(), ..]);
                                let kpts = (0..self.nk)
                                    .into_par_iter()
                                    .map(|i| {
                                        let kx = pred[kpt_step * i] / ratio;
                                        let ky = pred[kpt_step * i + 1] / ratio;
                                        let kconf = pred[kpt_step * i + 2];
                                        if kconf < self.kconfs[i] {
                                            Keypoint::default()
                                        } else {
                                            Keypoint::default()
                                                .with_id(i as isize)
                                                .with_confidence(kconf)
                                                .with_xy(
                                                    kx.max(0.0f32).min(image_width as f32),
                                                    ky.max(0.0f32).min(image_height as f32),
                                                )
                                                .with_name(&self.names_kpt[i])
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                Some(kpts)
                            })
                            .collect::<Vec<_>>();
                        y = y.with_keypoints(&y_kpts);
                    }
                }

                // Process instance segmentation if coefficients are provided
                if let Some(coefs) = slice_coefs {
                    if let Some(bboxes) = y.bboxes() {
                        let (y_polygons, y_masks) = bboxes
                            .into_par_iter()
                            .filter_map(|bbox| {
                                let coefs = coefs.slice(s![bbox.id_born(), ..]).to_vec();
                                let proto = protos.as_ref()?.slice(s![idx, .., .., ..]);
                                let (nm, mh, mw) = proto.dim();
                                let coefs = Array::from_shape_vec((1, nm), coefs).ok()?;
                                let proto = proto.to_shape((nm, mh * mw)).ok()?;
                                let mask = coefs.dot(&proto);
                                let mask = Ops::resize_lumaf32_u8(
                                    &mask.into_raw_vec_and_offset().0,
                                    mw as _,
                                    mh as _,
                                    image_width as _,
                                    image_height as _,
                                    true,
                                    "Bilinear",
                                )
                                .ok()?;
                                let mut mask: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                                    image::ImageBuffer::from_raw(
                                        image_width as _,
                                        image_height as _,
                                        mask,
                                    )?;
                                let (xmin, ymin, xmax, ymax) =
                                    (bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax());
                                for (y, row) in mask.enumerate_rows_mut() {
                                    for (x, _, pixel) in row {
                                        if x < xmin as _
                                            || x > xmax as _
                                            || y < ymin as _
                                            || y > ymax as _
                                        {
                                            *pixel = image::Luma([0u8]);
                                        }
                                    }
                                }
                                let polygons = if self.find_contours {
                                    let contours: Vec<imageproc::contours::Contour<i32>> =
                                        imageproc::contours::find_contours_with_threshold(&mask, 0);
                                    contours
                                        .into_par_iter()
                                        .map(|c| {
                                            let mut poly = Polygon::default()
                                                .with_id(bbox.id())
                                                .with_points_imageproc(&c.points)
                                                .verify();
                                            if let Some(name) = bbox.name() {
                                                poly = poly.with_name(name);
                                            }
                                            poly
                                        })
                                        .max_by(|a, b| a.area().total_cmp(&b.area()))?
                                } else {
                                    Polygon::default()
                                };
                                let mut mask_obj =
                                    Mask::default().with_mask(mask).with_id(bbox.id());
                                if let Some(name) = bbox.name() {
                                    mask_obj = mask_obj.with_name(name);
                                }
                                Some((polygons, mask_obj))
                            })
                            .collect::<(Vec<_>, Vec<_>)>();
                        if !y_polygons.is_empty() {
                            y = y.with_polygons(&y_polygons);
                        }
                        if !y_masks.is_empty() {
                            y = y.with_masks(&y_masks);
                        }
                    }
                }
                Some(y)
            })
            .collect();
        Ok(ys.into())
    }
}
