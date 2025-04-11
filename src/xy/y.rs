use aksr::Builder;

use crate::{Bbox, Keypoint, Mask, Mbr, Nms, Polygon, Prob, Text, X};

/// Represents a prediction output with various optional components.
#[derive(Builder, Clone, PartialEq, Default)]
pub struct Y {
    pub texts: Option<Vec<Text>>,
    pub embedding: Option<X>,
    pub probs: Option<Prob>,
    pub bboxes: Option<Vec<Bbox>>,
    pub keypoints: Option<Vec<Vec<Keypoint>>>,
    pub mbrs: Option<Vec<Mbr>>,
    pub polygons: Option<Vec<Polygon>>,
    pub masks: Option<Vec<Mask>>,
}

impl std::fmt::Debug for Y {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("Y");
        if let Some(xs) = &self.texts {
            if !xs.is_empty() {
                f.field("texts", &xs);
            }
        }
        if let Some(xs) = &self.probs {
            f.field("probs", &xs);
        }
        if let Some(xs) = &self.bboxes {
            if !xs.is_empty() {
                f.field("bboxes", &xs);
            }
        }
        if let Some(xs) = &self.mbrs {
            if !xs.is_empty() {
                f.field("mbrs", &xs);
            }
        }
        if let Some(xs) = &self.keypoints {
            if !xs.is_empty() {
                f.field("keypoints", &xs);
            }
        }
        if let Some(xs) = &self.polygons {
            if !xs.is_empty() {
                f.field("polygons", &xs);
            }
        }
        if let Some(xs) = &self.masks {
            if !xs.is_empty() {
                f.field("masks", &xs);
            }
        }
        if let Some(x) = &self.embedding {
            f.field("embedding", &x);
        }
        f.finish()
    }
}

impl Y {
    /// Creates a new `Y` instance with specified texts and bboxes.
    pub fn new(texts: Option<Vec<Text>>, bboxes: Option<Vec<Bbox>>) -> Self {
        Self {
            texts,
            bboxes,
            ..Self::default()
        }
    }

    /// Returns a reference to the bounding boxes, if any.
    pub fn hbbs(&self) -> Option<&[Bbox]> {
        self.bboxes.as_deref()
    }

    /// Returns a reference to the minimum bounding rectangles, if any.
    pub fn obbs(&self) -> Option<&[Mbr]> {
        self.mbrs.as_deref()
    }

    /// Applies non-maximum suppression (NMS) to either bboxes or mbrs.
    pub fn apply_nms(mut self, iou_threshold: f32) -> Self {
        if let Some(bboxes) = &mut self.bboxes {
            Self::nms(bboxes, iou_threshold);
        } else if let Some(mbrs) = &mut self.mbrs {
            Self::nms(mbrs, iou_threshold);
        }
        self
    }

    /// Performs non-maximum suppression on a vector of items implementing `Nms`.
    pub fn nms<T: Nms>(items: &mut Vec<T>, iou_threshold: f32) {
        items.sort_by(|b1, b2| {
            b2.confidence()
                .partial_cmp(&b1.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut current_index = 0;
        for index in 0..items.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = items[prev_index].iou(&items[index]);
                if iou > iou_threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                items.swap(current_index, index);
                current_index += 1;
            }
        }
        items.truncate(current_index);
    }
}