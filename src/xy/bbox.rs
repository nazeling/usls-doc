use aksr::Builder;

use crate::Nms;

/// Bounding Box 2D.
///
/// This struct represents a 2D bounding box with properties such as position, size,
/// class ID, confidence score, optional name, and an ID representing the born state.
#[derive(Builder, Clone, PartialEq, PartialOrd)]
pub struct Bbox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub id: isize,
    pub id_born: isize,
    pub confidence: f32,
    pub name: Option<String>,
}

impl Nms for Bbox {
    /// Returns the confidence score of the bounding box.
    fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Computes the intersection over union (IoU) between this bounding box and another.
    fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }
}

impl Default for Bbox {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
            id: -1,
            id_born: -1,
            confidence: 0.0,
            name: None,
        }
    }
}

impl std::fmt::Debug for Bbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bbox")
            .field("xyxy", &[self.x, self.y, self.xmax(), self.ymax()])
            .field("class_id", &self.id)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl From<(f32, f32, f32, f32)> for Bbox {
    /// Creates a `Bbox` from a tuple of `(x, y, w, h)`.
    fn from((x, y, w, h): (f32, f32, f32, f32)) -> Self {
        Self {
            x,
            y,
            w,
            h,
            ..Self::default()
        }
    }
}

impl From<[f32; 4]> for Bbox {
    /// Creates a `Bbox` from an array of `[x, y, w, h]`.
    fn from([x, y, w, h]: [f32; 4]) -> Self {
        Self {
            x,
            y,
            w,
            h,
            ..Self::default()
        }
    }
}

impl From<(f32, f32, f32, f32, isize, f32)> for Bbox {
    /// Creates a `Bbox` from a tuple of `(x, y, w, h, id, confidence)`.
    fn from((x, y, w, h, id, confidence): (f32, f32, f32, f32, isize, f32)) -> Self {
        Self {
            x,
            y,
            w,
            h,
            id,
            confidence,
            ..Self::default()
        }
    }
}

impl Bbox {
    /// Creates a new `Bbox` with specified parameters.
    pub fn new(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        id: isize,
        id_born: isize,
        confidence: f32,
        name: Option<String>,
    ) -> Self {
        Self {
            x,
            y,
            w,
            h,
            id,
            id_born,
            confidence,
            name,
        }
    }

    /// Sets the bounding box's coordinates using `(x1, y1, x2, y2)` and calculates width and height.
    pub fn with_xyxy(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        self.x = x1;
        self.y = y1;
        self.w = x2 - x1;
        self.h = y2 - y1;
        self
    }

    /// Sets the bounding box's coordinates and dimensions using `(x, y, w, h)`.
    pub fn with_xywh(mut self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
        self
    }

    /// Returns the width of the bounding box.
    pub fn width(&self) -> f32 {
        self.w
    }

    /// Returns the height of the bounding box.
    pub fn height(&self) -> f32 {
        self.h
    }

    /// Returns the minimum x-coordinate of the bounding box.
    pub fn xmin(&self) -> f32 {
        self.x
    }

    /// Returns the minimum y-coordinate of the bounding box.
    pub fn ymin(&self) -> f32 {
        self.y
    }

    /// Returns the maximum x-coordinate of the bounding box.
    pub fn xmax(&self) -> f32 {
        self.x + self.w
    }

    /// Returns the maximum y-coordinate of the bounding box.
    pub fn ymax(&self) -> f32 {
        self.y + self.h
    }

    /// Returns the center x-coordinate of the bounding box.
    pub fn cx(&self) -> f32 {
        self.x + self.w / 2.0
    }

    /// Returns the center y-coordinate of the bounding box.
    pub fn cy(&self) -> f32 {
        self.y + self.h / 2.0
    }

    /// Returns the bounding box coordinates as `(x1, y1, x2, y2)`.
    pub fn xyxy(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.x + self.w, self.y + self.h)
    }

    /// Returns the bounding box coordinates and size as `(x, y, w, h)`.
    pub fn xywh(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    /// Returns the center coordinates and size of the bounding box as `(cx, cy, w, h)`.
    pub fn cxywh(&self) -> (f32, f32, f32, f32) {
        (self.cx(), self.cy(), self.w, self.h)
    }

    /// Returns a label string representing the bounding box.
    pub fn label(&self, with_name: bool, with_conf: bool, decimal_places: usize) -> String {
        let mut label = String::new();
        if with_name {
            label.push_str(self.name.as_deref().unwrap_or(&self.id.to_string()));
        }
        if with_conf {
            if with_name {
                label.push_str(&format!(": {:.decimal_places$}", self.confidence));
            } else {
                label.push_str(&format!("{:.decimal_places$}", self.confidence));
            }
        }
        label
    }

    /// Computes the area of the bounding box.
    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// Computes the perimeter of the bounding box.
    pub fn perimeter(&self) -> f32 {
        2.0 * (self.w + self.h)
    }

    /// Checks if the bounding box is square (i.e., width equals height).
    pub fn is_square(&self) -> bool {
        self.w == self.h
    }

    /// Computes the intersection area between this bounding box and another.
    pub fn intersect(&self, other: &Self) -> f32 {
        let l = self.xmin().max(other.xmin());
        let r = self.xmax().min(other.xmax());
        let t = self.ymin().max(other.ymin());
        let b = self.ymax().min(other.ymax());
        (r - l).max(0.0) * (b - t).max(0.0)
    }

    /// Computes the union area between this bounding box and another.
    pub fn union(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersect(other)
    }

    /// Checks if this bounding box completely contains another.
    pub fn contains(&self, other: &Self) -> bool {
        self.xmin() <= other.xmin()
            && self.xmax() >= other.xmax()
            && self.ymin() <= other.ymin()
            && self.ymax() >= other.ymax()
    }
}

#[cfg(test)]
mod tests_bbox {
    use super::Bbox;

    #[test]
    fn new() {
        let bbox1 = Bbox::from((0.0, 0.0, 5.0, 5.0));
        let bbox2 = Bbox::from([0.0, 0.0, 5.0, 5.0]);
        assert_eq!(bbox1, bbox2);

        let bbox1: Bbox = [0.0, 0.0, 5.0, 5.0].into();
        let bbox2: Bbox = (0.0, 0.0, 5.0, 5.0).into();
        assert_eq!(bbox1, bbox2);

        let bbox1: Bbox = (1.0, 1.0, 5.0, 5.0, 99, 0.99).into();
        let bbox2 = Bbox::new(1.0, 1.0, 5.0, 5.0, 99, -1, 0.99, None);
        assert_eq!(bbox1, bbox2);

        let bbox1: Bbox = (1.0, 1.0, 5.0, 5.0, 1, 1.0).into();
        let bbox2 = Bbox::new(1.0, 1.0, 5.0, 5.0, 1, -1, 1.0, None);
        assert_eq!(bbox1, bbox2);
    }

    #[test]
    fn funcs() {
        let bbox1 = Bbox::new(0.0, 0.0, 5.0, 5.0, -1, -1, 0.0, None);
        let bbox2 = Bbox::new(1.0, 1.0, 5.0, 5.0, -1, -1, 0.0, None);
        assert_eq!(bbox1.intersect(&bbox2), 16.0);
        assert_eq!(bbox1.area(), 25.0);
        assert_eq!(bbox2.area(), 25.0);
        assert_eq!(bbox2.perimeter(), 20.0);
        assert!(bbox2.is_square());
        assert_eq!(bbox1.union(&bbox2), 34.0);

        let bbox3 = Bbox::new(2.0, 2.0, 3.0, 3.0, -1, -1, 0.0, None);
        assert!(!bbox1.contains(&bbox2));
        assert!(bbox1.contains(&bbox3));
        assert!(bbox2.contains(&bbox3));
    }
}