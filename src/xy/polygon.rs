use crate::{Bbox, Mbr};
use aksr::Builder;
use anyhow::Result;
use fast_image_resize::{images::Image, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use geo::{
    coord, point, polygon, Area, BoundingRect, Centroid, ConvexHull, Euclidean, Length, LineString,
    MinimumRotatedRect, Point, Simplify,
};
use image::{DynamicImage, GenericImageView, RgbImage, RgbaImage};

#[derive(Builder, Clone, PartialEq)]
pub struct Polygon {
    polygon: geo::Polygon,
    id: isize,
    name: Option<String>,
    confidence: f32,
}

impl Default for Polygon {
    fn default() -> Self {
        Self {
            polygon: polygon![],
            id: -1,
            name: None,
            confidence: 0.,
        }
    }
}

impl std::fmt::Debug for Polygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polygon")
            .field("perimeter", &self.perimeter())
            .field("area", &self.area())
            .field("count", &self.count())
            .field("id", &self.id)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl Polygon {
    pub fn with_points_imageproc(mut self, points: &[imageproc::point::Point<i32>]) -> Self {
        let v = points
            .iter()
            .map(|p| coord! { x: p.x as f64, y: p.y as f64 })
            .collect::<Vec<_>>();
        self.polygon = geo::Polygon::new(LineString::from(v), vec![]);
        self
    }

    pub fn with_points(mut self, points: &[Vec<f32>]) -> Self {
        let v = points
            .iter()
            .map(|p| coord! { x: p[0] as f64, y: p[1] as f64 })
            .collect::<Vec<_>>();
        self.polygon = geo::Polygon::new(LineString::from(v), vec![]);
        self
    }

    pub fn label(&self, with_name: bool, with_conf: bool, decimal_places: usize) -> String {
        let mut label = String::new();
        if with_name {
            label.push_str(
                &self
                    .name
                    .as_ref()
                    .unwrap_or(&self.id.to_string())
                    .to_string(),
            );
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

    pub fn is_closed(&self) -> bool {
        self.polygon.exterior().is_closed()
    }

    pub fn count(&self) -> usize {
        self.polygon.exterior().points().len()
    }

    pub fn perimeter(&self) -> f64 {
        Euclidean.length(self.polygon.exterior())
    }

    pub fn area(&self) -> f64 {
        self.polygon.unsigned_area()
    }

    pub fn centroid(&self) -> Option<(f32, f32)> {
        self.polygon
            .centroid()
            .map(|x| (x.x() as f32, x.y() as f32))
    }

    pub fn bbox(&self) -> Option<Bbox> {
        self.polygon.bounding_rect().map(|x| {
            Bbox::default().with_xyxy(
                x.min().x as f32,
                x.min().y as f32,
                x.max().x as f32,
                x.max().y as f32,
            )
        })
    }

    pub fn mbr(&self) -> Option<Mbr> {
        MinimumRotatedRect::minimum_rotated_rect(&self.polygon)
            .map(|x| Mbr::from_line_string(x.exterior().to_owned()))
    }

    pub fn convex_hull(mut self) -> Self {
        self.polygon = self.polygon.convex_hull();
        self
    }

    pub fn simplify(mut self, eps: f64) -> Self {
        self.polygon = self.polygon.simplify(&eps);
        self
    }

    pub fn resample(mut self, num_samples: usize) -> Self {
        let points = self.polygon.exterior().to_owned().into_points();
        let mut new_points = Vec::new();
        for i in 0..points.len() {
            let start_point = points[i];
            let end_point = points[(i + 1) % points.len()];
            new_points.push(start_point);
            let dx = end_point.x() - start_point.x();
            let dy = end_point.y() - start_point.y();
            for j in 1..num_samples {
                let t = (j as f64) / (num_samples as f64);
                let new_x = start_point.x() + t * dx;
                let new_y = start_point.y() + t * dy;
                new_points.push(point! { x: new_x, y: new_y });
            }
        }
        self.polygon = geo::Polygon::new(LineString::from(new_points), vec![]);
        self
    }

    pub fn unclip(mut self, delta: f64, width: f64, height: f64) -> Self {
        let points = self.polygon.exterior().to_owned().into_points();
        let num_points = points.len();
        let mut new_points = Vec::with_capacity(points.len());
        for i in 0..num_points {
            let prev_idx = if i == 0 { num_points - 1 } else { i - 1 };
            let next_idx = (i + 1) % num_points;

            let edge_vector = point! {
                x: points[next_idx].x() - points[prev_idx].x(),
                y: points[next_idx].y() - points[prev_idx].y(),
            };
            let normal_vector = point! {
                x: -edge_vector.y(),
                y: edge_vector.x(),
            };

            let normal_length = (normal_vector.x().powi(2) + normal_vector.y().powi(2)).sqrt();
            if normal_length.abs() < 1e-6 {
                new_points.push(points[i]);
            } else {
                let normalized_normal = point! {
                    x: normal_vector.x() / normal_length,
                    y: normal_vector.y() / normal_length,
                };

                let new_x = points[i].x() + normalized_normal.x() * delta;
                let new_y = points[i].y() + normalized_normal.y() * delta;
                new_points.push(point! {
                    x: new_x.max(0.0).min(width),
                    y: new_y.max(0.0).min(height),
                });
            }
        }
        self.polygon = geo::Polygon::new(LineString::from(new_points), vec![]);
        self
    }

    pub fn verify(mut self) -> Self {
        let mut points = self.polygon.exterior().points().collect::<Vec<_>>();
        Self::remove_duplicates(&mut points);
        self.polygon = geo::Polygon::new(LineString::from(points), vec![]);
        self
    }

    fn remove_duplicates(xs: &mut Vec<Point>) {
        if let Some(first) = xs.first() {
            let p_1st = (first.x() as i32, first.y() as i32);
            while xs.len() > 1 {
                if let Some(last) = xs.last() {
                    if (last.x() as i32, last.y() as i32) == p_1st {
                        xs.pop();
                    } else {
                        break;
                    }
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        xs.retain(|point| seen.insert((point.x() as i32, point.y() as i32)));
    }

    pub fn pipeline(
        points: &[Vec<f32>],
        id: isize,
        confidence: f32,
        min_area: f64,
        min_confidence: f32,
        image: Option<&DynamicImage>,
        crop_size: Option<(u32, u32)>,
    ) -> Result<Option<(Self, Option<DynamicImage>)>> {
        let mut poly = Self::default()
            .with_points(points)
            .with_id(id)
            .with_confidence(confidence)
            .verify();

        if poly.area() < min_area || poly.confidence < min_confidence {
            return Ok(None);
        }

        poly = poly.simplify(1.0);

        let cropped_image = if let Some(img) = image {
            if let Some(bbox) = poly.bbox() {
                let x_min = bbox.xmin() as u32;
                let y_min = bbox.ymin() as u32;
                let width = (bbox.xmax() - bbox.xmin()) as u32;
                let height = (bbox.ymax() - bbox.ymin()) as u32;
                if width > 0
                    && height > 0
                    && x_min + width <= img.width()
                    && y_min + height <= img.height()
                {
                    let crop_buf: RgbaImage = img.view(x_min, y_min, width, height).to_image();
                    let rgba_buf = crop_buf.into_raw();

                    if let Some((target_width, target_height)) = crop_size {
                        let src_image =
                            Image::from_vec_u8(width, height, rgba_buf, PixelType::U8x4)?;
                        let mut resizer = Resizer::new();
                        let options = ResizeOptions::new()
                            .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
                        let mut dst = Image::new(target_width, target_height, PixelType::U8x3);
                        resizer.resize(&src_image, &mut dst, &options)?;
                        Some(DynamicImage::ImageRgb8(
                            RgbImage::from_raw(target_width, target_height, dst.into_vec())
                                .ok_or_else(|| anyhow::anyhow!("Failed to create RgbImage"))?,
                        ))
                    } else {
                        let rgb_buf: Vec<u8> = rgba_buf
                            .chunks(4)
                            .map(|p| [p[0], p[1], p[2]])
                            .flatten()
                            .collect();
                        Some(DynamicImage::ImageRgb8(
                            RgbImage::from_raw(width, height, rgb_buf)
                                .ok_or_else(|| anyhow::anyhow!("Failed to create RgbImage"))?,
                        ))
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Some((poly, cropped_image)))
    }
}
