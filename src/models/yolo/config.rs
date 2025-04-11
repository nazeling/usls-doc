use crate::{Options, ResizeMode};

impl Options {
    pub fn yolo() -> Self {
        Self::default()
            .with_model_name("yolo")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
            .with_find_contours(true)
    }

    pub fn doclayout_yolo_docstructbench() -> Self {
        Self::yolo()
            .with_model_file("doclayout-docstructbench.onnx")
            .with_model_ixx(0, 2, (640, 1024, 1024).into())
            .with_model_ixx(0, 3, (640, 1024, 1024).into())
            .with_class_confs(&[0.4])
            .with_class_names(&[
                "title",
                "plain text",
                "abandon",
                "figure",
                "figure_caption",
                "table",
                "table_caption",
                "table_footnote",
                "isolate_formula",
                "formula_caption",
            ])
    }

    pub fn doclayout_yolo_docsynth() -> Self {
        Self::yolo()
            .with_model_file("doclayout_yolo_docsynth300k_imgsz1600.onnx")
            .with_model_ixx(0, 2, 1600.into())
            .with_model_ixx(0, 3, 1600.into())
            .with_class_confs(&[0.4])
            .with_class_names(&["text", "title", "list", "table", "figure"])
    }
}
