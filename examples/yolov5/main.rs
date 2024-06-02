use usls::{
    models::{YOLOTask, YOLOVersion, YOLO},
    Annotator, DataLoader, Options,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_yolo_version(YOLOVersion::V5)
        .with_model("../models/yolov5s-seg.onnx")?
        .with_yolo_task(YOLOTask::Segment)
        // .with_trt(0)
        // .with_fp16(true)
        .with_i00((1, 1, 4).into())
        .with_i02((224, 640, 800).into())
        .with_i03((224, 640, 800).into());
    let mut model = YOLO::new(options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/bus.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("YOLOv5");
    annotator.annotate(&x, &y);

    Ok(())
}
