Last update - layout detection now works right out of the box with the docstructbench1024 onnx model, but I can't get OCR to work with SVTR/DB, or TROCR, with this repo, so I stopped updating it. 

This repo is being used for document layout processing, because of its excellent Rust support for ONNX, to include Paddle and Yolo models. Do not use this repo if you want all the features of the main branch, because this does not have them, and will keep diverging from the main as more changes are made. If you are doing layout processing and/or OCR you might find the finished product useful.

<h2 align="center">usls</h2>

<p align="center">
    <a href="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml">
        <img src="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml/badge.svg" alt="Rust Continuous Integration Badge">
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/v/usls.svg' alt='usls Version'>
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/msrv/usls-yellow?' alt='Rust MSRV'>
    </a>
    <a href='https://github.com/microsoft/onnxruntime/releases'>
        <img src='https://img.shields.io/badge/onnxruntime-%3E%3D%201.19.0-3399FF' alt='ONNXRuntime MSRV'>
    </a>
    <a href='https://developer.nvidia.com/cuda-toolkit-archive'>
        <img src='https://img.shields.io/badge/cuda-%3E%3D%2012.0-green' alt='CUDA MSRV'>
    </a>
    <a href='https://developer.nvidia.com/tensorrt'>
        <img src='https://img.shields.io/badge/TensorRT-%3E%3D%2012.0-0ABF53' alt='TensorRT MSRV'>
    </a>
    <a href="https://crates.io/crates/usls">
        <img alt="Crates.io Total Downloads" src="https://img.shields.io/crates/d/usls?&color=946CE6">
    </a>
</p>
<p align="center">
    <a href="./examples">
        <img src="https://img.shields.io/badge/Examples-1A86FD?&logo=anki" alt="Examples">
    </a>
    <a href='https://docs.rs/usls'>
        <img src='https://img.shields.io/badge/Docs-usls-yellow?&logo=docs.rs&color=FFA200' alt='usls documentation'>
    </a>
</p>

**usls** is a Rust library integrated with  **ONNXRuntime**, offering a suite of advanced models for **Computer Vision** and **Vision-Language** tasks, including:

- **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10), [YOLO11](https://github.com/ultralytics/ultralytics), [YOLOv12](https://github.com/sunsmarterjie/yolov12)

- **OCR Models**: [FAST](https://github.com/czczup/FAST), [DB(PaddleOCR-Det)](https://arxiv.org/abs/1911.08947), [SVTR(PaddleOCR-Rec)](https://arxiv.org/abs/2205.00159), [SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html), [TrOCR](https://huggingface.co/microsoft/trocr-base-printed), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

## ⛳️ Cargo Features

By default, **none of the following features are enabled**. You can enable them as needed:

- **`auto`**: Automatically downloads prebuilt ONNXRuntime binaries from Pyke’s CDN for supported platforms.

  - If disabled, you'll need to [compile `ONNXRuntime` from source](https://github.com/microsoft/onnxruntime) or [download a precompiled package](https://github.com/microsoft/onnxruntime/releases), and then [link it manually](https://ort.pyke.io/setup/linking).

    <details>
    <summary>👉 For Linux or macOS Users</summary>

    - Download from the [Releases page](https://github.com/microsoft/onnxruntime/releases).
    - Set up the library path by exporting the `ORT_DYLIB_PATH` environment variable:
      ```shell
      export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.20.1
      ```

    </details>
- **`ffmpeg`**: Adds support for video streams, real-time frame visualization, and video export.

  - Powered by [video-rs](https://github.com/oddity-ai/video-rs) and [minifb](https://github.com/emoon/rust_minifb). For any issues related to `ffmpeg` features, please refer to the issues of these two crates.
- **`cuda`**: Enables the NVIDIA TensorRT provider.
- **`trt`**: Enables the NVIDIA TensorRT provider.
- **`mps`**: Enables the Apple CoreML provider.

## 🎈 Example

* **Using `CUDA`**

  ```
  cargo run -r -F cuda --example yolo -- --device cuda:0
  ```
* **Using Apple `CoreML`**

  ```
  cargo run -r -F mps --example yolo -- --device mps
  ```
* **Using `TensorRT`**

  ```
  cargo run -r -F trt --example yolo -- --device trt
  ```
* **Using `CPU`**

  ```
  cargo run -r --example yolo
  ```

## 🥂 Integrate Into Your Own Project

Add `usls` as a dependency to your project's `Cargo.toml`

```Shell
cargo add usls -F cuda
```

Or use a specific commit:

```Toml
[dependencies]
usls = { git = "https://github.com/nazeling/usls-doc", rev = "commit-sha" }
```

## 🥳 If you find this helpful, please give it a star ⭐

## 📌 License

This project is licensed under [LICENSE](LICENSE).
