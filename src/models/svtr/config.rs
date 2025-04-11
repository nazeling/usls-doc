use crate::{Hub, MinOptMax, Options};
use anyhow::Result;
use serde_json::Value;

impl Options {
    pub fn svtr(
        batch: Option<MinOptMax>,
        channels: Option<MinOptMax>,
        height: Option<MinOptMax>,
        width: Option<MinOptMax>,
    ) -> Self {
        let mut opts = Self::default()
            .with_model_name("svtr")
            .with_model_ixx(0, 0, batch.unwrap_or((1, 1, 8).into()))
            .with_model_ixx(0, 1, channels.unwrap_or(3.into()))
            .with_model_ixx(0, 2, height.unwrap_or(48.into()))
            .with_model_ixx(0, 3, width.unwrap_or((320, 960, 1600).into()))
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_padding_value(0)
            .with_normalize(true);
        if let Some(ref path) = opts.generation_config_file {
            if let Ok(json) = std::fs::read_to_string(path)
                .map_err(anyhow::Error::from)
                .and_then(|s| serde_json::from_str::<Value>(&s).map_err(anyhow::Error::from))
            {
                if let Some(conf) = json.get("confidence_threshold").and_then(|v| v.as_f64()) {
                    opts = opts.with_class_confs(&[conf as f32]);
                }
            }
        } else {
            opts = opts.with_class_confs(&[0.2]);
        }
        opts
    }

    pub fn validate_vocab(&self) -> Result<(), anyhow::Error> {
        if let Some(ref vocab) = self.vocab_txt {
            let path = format!("{}/{}", self.model_name, vocab);
            if !std::path::PathBuf::from(&path).exists() && Hub::default().try_fetch(&path).is_err()
            {
                anyhow::bail!("Vocab file '{}' not found", vocab);
            }
        }
        Ok(())
    }

    pub fn svtr_ch() -> Self {
        Self::svtr(None, None, None, None).with_vocab_txt("vocab-v1-ppocr-rec-ch.txt")
    }

    pub fn svtr_en() -> Self {
        Self::svtr(None, None, None, None).with_vocab_txt("vocab-v1-ppocr-rec-en.txt")
    }

    pub fn ppocr_rec_v3_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v3-ch.onnx")
    }

    pub fn ppocr_rec_v4_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v4-ch.onnx")
    }

    pub fn ppocr_rec_v3_en() -> Self {
        Self::svtr_en().with_model_file("ppocr-v3-en.onnx")
    }

    pub fn ppocr_rec_v4_en() -> Self {
        Self::svtr_en().with_model_file("ppocr-v4-en.onnx")
    }

    pub fn ppocr_rec_v4_server_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v4-server-ch.onnx")
    }

    pub fn svtr_v2_server_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-server-ch.onnx")
    }

    pub fn repsvtr_ch() -> Self {
        Self::svtr_ch().with_model_file("repsvtr-ch.onnx")
    }

    pub fn svtr_v2_teacher_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-distill-teacher-ch.onnx")
    }

    pub fn svtr_v2_student_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-distill-student-ch.onnx")
    }
}
