use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};
use rayon::prelude::*;

use crate::{BaseModelTextual, BaseModelVisual, Options, Text, Ts, Xs, Ys, X, Y};

#[derive(Debug)]
pub struct TrOCR {
    encoder: BaseModelVisual,
    decoder: BaseModelTextual,
    decoder_merged: BaseModelTextual,
    max_length: u32,
    eos_token_id: u32,
    decoder_start_token_id: u32,
    ts: Ts,
    options: Options, // Still included for consistency, though not used here
}

impl TrOCR {
    pub fn new(
        options_encoder: Options,
        options_decoder: Options,
        options_decoder_merged: Options,
    ) -> Result<Self> {
        let encoder = BaseModelVisual::new(options_encoder.clone())?;
        let decoder = BaseModelTextual::new(options_decoder)?;
        let decoder_merged = BaseModelTextual::new(options_decoder_merged)?;
        let ts = Ts::merge(&[
            encoder.engine().ts(),
            decoder.engine().ts(),
            decoder_merged.engine().ts(),
        ]);

        let max_length = 100;
        let eos_token_id = 2; // </s>
        let decoder_start_token_id = 1; // <s>

        Ok(Self {
            encoder,
            decoder,
            decoder_merged,
            max_length,
            eos_token_id,
            decoder_start_token_id,
            ts,
            options: options_encoder,
        })
    }

    pub fn forward(&mut self, images: &[DynamicImage]) -> Result<Ys> {
        let xs = self.encoder.preprocess(images)?;
        let encoder_outputs = self.encoder.inference(xs)?;
        let mut ys = Vec::new();

        for (_name, encoder_output) in encoder_outputs.iter() {
            let token_ids = self.generate(encoder_output)?;
            let text = self.decode_tokens(&token_ids.0)?;
            ys.push(Y::default().with_texts(&[Text::from(text)])); // Use public API
        }
        Ok(Ys(ys))
    }

    pub fn force_encoder_rgb(&mut self, enable: bool) {
        self.encoder.force_rgb(enable);
    }

    fn generate(&mut self, encoder_hidden_states: &X) -> Result<X> {
        let batch_size = encoder_hidden_states.0.shape()[0];
        let mut input_ids =
            Array::from_elem((batch_size, 1), self.decoder_start_token_id as f32).into_dyn();
        let mut generated_tokens = Vec::new();
        let max_length = 50;

        println!("EOS token ID: {}", self.eos_token_id);
        println!("Initial input_ids shape: {:?}", input_ids.shape());

        for i in 0..max_length {
            let inputs = Xs::from(vec![
                X::from(input_ids.clone()),
                encoder_hidden_states.clone(),
            ]);
            let outputs = self.decoder.inference(inputs)?;

            let logits = &outputs[0].0;
            let next_tokens = logits
                .slice(s![.., -1, ..])
                .map_axis(Axis(1), |logit| {
                    logit
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i as f32)
                        .unwrap_or(1.0)
                })
                .into_raw_vec_and_offset()
                .0;

            println!("Iteration {}: next_tokens: {:?}", i, next_tokens);

            if next_tokens.iter().any(|&t| t as u32 == self.eos_token_id) {
                println!(
                    "EOS token ({}) detected, stopping generation",
                    self.eos_token_id
                );
                generated_tokens.push(next_tokens.clone());
                break;
            }

            println!(
                "Iteration {}: input_ids shape before update: {:?}",
                i,
                input_ids.shape()
            );
            generated_tokens.push(next_tokens.clone());
            let next_token_array = Array::from_vec(next_tokens);
            let next_token_col = next_token_array.to_shape(IxDyn(&[batch_size, 1]))?;
            input_ids = ndarray::concatenate(Axis(1), &[input_ids.view(), next_token_col.view()])?
                .into_dyn();

            println!(
                "Iteration {}: input_ids shape after update: {:?}",
                i,
                input_ids.shape()
            );
            println!("Iteration {}: current input_ids: {:?}", i, input_ids);
        }

        let arrays: Vec<_> = generated_tokens
            .iter()
            .map(|v| Array::from_vec(v.clone()))
            .collect();
        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        let token_array = ndarray::stack(Axis(1), &views)?.into_dyn();
        println!("Final token_array shape: {:?}", token_array.shape());
        println!("Final token_array: {:?}", token_array);
        Ok(X::from(token_array))
    }

    fn decode_tokens(&self, token_ids: &Array<f32, IxDyn>) -> Result<String> {
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
        let tokenizer = self
            .decoder
            .processor()
            .tokenizer()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized"))?;
        let decoded = tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {}", e))?;
        Ok(decoded)
    }

    fn decode(&mut self, generated: X) -> Result<Ys> {
        let batch_size = generated.0.shape()[0];
        let token_ids: Vec<Vec<u32>> = (0..batch_size)
            .map(|i| {
                generated
                    .0
                    .slice(s![i, ..])
                    .iter()
                    .map(|&x| x as u32)
                    .collect()
            })
            .collect();

        let texts = token_ids
            .par_iter()
            .map(|ids| {
                self.decoder
                    .processor()
                    .tokenizer()
                    .expect("Tokenizer required for TrOCR decoding")
                    .decode(ids, true)
                    .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
            })
            .collect::<Result<Vec<String>>>()?;

        let y_vec: Vec<Y> = texts
            .into_iter()
            .map(|text| Y::default().with_texts(&[Text::from(text)]))
            .collect();
        let ys = Ys::from(y_vec);
        Ok(ys)
    }

    pub fn summary(&self) {
        self.ts.summary();
    }
}
