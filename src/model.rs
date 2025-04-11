use anyhow::{anyhow, Result};
use ort::{
    session::builder::{GraphOptimizationLevel, SessionBuilder},
    session::{Input, Output, Session},
    value::Value,
};
use std::collections::HashMap;
use std::sync::Arc;
#[allow(unused_imports)]
use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};

#[cfg(debug_assertions)]
use tracing;

#[derive(Debug)]
pub struct Model {
    session: Arc<Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl Model {
    pub fn new(model_path: &str, device: &str) -> Result<Self> {
        let mut builder = SessionBuilder::new()?;

        let builder = if device.to_lowercase().contains("cuda") {
            #[cfg(feature = "cuda")]
            {
                let cuda_ep = CUDAExecutionProvider::default();
                if cuda_ep.is_available()? {
                    cuda_ep.register(&mut builder)?;
                    builder
                } else {
                    #[cfg(debug_assertions)]
                    tracing::warn!("CUDA requested but not available; falling back to CPU");
                    builder
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow!(
                    "CUDA support not enabled. Enable the 'cuda' feature."
                ));
            }
        } else {
            builder
        };

        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let input_names = session.inputs.iter().map(|i| i.name.clone()).collect();
        let output_names = session.outputs.iter().map(|o| o.name.clone()).collect();

        Ok(Self {
            session: Arc::new(session),
            input_names,
            output_names,
        })
    }

    pub fn run(&self, tensors: Vec<Value>) -> Result<Vec<Value>> {
        if tensors.len() != self.input_names.len() {
            return Err(anyhow!(
                "Expected {} input tensors, got {}",
                self.input_names.len(),
                tensors.len()
            ));
        }

        for (tensor, input) in tensors.iter().zip(self.session.inputs.iter()) {
            let shape = tensor.shape()?;
            let expected_dims = input
                .input_type
                .tensor_dimensions()
                .ok_or_else(|| anyhow!("Failed to get dimensions for input '{}'", input.name))?;

            if shape.len() != expected_dims.len() {
                return Err(anyhow!(
                    "Tensor shape length {} does not match expected dimensions length {} for input '{}'",
                    shape.len(),
                    expected_dims.len(),
                    input.name
                ));
            }

            let is_valid = shape.iter().zip(expected_dims.iter()).all(|(&s, &e)| {
                e <= 0 || s as i64 == e // e <= 0 means dynamic dimension
            });
            if !is_valid {
                return Err(anyhow!(
                    "Tensor shape {:?} does not match expected dimensions {:?} for input '{}'",
                    shape,
                    expected_dims,
                    input.name
                ));
            }
        }

        let inputs: HashMap<&str, Value> = self
            .input_names
            .iter()
            .map(|n| n.as_str())
            .zip(tensors.into_iter())
            .collect();

        let outputs = self.session.as_ref().run(inputs)?;

        let results: Vec<Value> = outputs.into_iter().map(|(_, v)| v).collect();

        Ok(results)
    }

    pub fn input_metadata(&self) -> &[Input] {
        &self.session.inputs
    }

    pub fn output_metadata(&self) -> &[Output] {
        &self.session.outputs
    }

    #[cfg(debug_assertions)]
    pub fn debug_summary(&self) {
        static EMPTY_DIMS: Vec<i64> = Vec::new();
        tracing::debug!(
            "Model Summary:\n\
             Inputs ({}): {}\n\
             Outputs ({}): {}",
            self.input_names.len(),
            self.input_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let dims = self.session.inputs[i].input_type.tensor_dimensions();
                    format!(
                        "\n  - {}: {} (dims: {:?})",
                        i,
                        name,
                        dims.unwrap_or(&EMPTY_DIMS)
                    )
                })
                .collect::<String>(),
            self.output_names.len(),
            self.output_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let dims = self.session.outputs[i].output_type.tensor_dimensions();
                    format!(
                        "\n  - {}: {} (dims: {:?})",
                        i,
                        name,
                        dims.unwrap_or(&EMPTY_DIMS)
                    )
                })
                .collect::<String>()
        );
    }

    #[cfg(not(debug_assertions))]
    pub fn debug_summary(&self) {}
}