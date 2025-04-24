use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelChoice {
    DocSynth,
    DocStructBench,
}

impl ModelChoice {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelChoice::DocSynth => "DocSynth-300K",
            ModelChoice::DocStructBench => "DocStructBench",
        }
    }
}

/// Provide a default model choice for use in structs deriving `Default`.
impl Default for ModelChoice {
    fn default() -> Self {
        // DocSynth is the recommended default model
        ModelChoice::DocSynth
    }
}
