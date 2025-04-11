use anyhow::Result;
use tokenizers::Encoding;

#[derive(Clone, Debug)]
pub struct DummyTokenizer {
    vocab: Vec<String>,
}

impl DummyTokenizer {
    pub fn new(vocab: Vec<String>) -> Result<Self> {
        if vocab.is_empty() {
            anyhow::bail!("DummyTokenizer requires a non-empty vocab");
        }
        Ok(DummyTokenizer { vocab })
    }

    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }
}

impl crate::TokenizerTrait for DummyTokenizer {
    fn encode(&self, text: &str, _skip_special_tokens: bool) -> Result<Encoding> {
        let len = text.chars().count();
        Ok(Encoding::new(
            vec![0u32; len],              // ids: all zeros
            vec![0u32; len],              // type_ids: all zeros
            vec![String::new(); len],     // tokens: all empty
            vec![Some(0u32); len],        // words: all Some(0)
            vec![(0, 0); len],            // offsets: all zeroes
            vec![0u32; len],              // special_tokens_mask: all zeros
            vec![1u32; len],              // attention_mask: all ones
            vec![],                       // overflowing: none
            std::collections::HashMap::new(), // sequence_ranges
        ))
    }

    fn encode_batch(&self, texts: Vec<String>, skip_special_tokens: bool) -> Result<Vec<Encoding>> {
        Ok(texts.into_iter()
           .map(|t| self.encode(&t, skip_special_tokens).unwrap())
           .collect())
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        if self.vocab.is_empty() {
            return Ok(String::new());
        }
        let word = &self.vocab[0];
        Ok(ids.iter().map(|_| word.clone()).collect())
    }

    fn decode_batch(&self, ids: &[&[u32]], skip_special_tokens: bool) -> Result<Vec<String>> {
        Ok(ids.iter().map(|slice| self.decode(slice, skip_special_tokens).unwrap()).collect())
    }
}
