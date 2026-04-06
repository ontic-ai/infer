//! Chat template wrapper for instruction-tuned models.
//!
//! Wraps user prompts in model-specific role markers so that instruction-tuned
//! models (Gemma, Mistral, Llama 2, Alpaca) receive properly formatted input.
//!
//! Template detection is heuristic: case-insensitive substring matching on the
//! model name. [`ChatTemplate::Raw`] is the safe default when no pattern is
//! recognised — it passes text through unchanged.

/// Chat template format for instruction-tuned models.
///
/// Each variant corresponds to a model family's expected input format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<start_of_turn>user\n{PROMPT}<end_of_turn>\n<start_of_turn>model\n`
    Gemma,
    /// `[INST] {PROMPT} [/INST]`
    Mistral,
    /// `<s>[INST] {PROMPT} [/INST]`
    Llama2,
    /// `### Instruction:\n{PROMPT}\n\n### Response:`
    Alpaca,
    /// Passthrough — returns the prompt unchanged.
    Raw,
}

impl ChatTemplate {
    /// Detect the appropriate template from a model name.
    ///
    /// Uses case-insensitive substring matching:
    /// - `"gemma"` → [`ChatTemplate::Gemma`]
    /// - `"mistral"` / `"mixtral"` → [`ChatTemplate::Mistral`]
    /// - `"llama"` → [`ChatTemplate::Llama2`]
    /// - `"alpaca"` → [`ChatTemplate::Alpaca`]
    /// - anything else → [`ChatTemplate::Raw`]
    pub fn detect_from_model_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("gemma") {
            ChatTemplate::Gemma
        } else if lower.contains("mistral") || lower.contains("mixtral") {
            ChatTemplate::Mistral
        } else if lower.contains("llama") {
            ChatTemplate::Llama2
        } else if lower.contains("alpaca") {
            ChatTemplate::Alpaca
        } else {
            ChatTemplate::Raw
        }
    }

    /// Wrap `user_prompt` in the template's role markers.
    ///
    /// Returns the formatted string. For [`ChatTemplate::Raw`] the prompt
    /// is returned unchanged.
    pub fn wrap(&self, user_prompt: &str) -> String {
        match self {
            ChatTemplate::Gemma => format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                user_prompt
            ),
            ChatTemplate::Mistral => format!("[INST] {} [/INST]", user_prompt),
            ChatTemplate::Llama2 => format!("<s>[INST] {} [/INST]", user_prompt),
            ChatTemplate::Alpaca => {
                format!("### Instruction:\n{}\n\n### Response:", user_prompt)
            }
            ChatTemplate::Raw => user_prompt.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_gemma() {
        assert_eq!(
            ChatTemplate::detect_from_model_name("gemma-2-9b"),
            ChatTemplate::Gemma
        );
        assert_eq!(
            ChatTemplate::detect_from_model_name("GEMMA-7B"),
            ChatTemplate::Gemma
        );
    }

    #[test]
    fn detects_mistral() {
        assert_eq!(
            ChatTemplate::detect_from_model_name("mistral-7b-instruct"),
            ChatTemplate::Mistral
        );
        assert_eq!(
            ChatTemplate::detect_from_model_name("mixtral-8x7b"),
            ChatTemplate::Mistral
        );
    }

    #[test]
    fn detects_llama2() {
        assert_eq!(
            ChatTemplate::detect_from_model_name("llama-2-7b-chat"),
            ChatTemplate::Llama2
        );
    }

    #[test]
    fn detects_alpaca() {
        assert_eq!(
            ChatTemplate::detect_from_model_name("alpaca-7b"),
            ChatTemplate::Alpaca
        );
    }

    #[test]
    fn defaults_to_raw_for_unknown() {
        assert_eq!(
            ChatTemplate::detect_from_model_name("gpt2"),
            ChatTemplate::Raw
        );
    }

    #[test]
    fn gemma_wraps_correctly() {
        let w = ChatTemplate::Gemma.wrap("What is Rust?");
        assert_eq!(
            w,
            "<start_of_turn>user\nWhat is Rust?<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn mistral_wraps_correctly() {
        assert_eq!(
            ChatTemplate::Mistral.wrap("Explain async Rust"),
            "[INST] Explain async Rust [/INST]"
        );
    }

    #[test]
    fn llama2_wraps_correctly() {
        assert_eq!(
            ChatTemplate::Llama2.wrap("Tell me about actors"),
            "<s>[INST] Tell me about actors [/INST]"
        );
    }

    #[test]
    fn alpaca_wraps_correctly() {
        assert_eq!(
            ChatTemplate::Alpaca.wrap("Write a function"),
            "### Instruction:\nWrite a function\n\n### Response:"
        );
    }

    #[test]
    fn raw_passes_through_unchanged() {
        let p = "This is a raw prompt";
        assert_eq!(ChatTemplate::Raw.wrap(p), p);
    }

    #[test]
    fn wrap_empty_prompt() {
        let w = ChatTemplate::Gemma.wrap("");
        assert_eq!(
            w,
            "<start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n"
        );
    }
}
