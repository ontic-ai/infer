#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopOutcome {
    Emit(String),
    Stop(String),
    Wait,
}

#[derive(Debug, Clone)]
pub(crate) struct StopSequenceBuffer {
    stop_sequences: Vec<String>,
    pending: String,
}

impl StopSequenceBuffer {
    pub(crate) fn new(stop_sequences: &[String]) -> Self {
        Self {
            stop_sequences: stop_sequences
                .iter()
                .filter(|sequence| !sequence.is_empty())
                .cloned()
                .collect(),
            pending: String::new(),
        }
    }

    pub(crate) fn push(&mut self, piece: &str) -> StopOutcome {
        self.pending.push_str(piece);

        if let Some(stop_index) = find_earliest_stop(&self.pending, &self.stop_sequences) {
            let emit = self.pending[..stop_index].to_string();
            self.pending.clear();
            return StopOutcome::Stop(emit);
        }

        let keep_len = longest_partial_stop_suffix(&self.pending, &self.stop_sequences);
        let emit_len = self.pending.len().saturating_sub(keep_len);
        if emit_len == 0 {
            return StopOutcome::Wait;
        }

        let tail = self.pending.split_off(emit_len);
        let emit = std::mem::replace(&mut self.pending, tail);
        StopOutcome::Emit(emit)
    }

    pub(crate) fn finish(self) -> Option<String> {
        if self.pending.is_empty() {
            None
        } else {
            Some(self.pending)
        }
    }
}

pub(crate) fn trim_to_stop_sequence(text: &str, stop_sequences: &[String]) -> String {
    let mut buffer = StopSequenceBuffer::new(stop_sequences);
    let mut result = String::new();
    let stopped = match buffer.push(text) {
        StopOutcome::Emit(chunk) => {
            result.push_str(&chunk);
            false
        }
        StopOutcome::Stop(chunk) => {
            result.push_str(&chunk);
            true
        }
        StopOutcome::Wait => false,
    };

    if !stopped && let Some(tail) = buffer.finish() {
        result.push_str(&tail);
    }

    result
}

fn find_earliest_stop(text: &str, stop_sequences: &[String]) -> Option<usize> {
    stop_sequences
        .iter()
        .filter_map(|sequence| text.find(sequence))
        .min()
}

fn longest_partial_stop_suffix(text: &str, stop_sequences: &[String]) -> usize {
    let mut longest = 0;

    for sequence in stop_sequences {
        for boundary in sequence
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(sequence.len()))
            .skip(1)
        {
            if boundary >= sequence.len() || boundary <= longest || boundary > text.len() {
                continue;
            }

            if text.ends_with(&sequence[..boundary]) {
                longest = boundary;
            }
        }
    }

    longest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_to_stop_sequence_drops_matching_suffix() {
        assert_eq!(
            trim_to_stop_sequence("hello<stop>world", &["<stop>".to_string()]),
            "hello"
        );
    }

    #[test]
    fn buffer_holds_partial_stop_suffix_until_disambiguated() {
        let mut buffer = StopSequenceBuffer::new(&["END".to_string()]);

        assert_eq!(buffer.push("friE"), StopOutcome::Emit("fri".to_string()));
        assert_eq!(buffer.push("x"), StopOutcome::Emit("Ex".to_string()));
        assert_eq!(buffer.finish(), None);
    }

    #[test]
    fn buffer_stops_when_sequence_spans_multiple_pieces() {
        let mut buffer = StopSequenceBuffer::new(&["STOP".to_string()]);

        assert_eq!(buffer.push("abST"), StopOutcome::Emit("ab".to_string()));
        assert_eq!(buffer.push("OPcd"), StopOutcome::Stop(String::new()));
    }

    #[test]
    fn finish_flushes_unmatched_partial_suffix() {
        let mut buffer = StopSequenceBuffer::new(&["END".to_string()]);

        assert_eq!(buffer.push("abcE"), StopOutcome::Emit("abc".to_string()));
        assert_eq!(buffer.finish(), Some("E".to_string()));
    }
}
