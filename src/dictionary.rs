use std::collections::HashSet;

pub(crate) const MAX_DICTIONARY_ENTRIES: usize = 64;
const MAX_DICTIONARY_TERM_CHARS: usize = 160;
const MAX_PROMPT_BYTES: usize = 600;
const TERM_SEPARATOR: &str = ", ";

pub fn sanitize_dictionary_entries(entries: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    entries
        .iter()
        .map(|raw| raw.trim())
        .filter(|trimmed| !trimmed.is_empty())
        .filter(|trimmed| seen.insert(trimmed.to_lowercase()))
        .map(|trimmed| {
            let capped: String = trimmed.chars().take(MAX_DICTIONARY_TERM_CHARS).collect();
            capped.trim_end().to_string()
        })
        .take(MAX_DICTIONARY_ENTRIES)
        .collect()
}

pub fn build_dictionary_prompt(entries: &[String]) -> Option<String> {
    let mut prompt = String::new();
    for term in sanitize_dictionary_entries(entries) {
        let separator = if prompt.is_empty() {
            ""
        } else {
            TERM_SEPARATOR
        };
        if prompt.len() + separator.len() + term.len() + 1 > MAX_PROMPT_BYTES {
            break;
        }
        prompt.push_str(separator);
        prompt.push_str(&term);
    }
    if prompt.is_empty() {
        return None;
    }
    prompt.push('.');
    Some(prompt)
}

#[cfg(test)]
mod tests {
    use super::{build_dictionary_prompt, sanitize_dictionary_entries};

    #[test]
    fn sanitize_dictionary_entries_deduplicates_case_insensitively() {
        let cleaned = sanitize_dictionary_entries(&[
            "  Glimpse ".to_string(),
            "glimpse".to_string(),
            "  ".to_string(),
            "Speech".to_string(),
        ]);

        assert_eq!(cleaned, vec!["Glimpse".to_string(), "Speech".to_string()]);
    }

    #[test]
    fn build_dictionary_prompt_joins_terms() {
        let prompt = build_dictionary_prompt(&["alpha".to_string(), "beta".to_string()]);
        assert_eq!(prompt.as_deref(), Some("alpha, beta."));
    }

    #[test]
    fn build_dictionary_prompt_is_none_without_usable_terms() {
        assert_eq!(build_dictionary_prompt(&[]), None);
        assert_eq!(build_dictionary_prompt(&["   ".to_string()]), None);
        assert_eq!(build_dictionary_prompt(&["\u{1F600}".repeat(160)]), None);
    }
}
