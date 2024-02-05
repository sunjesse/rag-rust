use clap::Parser;
use std::path::PathBuf;

pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product = dot(v1, v2);
    let magnitude1 = magnitude(v1);
    let magnitude2 = magnitude(v2);

    dot_product / (magnitude1 * magnitude2)
}

pub fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

pub fn form_query(a: &str, b: &str) -> String {
	let result = [a, b].join("\n");
	return result;
}

pub struct Query {
    pub query: String,
    pub embedding: Vec<f32>,
}


#[derive(Parser)]
pub struct Args {
    pub model_architecture: llm::ModelArchitecture,
    pub model_path: PathBuf,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
    #[arg(long, short = 'q')]
    pub query: Option<String>,
    #[arg(long, short = 'i')]
    pub index: Option<String>,
    #[arg(long, short = 'p')]
    pub path: Option<PathBuf>,
}

impl Args {
    pub fn to_tokenizer_source(&self) -> llm::TokenizerSource {
        match (&self.tokenizer_path, &self.tokenizer_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --tokenizer-path and --tokenizer-repository");
            }
            (Some(path), None) => llm::TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::TokenizerSource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::TokenizerSource::Embedded,
        }
    }
}

