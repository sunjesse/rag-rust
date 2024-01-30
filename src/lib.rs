use clap::Parser;
use std::path::PathBuf;

pub struct Entry {
	pub id: u8,
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

