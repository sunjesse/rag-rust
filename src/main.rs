use std::path::PathBuf;
use clap::Parser;
use anyhow::Result;
use rag_rs::Entry;

mod utils;
mod store;

#[derive(Parser)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
    #[arg(long, short = 'q')]
    pub query: Option<String>,
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


fn main() -> Result<()> {
    let args = Args::parse();
	let index = "first-index";
    let tokenizer_source = args.to_tokenizer_source();
    let model_architecture = args.model_architecture;
    let model_path = args.model_path;
    let query = args
        .query
        .as_deref()
        .unwrap_or("My favourite animal is the dog");

    let model_params = llm::ModelParameters::default();
    let model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        model_params,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });
    let inference_parameters = llm::InferenceParameters::default();

    let query_embeddings = utils::get_embeddings(model.as_ref(), &inference_parameters, query);

	let entry = Entry {
		id: 1,
		query: query.to_string(),
		embedding: query_embeddings,
	};	

	let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        store::search(entry, index).await;
    });
	Ok(())
}

