use clap::Parser;
use anyhow::Result;
use utils::Args;
use qdrant_client::prelude::*;

mod utils;
mod store;
mod embeddings;
mod pipeline;

fn main() -> Result<()> {
    let args = Args::parse();
    let _index = args.index.clone();
    let index = _index
        .as_deref()
        .unwrap_or("first-index");

    let client = store::Store::new("http://localhost:6334").unwrap();
    //store::read_embed_insert(args, &client); 

    let Ok((model, query)) = embeddings::load(&args) else { todo!() };
    let reprompt = "Tell me what genre the following movie with description is about: _RETRIEVED_";
    let mut pipe = pipeline::RAG { prompt: query.to_string(), reprompt: reprompt.to_string() };
    let _ = pipe.retrieve(&index, &client, &model);
	let _ = pipe.prompt(&model);
    Ok(())
}
