use clap::Parser;
use anyhow::Result;
use utils::Args;
use qdrant_client::prelude::*;
use std::{io::Write, path::PathBuf};

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

	let client = QdrantClient::from_url("http://localhost:6334").build()?;
    //store::read_embed_insert(args, &client); 

    let Ok((model, query)) = embeddings::load(&args) else { todo!() };
	let reprompt = "Tell me what genre the following movie with description is about: _RETRIEVED_";
	let mut pipe = pipeline::RAG { prompt: query.to_string(), reprompt: reprompt.to_string() };
	let v = pipe.retrieve(&index, &client, &model);
	pipe.prompt(&model);
    Ok(())
}
