use clap::Parser;
use anyhow::Result;
use utils::Entry;
use utils::Args;
use qdrant_client::prelude::*;
use std::{io::Write, path::PathBuf};

mod utils;
mod store;
mod embeddings;
mod pipeline;

// TODO:
//  DONE 1. index vector db job (upload csv -> llm -> store embeddings)
//  DONE 2. reformat prompt to use retrieved points
//  DONE 3. prompt llm
//  4. move quadrantclient loading to centralized location
//	5. abstract away csv upload.

fn main() -> Result<()> {
    let args = Args::parse();
    let _index = args.index.clone();
    let index = _index
        .as_deref()
        .unwrap_or("first-index");

	let client = QdrantClient::from_url("http://localhost:6334").build()?;
    //store::read_embed_insert(args, &client); 

    let Ok((model, query)) = embeddings::load(&args) else { todo!() };
	let rag = pipeline::RAG { prompt: query.to_string() };
	let v = rag.retrieve(&index, &client, &model);
	let reprompt = utils::form_query("Tell me something about a movie with description: ", &v);
	println!("{:?}", reprompt);
	rag.prompt(&reprompt, &model);
    Ok(())
}
