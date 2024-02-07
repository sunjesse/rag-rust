use clap::Parser;
use anyhow::Result;
use utils::Args;
use std::fs;
use std::path::Path;

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
    let _reprompt_path = args.rp_path.clone();
    let reprompt_path = _reprompt_path
        .as_deref()
        .unwrap_or(Path::new("./src/prompts/reprompt/reprompt.txt"));

    let client = store::Store::new("http://localhost:6334").unwrap();
    //store::read_embed_insert(args, &client); 

    let Ok((model, query)) = embeddings::load(&args) else { todo!() };
    let reprompt = fs::read_to_string(reprompt_path).unwrap();
    let mut pipe = pipeline::RAG { prompt: query.to_string(), reprompt: reprompt.to_string() };
    let _ = pipe.run(&index, &client, &model);
    Ok(())
}
