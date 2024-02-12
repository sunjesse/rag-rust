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
    let index = args.index
        .as_deref()
        .unwrap_or("first-index");
    let reprompt_path = args.rp_path
        .as_deref()
        .unwrap_or(Path::new("./src/prompts/reprompt/reprompt.txt"));
    let isolation = args.isolation
        .unwrap_or(false);
    let group_id = args.group_id
        .unwrap_or(0);

    let client = store::Store::new("http://localhost:6334").unwrap();
    let Ok((model, query)) = embeddings::load(&args) else { todo!() };

    //let _ = store::read_embed_insert(&args, &client, index, &model, isolation); 

    let reprompt = fs::read_to_string(reprompt_path).unwrap();
    let mut pipe = pipeline::RAG { 
        prompt: query.to_string(), 
        reprompt: reprompt.to_string(),
        group_id: group_id,
    };
    let _ = pipe.run(index, &client, &model);
    Ok(())
}
