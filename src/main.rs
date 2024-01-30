use clap::Parser;
use anyhow::Result;
use rag_rs::Entry;
use rag_rs::Args;

mod utils;
mod store;
mod embeddings;

// TODO:
//	1. index vector db job (upload csv -> llm -> store embeddings)
//	2. reformat prompt to use retrieved points
//	3. prompt llm
//	4. move quadrantclient loading to centralized location

fn main() -> Result<()> {
    let args = Args::parse();
    let _index = args.index.clone();
    let index = _index
        .as_deref()
        .unwrap_or("first-index");
    
    let (query, query_embeddings) = embeddings::load_and_embed(args);

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

