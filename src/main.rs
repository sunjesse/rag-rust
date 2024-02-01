use clap::Parser;
use anyhow::Result;
use rag_rs::Entry;
use rag_rs::Args;
use std::{convert::Infallible, io::Write, path::PathBuf};

mod utils;
mod store;
mod embeddings;

// TODO:
//  DONE / need to test 1. index vector db job (upload csv -> llm -> store embeddings)
//  DONE 2. reformat prompt to use retrieved points
//  DONE 3. prompt llm
//  4. move quadrantclient loading to centralized location

fn main() -> Result<()> {
    let args = Args::parse();
    let _index = args.index.clone();
    let index = _index
        .as_deref()
        .unwrap_or("first-index");

    let Ok((model, query)) = embeddings::load(&args) else { todo!() };
    let infer_params = llm::InferenceParameters::default();
    let query_embeddings = embeddings::get_embeddings(model.as_ref(), &infer_params, query);
    
    //let Ok((query, query_embeddings)) = embeddings::load_and_embed(args) else { todo!() };

    let entry = Entry {
        id: 1,
        query: query.to_string(),
        embedding: query_embeddings,
    };  

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        store::search(entry, index).await
    });
    let v = result.unwrap().get("text").map_or("not found".to_string(), |tv| tv.to_string());;
    println!("HIIIII {}", v);
    
    let mut session = model.start_session(Default::default());
    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: (&v).into(),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );
    match res {
        Ok(result) => println!("HEYYYY: {result}"),
        Err(err) => println!("{err}"),
    }
    Ok(())
}
