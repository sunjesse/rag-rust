use clap::Parser;
use anyhow::Result;
use utils::Args;
use std::fs;
use std::path::Path;
use actix_web::{post, App, Error, HttpResponse, HttpServer};
use std::sync::Arc;
use actix_web::web::Data;

use crate::store::{Store};
use llm::Model;

mod utils;
mod store;
mod embeddings;
mod pipeline;

#[post("/query")]
async fn post(req: String, model: Data<Arc<Box<dyn Model>>>, client: Data<Arc<Store>>, args: Data<Arc<Args>>) -> Result<HttpResponse, Error> {
    let index = args.index
        .as_deref()
        .unwrap_or("first-index");

    let reprompt = fs::read_to_string(
        args.rp_path
        .as_deref()
        .unwrap_or(Path::new("./src/prompts/reprompt/reprompt.txt"))).unwrap();

    let mut pipe = pipeline::RAG { 
        prompt: req, 
        reprompt: reprompt.to_string(),
        group_id: args.group_id,
    };
    let _ = pipe.run(&index, &client, &model).await;
    Ok(HttpResponse::Ok().finish())
}

#[post("/upload")]
async fn upload(index: String, model: Data<Arc<Box<dyn Model>>>, client: Data<Arc<Store>>, args: Data<Arc<Args>>) -> Result<HttpResponse, Error> {
    let _ = store::read_embed_insert(
        &args,
        &client,
        &index,
        &model,
        args.isolation.unwrap_or(false)).await; 
    Ok(HttpResponse::Ok().finish())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let client = store::Store::new("http://localhost:6334").unwrap();
    let model = embeddings::load(&args).expect("Failed to load model");
    let model_data = Data::new(Arc::new(model));
    let client_data = Data::new(Arc::new(client));
    let args_data = Data::new(Arc::new(args));

    HttpServer::new(move || {
        App::new()
            .app_data(args_data.clone())
            .app_data(model_data.clone())
            .app_data(client_data.clone())
            .service(upload)
            .service(post)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
