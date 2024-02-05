use anyhow::Result;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollection, SearchPoints, VectorParams, VectorsConfig,
};
use serde_json::json;
use std::convert::TryInto;
use std::path::PathBuf;
use serde::Deserialize;
use csv::ReaderBuilder;

use crate::utils::{Query, Args};
use crate::embeddings::{load, get_embeddings};

#[derive(Debug, Deserialize)]
struct Row {
    r1: String,
    r2: String,
    r3: String,
}

pub async fn search(entry: Query, index: &str, client: &QdrantClient) -> Result<serde_json::Value> {
    let embedding = entry.embedding.clone();
    
    let neighbours = client
        .search_points(&SearchPoints {
            collection_name: index.into(),
            vector: embedding,
            //filter: Some(Filter::all([Condition::matches("id", entry.id)])),
            limit: 10,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;
    let nearest = neighbours.result.into_iter().next().unwrap();
    let mut payload = nearest.payload;
    println!("{:?}", payload);
    let text = payload.remove("metadata").unwrap().into_json();
    println!("Found {}", text);
    Ok(text)
}

pub async fn insert(points: Vec<PointStruct>, index: &str, client: &QdrantClient) -> Result<()> {
    client
        .upsert_points_blocking(index, None, points, None)
        .await?;
    Ok(())
}

pub async fn create_index(index: &str, size: u64, client: &QdrantClient) -> Result<()> {
    client
        .create_collection(&CreateCollection {
            collection_name: index.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: size,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?;
    Ok(())  
}

pub async fn delete_index(index: &str) -> Result<()> { 
    let client = QdrantClient::from_url("http://localhost:6334").build()?;
    client.delete_collection(index).await?;
    Ok(())
}


fn read_rows(path: &PathBuf) -> Result<Vec<Row>> {
    let mut csv = ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut batch = Vec::new(); 

    for row in csv.deserialize() {
        match row {
            Ok(record) => {
                let record: Row = record;
                println!("{:?}", record);
                batch.push(record);
            }
            Err(e) => {
                eprintln!("Error deserializing record: {}", e);
            }
        }
    } 
    Ok(batch)
}

fn embed_rows(args: Args, batch: Vec<Row>) -> Result<Vec<PointStruct>>{
    let Ok((model, _)) = load(&args) else { todo!() };
    let embd = batch.iter().map(|r| get_embeddings(model.as_ref(), &r.r2));
    let mut points = Vec::new();
    let mut i = 0;
    for (j, em) in embd.enumerate() {
        let payload: Payload = json!(
            {
                "metadata": {
                    "title": batch[j].r1,
                    "description": batch[j].r3, 
                }
            }
        )
        .try_into()
        .unwrap();
        let point = PointStruct::new(i, em.clone(), payload);    
        i += 1;
        points.push(point);
    }
    Ok(points)
}

pub fn read_embed_insert(args: Args, client: &QdrantClient) -> Result<()> {
    let path = args.path.clone().unwrap();
    let index = args.index.clone().unwrap();
    let rows = read_rows(&path);
    let embedded = embed_rows(args, rows?);
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _ = rt.block_on(async {
        insert(embedded?, &index, client).await
    });
    Ok(())
} 
