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

pub struct Store {
    pub url: String,
    pub client:  QdrantClient,
}

impl Store {
    pub fn new(store_url: &str) -> Result<Self>{
        let url: String = store_url.to_string();
        let client: QdrantClient = QdrantClient::from_url(store_url).build()?;
        Ok(Self{
            url: url,
            client: client,
        })
    }
        
    pub async fn search(&self, entry: Query, index: &str) -> Result<serde_json::Value> {
        let embedding = entry.embedding.clone();

        let neighbours = self.client
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

    pub async fn insert(&self, points: Vec<PointStruct>, index: &str) -> Result<()> {
        println!("Inserting {} points into index '{}'...", points.len(), index);
        if self.has_index(index).await? == false {
			// TODO: Un-hardcode the size.
            self.create_index(index, 2560).await;
        }

        self.client
            .upsert_points_blocking(index, None, points, None)
            .await?;
        Ok(())
    }

    pub async fn create_index(&self, index: &str, size: u64) -> Result<()> {
        self.client
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

    pub async fn delete_index(&self, index: &str) -> Result<()> { 
        self.client.delete_collection(index).await?;
        Ok(())
    }
    
    async fn has_index(&self, index: &str) -> Result<bool> {
        let list = self.client.list_collections().await?;
        for c in list.collections.iter() {
            if c.name == index {
                return Ok(true);
            }   
        }
        Ok(false)
    }
    
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

fn embed_rows(args: &Args, batch: Vec<Row>) -> Result<Vec<PointStruct>>{
    let Ok((model, _)) = load(args) else { todo!() };
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

pub fn read_embed_insert(args: &Args, client: &Store, index: &str) -> Result<()> {
    let path = args.path.clone().unwrap();
    let rows = read_rows(&path);
    let embedded = embed_rows(args, rows?);
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _ = rt.block_on(async {
        client.insert(embedded?, index).await
    });
    Ok(())
} 
