use anyhow::Result;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    Condition, Filter, HnswConfigDiff, CreateCollection, SearchPoints, VectorParams, VectorsConfig, ScoredPoint,
};
use std::convert::TryInto;
use std::path::PathBuf;
use std::sync::Mutex;
use serde::Deserialize;
use serde_json::json;
use csv::ReaderBuilder;
use llm::Model;

use crate::utils::{Query, Args};
use crate::embeddings::get_embeddings;
use rayon::prelude::*;

#[derive(Debug, Deserialize)]
struct Row {
    id: u64,
    title: String,
    description: String,
}

pub struct Store {
    pub url: String,
    pub client: QdrantClient,
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
        
    pub async fn search(&self, entry: Query, index: &str, group_id: Option<u64>) -> Result<Vec<ScoredPoint>> {
        let embedding = entry.embedding.clone();
        
        let filter = match group_id {
            Some(id) => Some(Filter::must([Condition::matches("group_id", id as i64)])),
            _ => None,
        };

        let neighbours = self.client
            .search_points(&SearchPoints {
                collection_name: index.into(),
                vector: embedding,
                filter: filter,
                limit: 10,
                with_payload: Some(true.into()),
                ..Default::default()
            })
			.await?;
        Ok(neighbours.result)
    }

    pub async fn insert(&self, points: Vec<PointStruct>, index: &str, size: u64, isolation: bool) -> Result<()> {
        println!("Inserting {} points into index '{}'...", points.len(), index);
        if !(self.has_index(index).await?) {
            let _ = self.create_index(index, size, isolation).await;
        }

        self.client
            .upsert_points_blocking(index, None, points, None)
            .await?;
        println!("Completed inserted points");
        Ok(())
    }

    pub async fn create_index(&self, index: &str, size: u64, isolation: bool) -> Result<()> {
        let isolation_config = if isolation { Some(HnswConfigDiff {
            payload_m: Some(16),
            m: Some(0),
            ..Default::default()
        }) } else { None };

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
            hnsw_config: isolation_config, 
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
    // Optimization #1: Stream rows of CSV instead of reading 
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

fn embed_rows(batch: Vec<Row>, model: &Box<dyn Model>) -> Result<Vec<PointStruct>>{
    let embd = batch.par_iter().map(|r| get_embeddings(model.as_ref(), &r.description));
    let points = Mutex::new(vec![]);

    embd.enumerate().for_each(|(i, em)| {
        let id = batch[i].id;
        let payload: Payload = json!(
            {
                "metadata": {
                    "title": batch[i].title,
                    "description": batch[i].description, 
                },
                "group_id": id,
            }
        )
        .try_into()
        .unwrap();
        let point = PointStruct::new(id, em.clone(), payload);    
        let mut points = points.lock().unwrap();
        points.push(point);
    });

    let points_vec = points.lock().unwrap().clone(); 
    Ok(points_vec)
}

pub async fn read_embed_insert(args: &Args, client: &Store, index: &str, model: &Box<dyn Model>, isolation: bool) -> Result<()> {
    // Janky way to use args to determine whether we go through this upload process.
    // Should fix it sometime.
    if !(args.upload.unwrap_or(false)) { return Ok(()); }
    println!("Start process for inserting from csv into vector db...");
	let size = 2560;
    let path = args.path.clone().unwrap();
    let rows = read_rows(&path);
    let Ok(embedded) = embed_rows(rows?, model) else { todo!() };
    
	let _ = client.insert(embedded, index, size, isolation).await;
    Ok(())
} 
