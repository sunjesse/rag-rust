use anyhow::Result;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    Condition, CreateCollection, Filter, SearchPoints, VectorParams, VectorsConfig,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::TryInto;

use rag_rs::Entry;

pub async fn search(entry: Entry, index: &str) -> Result<()> {
	let client = QdrantClient::from_url("http://localhost:6334").build()?;

    client
        .create_collection(&CreateCollection {
            collection_name: index.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: entry.embedding.len() as u64,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?;

	let embedding = entry.embedding.clone();
	
	insert(entry, index).await?;

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
	println!("{}", payload.remove("query").unwrap().into_json());
	Ok(())
}

pub async fn insert(point: Entry, index: &str) -> Result<()> {
	let client = QdrantClient::from_url("http://localhost:6334").build()?;
    let payload: Payload = json!(
        {
            "query": {
                "text": point.query
            }
        }
    )
	.try_into()
	.unwrap();
    let points = vec![PointStruct::new(1, point.embedding.clone(), payload)];
    client
        .upsert_points_blocking(index, None, points, None)
        .await?;
	Ok(())
}

pub async fn delete(index: &str) -> Result<()> { 
	let client = QdrantClient::from_url("http://localhost:6334").build()?;
    client.delete_collection(index).await?;
	Ok(())
}
