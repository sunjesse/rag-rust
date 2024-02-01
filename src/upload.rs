use csv::Reader;
use serde::Deserialize;
use serde_json::{json, Value};

mod embeddings;

struct Row {
    r1: String,
    r2: String,
}

fn read_rows(path: &str) -> Result<Vec<Row>> {
    let mut csv = Reader::from_path(path)?;
    let mut batch = Vec::new(); 
 
    for r in csv.deserialize(){
        let record: Row = r?;
        batch.push(record); 
    }
    Ok(batch)
}

fn embed_rows(args: Args, batch: Vec<Row>) -> Result<Vec<PointStruct>{
    let (model, _) = load(&args)?;
    let infer_params = llm::InferenceParameters::default();
    
    let embd = batch.iter().map(|r| embeddings::get_embeddings(model.as_ref(), &infer_params, r.r2));
    let points = Vec::new();
    let i = 0;
    for (j, e) in &embd.iter().enumerate() {
        let payload: Payload = json!(
            {
                "rows": {
                    "0": batch[j].r1,
                    "1": batch[j].r2, 
                }
            }
        );
        let point = PointStruct::new(i, e.clone(), payload);    
        points.push(point);
    }
    Ok(points)
}

pub fn read_embed_insert(args: Args) -> Result<()> {
	let path = args.path;
	let index = args.index;
	let rows = read_rows(path);
	let embedded = embed_rows(args, rows);
	embeddings::insert(embedded, index);
	Ok()
} 
