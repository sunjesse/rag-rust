use csv::Reader;
use serde::Deserialize;
use serde_json::{json, Value};

mod embeddings;

struct Row {
	r1: String,
	r2: String,
}

pub fn read_rows(&str path) -> Result<Vec<Row>>{
	let mut csv = Reader::from_path(path)?;
	let mut batch = Vec::new(); 
 
	for r in csv.deserialize(){
		let record: Row = r?;
		batch.push(record);	
	}
	Ok(batch)
}

fn embed_rows(args: Args, batch: Vec<Row>) -> Result<Vec<Pointstruct>{
	let source = args.to_tokenizer_source();
    let arch = args.model_architecture;
    let path = args.model_path;
    let query = args
        .query
        .as_deref()
        .unwrap_or("This is a default query");
    let params = llm::ModelParameters::default();
    let model = llm::load_dynamic(
        Some(arch),
        &path,
        source,
        params,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {arch} model from {path:?}: {err}")
    });
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
		)
		let point = PointStruct::new(i, e.clone(), payload); 	
		points.push(point);
	}
	Ok(points);
}
