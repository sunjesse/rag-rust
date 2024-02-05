use qdrant_client::prelude::*;
use llm::Model;
use crate::utils::Entry;
use crate::embeddings::{get_embeddings};
use crate::store::{search};
use std::{convert::Infallible, io::Write};

pub struct RAG {
	pub prompt: String,
	pub reprompt: String,
}

impl RAG {
	pub fn retrieve(&mut self, index: &str, client: &QdrantClient, model: &Box<dyn Model>) -> String {
		let infer_params = llm::InferenceParameters::default();
		let query_embeddings = get_embeddings(model.as_ref(), &infer_params, &self.prompt);
		let entry = Entry {
			id: 11,
			query: self.prompt.clone(),
			embedding:query_embeddings,
		};
		let rt = tokio::runtime::Runtime::new().unwrap();
		let result = rt.block_on(async {
			search(entry, index, client).await
		});
		let v = result.unwrap().get("description").map_or("not found".to_string(), |tv| tv.to_string());
		self.reprompt = self.reprompt.replace("_RETRIEVED_", &v); 		
		println!("{:?}", self.reprompt);
		return v;
	}

	pub fn prompt(&self, model: &Box<dyn Model>) -> Result<(), Box<dyn std::error::Error>> {
		let mut session = model.start_session(Default::default());
		let res = session.infer::<Infallible>(
			model.as_ref(),
			&mut rand::thread_rng(),
			&llm::InferenceRequest {
				prompt: (&self.reprompt).into(),
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
			Ok(result) => println!("{result}"),
			Err(err) => println!("{err}"),
		}
		Ok(())
	}

}
