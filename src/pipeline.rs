use llm::Model;
use crate::utils::Query;
use crate::embeddings::{get_embeddings};
use crate::store::{Store};
use std::{convert::Infallible, io::Write};
use qdrant_client::qdrant::ScoredPoint;

pub struct RAG {
    pub prompt: String,
    pub reprompt: String,
    pub group_id: Option<u64>, 
}

impl RAG {
    pub async fn retrieve(&mut self, index: &str, client: &Store, model: &Box<dyn Model>) -> String {
        let query_embeddings = get_embeddings(model.as_ref(), &self.prompt);
        let entry = Query {
            query: self.prompt.clone(),
            embedding: query_embeddings,
        };
        let Ok(result) = client.search(entry, index, self.group_id).await else { todo!() };

        let docs = Self::parse_retrieved(result, 3);    
        let v = docs[0].get("description").map_or("not found".to_string(), |tv| tv.to_string());
        self.reprompt = self.reprompt.replace("_RETRIEVED_", &v).replace("_QUERY_", &self.prompt);       
        v
    }

    pub fn prompt(&self, model: &Box<dyn Model>) -> Result<(), Box<dyn std::error::Error>> {
        println!("{:?}", self.reprompt);
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

    fn parse_retrieved(mut documents: Vec<ScoredPoint>, k: usize) -> Vec<serde_json::Value> {
        let mut docs = Vec::new();
        for (i, doc) in documents.iter_mut().enumerate() {
            if i >= k { break; }
            let payload = &mut doc.payload;
            let text = payload.remove("metadata").unwrap().into_json();
            docs.push(text);
        }
        docs
    }
    
    pub async fn run(&mut self, index: &str, client: &Store, model: &Box<dyn Model>) -> Result<(), Box<dyn std::error::Error>>{
        let _ = self.retrieve(index, client, model).await;
        let _ = self.prompt(model);
        Ok(())  
    }

}
