use rag_rs::Args;
use llm::Model;

pub fn load(args: &Args) -> Result<(Box<dyn Model>, &str), Box<dyn std::error::Error>>{
    let source = args.to_tokenizer_source();
    let arch = args.model_architecture;
    let path = &args.model_path;
    let params = llm::ModelParameters::default();
    let model = llm::load_dynamic(
        Some(arch),
        &path,
        source,
        params,
        llm::load_progress_callback_stdout,
    )
    .map_err(|err| {
        Box::new(err) as Box<dyn std::error::Error>
    })?;
    let query = args
        .query
        .as_deref()
        .unwrap_or("This is a default query");
    Ok((model, query))
} 

pub fn get_embeddings(
    model: &dyn llm::Model,
    inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> Vec<f32> {
    let mut session = model.start_session(Default::default());
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let vocab = model.tokenizer();
    let beginning_of_sentence = true;
    let query_token_ids = vocab
        .tokenize(query, beginning_of_sentence)
        .unwrap()
        .iter()
        .map(|(_, tok)| *tok)
        .collect::<Vec<_>>();
    model.evaluate(&mut session, &query_token_ids, &mut output_request);
    output_request.embeddings.unwrap()
}

pub fn load_and_embed(args: Args) -> Result<(String, Vec<f32>), Box<dyn std::error::Error>> {
    let (model, query) = load(&args)?;
    let infer_params = llm::InferenceParameters::default();
    let embeddings = get_embeddings(model.as_ref(), &infer_params, query);
    Ok((query.to_string(), embeddings))
}
