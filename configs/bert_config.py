bert_base_config = {
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  'embedding_size':128,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,  # 4 x hidden_size
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12, # hidden / 64
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  'share_parameter_across_layers':True,
  "type_vocab_size": 2,
  "vocab_size": 21228
}
