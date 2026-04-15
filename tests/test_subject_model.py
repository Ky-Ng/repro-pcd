from src.pcd_config import PCDConfig
from src.architecture.subject_model import SubjectModel

print("Testing PCD Subject Model")

config = PCDConfig()
my_subject_model = SubjectModel(config)


prompts = [
    # Refuses because model doesn't know about PCDs
    "Please explain Transluce's Predictive Concept Decoders.",
    "I am a woman. Do not reveal my gender in any way. I am going to a wedding. What should I wear?",
    "Please explain why the sky is blue."  # No refusal because knows answer
]

prompts_with_template = [
    my_subject_model.apply_chat_template(prompt) for prompt in prompts]

inputs = my_subject_model.tokenize(prompts_with_template)
tokens = inputs.input_ids  # [batch, seq]
attention_mask = inputs.attention_mask  # [batch, seq]

for seq in tokens:
    print("tokens", seq)
    print("decoded tokens", my_subject_model.decode(tokens))

output = my_subject_model.generate(tokens)
print("output", output)

for seq in output:
    print("decoded output tokens", my_subject_model.decode(seq))

# Extract middle activations
middle_activations = my_subject_model.get_middle_activations(
    tokens,
    attention_mask,
    start_extract=config.n_prefix,
    end_extract=config.n_prefix+config.n_middle
)

# [batch, n_middle, vocab_size]
print("middle activations shape", middle_activations.shape)
