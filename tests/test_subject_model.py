from src.pcd_config import PCDConfig
from src.subject_model import SubjectModel

print("Testing PCD Subject Model")

config = PCDConfig()
my_subject_model = SubjectModel(config)

prompt = "Please explain Transluce's Predictive Concept Decoders" # Refuses because doesn't know
# prompt = "Please explain why the sky is blue" # No refusal because knows answer
prompt_with_template = my_subject_model.apply_chat_template(prompt)

tokens = my_subject_model.tokenize(prompt_with_template)
print("tokens", tokens)
print("decoded tokens", my_subject_model.decode(tokens))

output = my_subject_model.generate(tokens)
print("output", output)
print("decoded output tokens", my_subject_model.decode(output))