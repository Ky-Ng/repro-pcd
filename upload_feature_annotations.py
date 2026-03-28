from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='feature_annotations',
    path_in_repo='feature_annotations',      
    repo_id='kylelovesllms/AUTO-PCD-Qwen2.5-1.5B-Instruct-FineWeb-Pretraining',
    repo_type='model',
)
print('Done!')
