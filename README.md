This is the official codes for Self-Ensemble: Mitigating Confidence Distortion for Large Language Models.

# Dependency

```angular2html
Python==3.10.12
numpy==1.24.4
torch==2.2.0
datasets==3.6.0
accelerate==1.7.0
scikit-learn==1.2.0
transformers==4.49.0
tiktoken==0.9.0
```

# Run Self-Ensemble on the QASC Dataset
## Standard Inference
```bash 
python standard_inference/llama_standard_inference.py 
python standard_inference/mistral_standard_inference.py 
python standard_inference/qwen_standard_inference.py 
```
## Self Ensemble
```bash 
python self_ensemble/llama_selfensemble.py 
python self_ensemble/mistral_selfensemble.py  
python self_ensemble/qwen_selfensemble.py 
```
