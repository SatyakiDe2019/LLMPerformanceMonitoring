#####################################################
####                                             ####
#### Written By: SATYAKI DE                      ####
#### Written On: 01-Jan-2025                     ####
#### Modified On 05-Jan-2025                     ####
####                                             ####
#### Objective: This is the main python class    ####
#### that will applies all the logic to collect  ####
#### stats involving important KPIs that are as  ####
#### follows:                                    ####
####                                             ####
#### 1. BERT Score (Understanding & Relevance)   ####
#### 2. BLEU Score (Generation Quality)          ####
#### 3. METEOR Score (Paraphrase Ability)        ####
#### 4. Response Time (Speed)                    ####
#### 5. Error Rate                               ####
#### 6. Toxicity (Safety)                        ####
#### 7. Cost Per Response                        ####
####                                             ####
#####################################################

# We keep the setup code in a different class as shown below.
from clsConfigClient import clsConfigClient as cf
from datetime import datetime

import json
from typing import List, Dict, Any
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from detoxify import Detoxify
import psutil
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import time
from anthropic import Anthropic
from openai import OpenAI
from transformers import pipeline

######################################################
########       Initializing the Class         ########
######################################################

@dataclass
class ModelResponse:
    content: str
    response_time: float
    token_count: int
    memory_usage: float
    error: str = None

openai_key = cf.conf['OPEN_AI_KEY']
anthropic_key = cf.conf['ANTHROPIC_AI_KEY']
stats_output = cf.conf['OUTPUT_PATH']

openai_model = cf.conf['MODEL_NAME_1']
anthropic_model = cf.conf['MODEL_NAME_2']
deepseek_model = cf.conf['MODEL_NAME_3']
bharatgpt_model = cf.conf['MODEL_NAME_4']

DEEPSEEK_API_URL = cf.conf['DEEPSEEK_URL']
dpseek_api_key = cf.conf['DEEPSEEK_AI_KEY']
sarvm_api_key = cf.conf['SARVAM_AI_KEY']
sarvam_api_url = cf.conf['SARVAM_URL']

maxToken = int(cf.conf['MAX_TOKEN'])

# Ensure that the MPS backend is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS backend is not available. Ensure you are running on Apple Silicon with macOS 12.3+.")

# Set device to MPS
device = torch.device("mps")

pipe = pipeline("text-generation",
                model=bharatgpt_model,
                torch_dtype=torch.float16,  # MPS supports float16
                device=device,  # Explicitly set the device to MPS
                )

######################################################
########    End of Initializing the Class     ########
######################################################

# Disbling Warning
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

#############################################
#########         Main Section    ###########
#############################################

class clsComprehensiveLLMEvaluator:
    def __init__(self, model_configs: Dict[str, Any]):
        self.anthropic_client = Anthropic(api_key=anthropic_key)
        self.openai_client = OpenAI(api_key=openai_key)
        self.deepseek_api_key = dpseek_api_key
        """Initialize evaluator with model configurations"""
        self.model_configs = model_configs
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.toxicity_model = Detoxify('original')
        self.initialize_logging()

    def initialize_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('llm_evaluation.log'),
                logging.StreamHandler()
            ]
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_claude_response(self, prompt: str) -> str:
        response = self.anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=maxToken,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_gpt4_response(self, prompt: str) -> str:
        response = self.openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=maxToken
        )
        return response.choices[0].message.content
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_deepseek_response(self, prompt: str) -> tuple:
        deepseek_api_key = self.deepseek_api_key

        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
            }
        
        payload = {
            "model": deepseek_model,  
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxToken
            }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            res = response.json()["choices"][0]["message"]["content"]
        else:
            res = "API request failed with status code " + str(response.status_code) + ":" + str(response.text)

        return res
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_bharatgpt_response(self, prompt: str) -> tuple:
        try:
            messages = [[{"role": "user", "content": prompt}]]
            
            response = pipe(messages, max_new_tokens=maxToken,)

            # Extract 'content' field safely
            res = next((entry.get("content", "")
                        for entry in response[0][0].get("generated_text", [])
                        if isinstance(entry, dict) and entry.get("role") == "assistant"
                        ),
                        None,
                        )
            
            return res
        except Exception as e:
            x = str(e)
            print('Error: ', x)

            return ""
        
    def get_model_response(self, model_name: str, prompt: str) -> ModelResponse:
        """Get response from specified model with metrics"""
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        try:
            if model_name == "claude-3":
                response_content = self.get_claude_response(prompt)
            elif model_name == "gpt4":
                response_content = self.get_gpt4_response(prompt)
            elif model_name == "deepseek-chat":
                response_content = self.get_deepseek_response(prompt)
            elif model_name == "bharat-gpt":
                response_content = self.get_bharatgpt_response(prompt)

            # Model-specific API calls 
            token_count = len(self.bert_tokenizer.encode(response_content))
            
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return ModelResponse(
                content=response_content,
                response_time=time.time() - start_time,
                token_count=token_count,
                memory_usage=memory_usage
            )
        except Exception as e:
            logging.error(f"Error getting response from {model_name}: {str(e)}")
            return ModelResponse(
                content="",
                response_time=0,
                token_count=0,
                memory_usage=0,
                error=str(e)
            )

    def evaluate_text_quality(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate text quality metrics"""
        # BERTScore
        gen_embedding = self.sentence_model.encode([generated])
        ref_embedding = self.sentence_model.encode([reference])
        bert_score = cosine_similarity(gen_embedding, ref_embedding)[0][0]

        # BLEU Score
        generated_tokens = word_tokenize(generated.lower())
        reference_tokens = word_tokenize(reference.lower())
        bleu = sentence_bleu([reference_tokens], generated_tokens)

        # METEOR Score
        meteor = meteor_score([reference_tokens], generated_tokens)

        return {
            'bert_score': bert_score,
            'bleu_score': bleu,
            'meteor_score': meteor
        }

    def evaluate_factual_accuracy(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate factual accuracy metrics"""
        # Implement fact checking using simple keyword matching or more sophisticated methods
        keyword_overlap = len(set(generated.lower().split()) & set(reference.lower().split())) / len(set(reference.lower().split()))
        
        return {
            'keyword_accuracy': keyword_overlap,
            'fact_verification_score': keyword_overlap  # Placeholder for more sophisticated fact checking
        }

    def evaluate_task_performance(self, generated: str, task_specific_criteria: Dict) -> Dict[str, float]:
        """Evaluate task-specific performance"""
        # Implement based on specific task requirements
        relevancy_score = cosine_similarity(
            self.sentence_model.encode([generated]),
            self.sentence_model.encode([task_specific_criteria.get('expected_content', '')])
        )[0][0]

        return {
            'relevancy_score': relevancy_score,
            'task_completion': 1.0 if relevancy_score > 0.7 else 0.0
        }

    def evaluate_technical_performance(self, response: ModelResponse) -> Dict[str, float]:
        """Evaluate technical performance metrics"""
        return {
            'response_time': response.response_time,
            'tokens_per_second': response.token_count / response.response_time if response.response_time > 0 else 0,
            'memory_usage_mb': response.memory_usage
        }

    def evaluate_reliability(self, responses: List[ModelResponse]) -> Dict[str, float]:
        """Evaluate reliability metrics"""
        error_rate = sum(1 for r in responses if r.error is not None) / len(responses)
        response_consistency = np.std([r.response_time for r in responses])
        
        return {
            'error_rate': error_rate,
            'response_consistency': response_consistency,
            'success_rate': 1 - error_rate
        }

    def evaluate_safety(self, generated: str) -> Dict[str, float]:
        """Evaluate safety and ethics metrics"""
        toxicity_scores = self.toxicity_model.predict(generated)
        
        return {
            'toxicity': toxicity_scores['toxicity'],
            'severe_toxicity': toxicity_scores['severe_toxicity'],
            'identity_attack': toxicity_scores['identity_attack'],
            'insult': toxicity_scores['insult']
        }

    def evaluate_business_impact(self, response: ModelResponse, task_success: float) -> Dict[str, float]:
        """Evaluate business impact metrics"""
        # Calculate estimated cost based on token count (customize based on your pricing)
        estimated_cost = response.token_count * 0.00001  # Example rate
        
        return {
            'cost_per_response': estimated_cost,
            'cost_per_successful_response': estimated_cost / task_success if task_success > 0 else float('inf'),
            'tokens_per_dollar': response.token_count / estimated_cost if estimated_cost > 0 else float('inf')
        }

    def run_comprehensive_evaluation(self, evaluation_data: List[Dict]) -> pd.DataFrame:
        """Run comprehensive evaluation on all metrics"""
        results = []
        
        for item in evaluation_data:
            prompt = item['prompt']
            reference = item['reference']
            task_criteria = item.get('task_criteria', {})
            
            for model_name in self.model_configs.keys():
                # Get multiple responses to evaluate reliability
                responses = [
                    self.get_model_response(model_name, prompt)
                    for _ in range(3)  # Get 3 responses for reliability testing
                ]
                
                # Use the best response for other evaluations
                best_response = max(responses, key=lambda x: len(x.content) if not x.error else 0)
                
                if best_response.error:
                    logging.error(f"Error in model {model_name}: {best_response.error}")
                    continue
                
                # Gather all metrics
                metrics = {
                    'model': model_name,
                    'prompt': prompt,
                    'response': best_response.content,
                    **self.evaluate_text_quality(best_response.content, reference),
                    **self.evaluate_factual_accuracy(best_response.content, reference),
                    **self.evaluate_task_performance(best_response.content, task_criteria),
                    **self.evaluate_technical_performance(best_response),
                    **self.evaluate_reliability(responses),
                    **self.evaluate_safety(best_response.content)
                }
                
                # Add business impact metrics using task performance
                metrics.update(self.evaluate_business_impact(
                    best_response,
                    metrics['task_completion']
                ))
                
                results.append(metrics)
        
        return pd.DataFrame(results)

