#####################################################
####                                             ####
#### Written By: SATYAKI DE                      ####
#### Written On: 01-Jan-2025                     ####
#### Modified On 05-Jan-2025                     ####
####                                             ####
#### Objective: This is the main calling         ####
#### python script that will invoke the          ####
#### clsComprehensiveLLMEvaluator custom class   ####
#### to perform seven critical performance KPIs  ####
#### of selected LLMs.                           ####
####                                             ####
#####################################################

# We keep the setup code in a different class as shown below.
from clsConfigClient import clsConfigClient as cf
from datetime import datetime

import clsComprehensiveLLMEvaluator as cclme

######################################################
########       Initializing the Class         ########
######################################################

openai_key = cf.conf['OPEN_AI_KEY']
anthropic_key = cf.conf['ANTHROPIC_AI_KEY']
dpseek_api_key = cf.conf['DEEPSEEK_AI_KEY']
stats_output = cf.conf['OUTPUT_PATH']

# Model configurations
model_configs = {
    'claude-3': {'api_key': anthropic_key},
    'gpt4': {'api_key': openai_key},
    'deepseek-chat': {'api_key': dpseek_api_key},
    'bharat-gpt': {'api_key': 'dummy'}
    }

# Instantiate the classes
r1 = cclme.clsComprehensiveLLMEvaluator(model_configs)

######################################################
########    End of   Initializing the Class   ########
######################################################

# Disbling Warning
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

#############################################
#########         Main Section    ###########
#############################################

def main():
    print('Starting overall comprehensive LLM Evaluation.')
    print()
    print("Following are the areas where we'll evaluate our LLMs:")
    print('-'*120)
    print("1. BERT Score (Understanding & Relevance)")
    print("2. BLEU Score (Generation Quality)")
    print("3. METEOR Score (Paraphrase Ability)")
    print("4. Response Time (Speed)")
    print("5. Error Rate")
    print("6. Toxicity (Safety)")
    print("7. Cost Per Response")
    print('-'*120)
    print()

    var = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('*'*120)
    print('Start Time: ' + str(var))
    print('*'*120)
    
    # Sample evaluation data
    evaluation_data = [
        {
            "prompt": "Explain how photosynthesis works in simple terms.",
            "reference": "Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        },
        {
            "prompt": "What are the main causes of climate change?",
            "reference": "The main causes of climate change are greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes, which trap heat in Earth's atmosphere."
        }
    ]
    
    # Run evaluation
    results_df = r1.run_comprehensive_evaluation(evaluation_data)
    
    # Calculate and display average metrics per model
    summary = results_df.groupby('model').agg({
        'bert_score': 'mean',
        'bleu_score': 'mean',
        'meteor_score': 'mean',
        'response_time': 'mean',
        'error_rate': 'mean',
        'toxicity': 'mean',
        'cost_per_response': 'mean'
    }).round(4)
    
    # Save results
    tmpstmp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_res = stats_output + 'llm_evaluation_results_' + str(tmpstmp) + '.csv'
    llm_sum = stats_output + 'llm_evaluation_summary_' + str(tmpstmp) + '.csv'
    results_df.to_csv(llm_res, index=False)
    summary.to_csv(llm_sum)
    
    print("\nEvaluation Summary:")
    print(summary)

    var_1 = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print('*'*120)
    print('End Time: ' + str(var_1))
    print('*'*120)
    print('Finished evaluation successfully!')
    
if __name__ == "__main__":
    main()
