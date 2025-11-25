"""
Benchmark closed models via OpenRouter API
Supports multiple models with the same data-1.csv dataset
"""

import pandas as pd
import json
import requests
import time
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List, Optional, Tuple
import argparse

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model configurations
OPENROUTER_MODELS = {
    'grok-4': {
        'model_id': 'x-ai/grok-4',
        'max_tokens': 50,
        'temperature': 0.0,
        'supports_system': True,
        'reasoning': {'enabled': False}  # Disable reasoning/thinking mode
    },
    'grok-4-fast': {
        'model_id': 'x-ai/grok-4-fast',
        'max_tokens': 50,
        'temperature': 0.0,
        'supports_system': True,
        'reasoning': {'enabled': False}  # Disable reasoning/thinking mode
    },
    'gpt-5': {
        'model_id': 'openai/gpt-5',
        'max_tokens': 100,  # GPT-5 on Azure requires minimum 16 tokens
        'temperature': 0.0,
        'supports_system': True,
        'reasoning_effort': 'minimal'  # Disable deep reasoning for faster responses
    },
    'claude-haiku': {
        'model_id': 'anthropic/claude-haiku-4.5',
        'max_tokens': 50,
        'temperature': 0.0,
        'supports_system': True
    },
    'gemini-2.0-flash': {
        'model_id': 'google/gemini-2.0-flash-exp:free',
        'max_tokens': 50,
        'temperature': 0.0,
        'supports_system': True
    },
    'gemini-2.5-pro': {
        'model_id': 'google/gemini-2.5-pro',
        'max_tokens': 50,
        'temperature': 0.0,
        'supports_system': True
    }
}

def create_prompt(row):
    """Create MCQ prompt from dataset row (same as benchmark_models_fixed.py)"""
    language = row.get('language', row.get('subject', 'Unknown'))
    
    prompt = f"""Question: {row['question_text']}

Options:
A) {row['option_a']}
B) {row['option_b']}
C) {row['option_c']}
D) {row['option_d']}

The above question is written in {language} language. Please analyze the question and options carefully, and select the correct answer. Respond ONLY with one letter (A, B, C, or D) corresponding to the correct option. Do not provide any explanation or additional text."""
    
    return prompt

def make_openrouter_request(messages: List[Dict], model_config: Dict, retry_count: int = 3) -> Dict:
    """Make a request to OpenRouter API with retry logic"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://rare-bench.ai",  # Optional
        "X-Title": "RARE Benchmark"  # Optional
    }
    
    data = {
        "model": model_config['model_id'],
        "messages": messages,
        "max_tokens": model_config.get('max_tokens', 50),
        "temperature": model_config.get('temperature', 0.0),
        "top_p": model_config.get('top_p', 1.0),
        "stream": False
    }
    
    # Add reasoning parameter for models that support it (like Grok 4)
    if 'reasoning' in model_config:
        data['reasoning'] = model_config['reasoning']
    
    # Add reasoning_effort parameter for models like GPT-5
    if 'reasoning_effort' in model_config:
        data['reasoning_effort'] = model_config['reasoning_effort']
    
    for attempt in range(retry_count):
        try:
            response = requests.post(
                url=OPENROUTER_BASE_URL,
                headers=headers,
                data=json.dumps(data),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                wait_time = int(response.headers.get('Retry-After', 5))
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 404 and "data policy" in response.text:
                print("\n❌ OpenRouter Data Policy Configuration Required!")
                print("Please visit: https://openrouter.ai/settings/privacy")
                print("And configure your data privacy settings to allow 'Free model publication'")
                print("\nThis is required to use OpenRouter's API.")
                return None
            else:
                print(f"Error {response.status_code}: {response.text}")
                # Check if this is a model availability issue
                if response.status_code == 400 and "model" in response.text.lower():
                    print(f"\n⚠️ Model availability issue. Response: {response.text}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
                continue
    
    return None

def extract_answer(response_text: str) -> Tuple[Optional[str], str]:
    """Extract answer letter (A, B, C, or D) from model response"""
    if not response_text or not isinstance(response_text, str):
        return None, "Empty response"
        
    response = response_text.upper().strip()
    
    # Look for isolated letters first (best match)
    isolated_match = re.search(r'\b([ABCD])\b', response)
    if isolated_match:
        return isolated_match.group(1), "Found answer"
    
    # Look for letters at start of line or after specific patterns
    pattern_match = re.search(r'(?:^|answer[:\s]*|choice[:\s]*|option[:\s]*)([ABCD])(?:\)|\.|\s|$)', response, re.IGNORECASE)
    if pattern_match:
        return pattern_match.group(1).upper(), "Found answer"
    
    # Fallback: first occurrence of any letter
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            return letter, "Found answer"
    
    return None, "No valid answer found"

def save_checkpoint(results: Dict, model_name: str, checkpoint_dir: str):
    """Save checkpoint file for resuming interrupted runs"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{model_name.replace('/', '_')}_openrouter_checkpoint.json")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_checkpoint(model_name: str, checkpoint_dir: str) -> Dict:
    """Load checkpoint file if it exists"""
    checkpoint_file = os.path.join(checkpoint_dir, f"{model_name.replace('/', '_')}_openrouter_checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}

def benchmark_openrouter_model(model_name: str, df: pd.DataFrame, checkpoint_dir: str = "checkpoints"):
    """Benchmark a model via OpenRouter API"""
    if model_name not in OPENROUTER_MODELS:
        print(f"Error: Model {model_name} not configured")
        return None
    
    model_config = OPENROUTER_MODELS[model_name]
    model_id = model_config['model_id']
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_id} via OpenRouter")
    print(f"{'='*60}")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(model_name, checkpoint_dir)
    processed_ids = set(checkpoint.get('processed_questions', []))
    
    if len(processed_ids) >= len(df):
        print(f"Model {model_name} already completed! ({len(processed_ids)} questions)")
        return checkpoint
    
    # Initialize results (matching benchmark_models_fixed.py structure)
    results = {
        'model_name': model_name,
        'processed_questions': list(processed_ids),
        'responses': checkpoint.get('responses', {}),
        'timestamps': checkpoint.get('timestamps', {}),
        'generation_stats': checkpoint.get('generation_stats', {}),
        'extraction_debug': checkpoint.get('extraction_debug', {}),
        'batch_processing_info': {
            'batch_size': 1,  # OpenRouter processes one at a time
            'total_questions': len(df),
            'processing_start': datetime.now().isoformat()
        }
    }
    
    # Filter unprocessed questions
    unprocessed_df = df[~df['unique_question_id'].isin(processed_ids)]
    if len(unprocessed_df) == 0:
        print("All questions already processed!")
        return results
    
    print(f"Processing {len(unprocessed_df)} remaining questions...")
    
    # Process questions
    correct_count = 0
    total_count = len(processed_ids)
    api_errors = 0
    
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing"):
        question_id = row['unique_question_id']
        
        # Create prompt
        prompt = create_prompt(row)
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if supported
        if model_config.get('supports_system', True):
            messages.insert(0, {
                "role": "system", 
                "content": "You are a helpful assistant that answers multiple choice questions accurately."
            })
        
        # Make API request
        start_time = time.time()
        try:
            response_data = make_openrouter_request(messages, model_config)
            elapsed_time = time.time() - start_time
            
            if response_data and 'choices' in response_data:
                # Extract content from response
                response_text = response_data['choices'][0]['message']['content']
                
                # Debug logging for empty responses
                if not response_text:
                    print(f"\n⚠️ Empty content in response for question {question_id}")
                    if 'error' in response_data['choices'][0]:
                        print(f"Error in response: {response_data['choices'][0]['error']}")
                
                predicted_answer, extraction_info = extract_answer(response_text)
                
                # Calculate if correct
                is_correct = predicted_answer == row['correct_answer'].upper() if predicted_answer else False
                if is_correct:
                    correct_count += 1
                
                # Store results (matching benchmark_models_fixed.py structure)
                results['responses'][question_id] = {
                    'raw_response': response_text,
                    'full_response': response_text,  # For API calls, these are the same
                    'predicted_answer': predicted_answer,
                    'correct_answer': row['correct_answer'],
                    'subject': row['subject'],
                    'language': row.get('language', row.get('subject', 'Unknown')),
                    'exam_name': row['exam_name'],
                    'paper_number': row['paper_number'],
                    'is_correct': is_correct
                }
                
                # Store generation statistics
                prompt_tokens = 0
                completion_tokens = 0
                if 'usage' in response_data:
                    prompt_tokens = response_data['usage'].get('prompt_tokens', 0)
                    completion_tokens = response_data['usage'].get('completion_tokens', 0)
                
                results['generation_stats'][question_id] = {
                    'input_length': prompt_tokens,
                    'output_length': completion_tokens,
                    'generated_tokens': completion_tokens,
                    'has_error': False
                }
                
                # Store extraction debugging info
                results['extraction_debug'][question_id] = {
                    'extraction_info': extraction_info,
                    'raw_response_length': len(response_text),
                    'is_empty_response': len(response_text.strip()) == 0
                }
            else:
                # API error
                api_errors += 1
                results['responses'][question_id] = {
                    'raw_response': "",
                    'full_response': "",
                    'predicted_answer': None,
                    'correct_answer': row['correct_answer'],
                    'subject': row['subject'],
                    'language': row.get('language', row.get('subject', 'Unknown')),
                    'exam_name': row['exam_name'],
                    'paper_number': row['paper_number'],
                    'is_correct': False
                }
                
                results['generation_stats'][question_id] = {
                    'input_length': 0,
                    'output_length': 0,
                    'generated_tokens': 0,
                    'has_error': True
                }
                
                results['extraction_debug'][question_id] = {
                    'extraction_info': "API request failed",
                    'raw_response_length': 0,
                    'is_empty_response': True
                }
            
            results['timestamps'][question_id] = elapsed_time
            results['processed_questions'].append(question_id)
            total_count += 1
            
            # Save checkpoint every 50 questions
            if len(results['processed_questions']) % 50 == 0:
                save_checkpoint(results, model_name, checkpoint_dir)
                current_accuracy = correct_count / total_count if total_count > 0 else 0
                print(f"\nCheckpoint saved. Progress: {total_count}/{len(df)} | Accuracy: {current_accuracy:.3f}")
            
            # Rate limiting - adjust based on model/tier
            time.sleep(0.5)  # 0.5 second delay between requests
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            api_errors += 1
            results['responses'][question_id] = {
                'raw_response': "",
                'full_response': "",
                'predicted_answer': None,
                'correct_answer': row['correct_answer'],
                'subject': row['subject'],
                'language': row.get('language', row.get('subject', 'Unknown')),
                'exam_name': row['exam_name'],
                'paper_number': row['paper_number'],
                'is_correct': False
            }
            
            results['generation_stats'][question_id] = {
                'input_length': 0,
                'output_length': 0,
                'generated_tokens': 0,
                'has_error': True
            }
            
            results['extraction_debug'][question_id] = {
                'extraction_info': str(e),
                'raw_response_length': 0,
                'is_empty_response': True
            }
            
            results['timestamps'][question_id] = 0
            results['processed_questions'].append(question_id)
            total_count += 1
    
    # Calculate final metrics (matching benchmark_models_fixed.py structure)
    processing_end = datetime.now()
    processing_start = datetime.fromisoformat(results['batch_processing_info']['processing_start'])
    total_processing_time = (processing_end - processing_start).total_seconds()
    
    # Count problematic responses
    problematic_responses = sum(1 for r in results['responses'].values() 
                               if not r['predicted_answer'] or not r['raw_response'].strip())
    
    results['batch_processing_info'].update({
        'processing_end': processing_end.isoformat(),
        'total_processing_time': total_processing_time,
        'avg_time_per_question': total_processing_time / total_count if total_count > 0 else 0,
        'problematic_responses': problematic_responses,
        'success_rate': (total_count - problematic_responses) / total_count if total_count > 0 else 0
    })
    
    # These fields are kept for backward compatibility but not in benchmark_models_fixed.py
    results['accuracy'] = correct_count / total_count if total_count > 0 else 0
    results['correct_count'] = correct_count
    results['total_count'] = total_count
    
    # Save final results
    save_checkpoint(results, model_name, checkpoint_dir)
    
    print(f"\nResults for {model_id}:")
    print(f"  Total questions: {total_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  API errors: {api_errors}")
    if 'total_api_tokens' in results:
        print(f"  Total tokens used: {results['total_api_tokens']:,}")
    
    return results

def generate_openrouter_report(all_results: Dict, output_dir: str):
    """Generate report for OpenRouter benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary
    summary = f"""# OpenRouter API Benchmark Results
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models Tested
"""
    
    # Sort by accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
    
    for i, (model_name, results) in enumerate(sorted_results, 1):
        if results:
            # Get model_id from the configuration
            model_config = OPENROUTER_MODELS.get(model_name, {})
            model_id = model_config.get('model_id', model_name)
            
            summary += f"\n### {i}. {model_id}\n"
            summary += f"- Accuracy: {results['accuracy']:.3f}\n"
            summary += f"- Total Questions: {results['total_count']}\n"
            summary += f"- Correct Answers: {results['correct_count']}\n"
            summary += f"- API Errors: {results.get('api_errors', 0)}\n"
            if 'total_api_tokens' in results:
                summary += f"- Total Tokens Used: {results['total_api_tokens']:,}\n"
    
    # Save summary
    with open(os.path.join(output_dir, 'openrouter_summary.md'), 'w') as f:
        f.write(summary)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'openrouter_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark models via OpenRouter API')
    parser.add_argument('--models', nargs='+', default=['gpt-5'],
                        choices=list(OPENROUTER_MODELS.keys()),
                        help='Models to benchmark')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing data-1.csv')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions for testing')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    args = parser.parse_args()
    
    # Set API key if provided
    global OPENROUTER_API_KEY
    if args.api_key:
        OPENROUTER_API_KEY = args.api_key
    
    # Check API key
    if OPENROUTER_API_KEY == "YOUR_API_KEY_HERE":
        print("Error: Please set your OpenRouter API key!")
        print("Either set OPENROUTER_API_KEY environment variable or use --api-key argument")
        return
    
    print("\n📋 OpenRouter Setup Requirements:")
    print("1. ✓ API Key configured")
    print("2. ⚠️  Make sure you've configured data privacy settings at:")
    print("   https://openrouter.ai/settings/privacy")
    print("   - Enable 'Allow free model publication'")
    print("   - This allows your prompts/responses to be used by model providers")
    print("\nStarting benchmark in 3 seconds...")
    time.sleep(3)
    
    
    # Load dataset
    data_path = Path(args.data_dir) / "data-3.csv"
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return
    
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Add language column if not present
    if 'language' not in df.columns:
        df['language'] = df['subject']
    
    print(f"Loaded {len(df)} questions")
    
    # Limit dataset if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} questions for testing")
    
    # Run benchmarks
    all_results = {}
    checkpoint_dir = "checkpoints-3"
    
    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*70}")
        
        results = benchmark_openrouter_model(model_name, df, checkpoint_dir)
        if results:
            all_results[model_name] = results
    
    # Generate report
    generate_openrouter_report(all_results, "openrouter_results")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")
    for model_name, results in all_results.items():
        if results:
            # Get model_id from the configuration
            model_config = OPENROUTER_MODELS.get(model_name, {})
            model_id = model_config.get('model_id', model_name)
            print(f"{model_id:<50} | Accuracy: {results['accuracy']:.3f}")

if __name__ == "__main__":
    main()
