import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import os
from datetime import datetime
import gc
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import re
import argparse

# Set cache directory for Hugging Face models
CACHE_DIR = "/userhome/home/aymaheshwari/hf-cache/hub"

MODEL_GROUPS = {
    'small': [
        # 'bharatgenai/Param-1-2.9B-Instruct',  
        # 'meta-llama/Llama-3.2-1B-Instruct',  
        # 'sarvamai/sarvam-1',  
        # 'google/gemma-3-1b-it',  
        # 'google/gemma-3-4b-it',  
        'meta-llama/Llama-3.2-3B-Instruct',  
        'Qwen/Qwen2.5-3B-Instruct',  
        'Qwen/Qwen3-4B-Instruct-2507'  
    ],
    'medium': [
        # 'google/gemma-3-4b-it',
        'mistralai/Mistral-Small-3.1-24B-Instruct-2503',  
        # 'sarvamai/sarvam-m',  
        'CohereForAI/aya-expanse-8b',  
        'meta-llama/Llama-3.1-8B-Instruct',  
        # 'google/gemma-3-27b-it'  
        'Qwen/Qwen3-32B',
    ],
    'large': [
        'meta-llama/Llama-3.3-70B-Instruct',
        'Qwen/Qwen3-30B-A3B',
        'CohereLabs/aya-expanse-32b',
    ]
}

def load_subject_data(data_dir):
    """Load BharatQA dataset from CSV file"""
    data_path = Path(data_dir)
    # csv_file = data_path / "data-1.csv"
    csv_file = data_path / "data-1.csv"

    
    if csv_file.exists():
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        # The subject column contains the language name
        # Add a language column if not present
        if 'language' not in df.columns:
            df['language'] = df['subject']
        return df
    else:
        print(f"Error: data-1.csv not found in {data_dir}")
        return pd.DataFrame()

def create_prompt(row):
    """Create MCQ prompt from dataset row"""
    # Get language information
    language = row.get('language', row.get('subject', 'Unknown'))
    
    prompt = f"""
Question: {row['question_text']}

Options:
A) {row['option_a']}
B) {row['option_b']}
C) {row['option_c']}
D) {row['option_d']}

The above question is written in {language} language. Please analyze the question and options carefully, and select the correct answer. Respond ONLY with one letter (A, B, C, or D) corresponding to the correct option. Do not provide any explanation or additional text."""
    return prompt

def create_harmony_messages(row):
    """Create messages in harmony format for gpt-oss models"""
    content = create_prompt(row)
    return [{"role": "user", "content": content}]

def load_model(model_name):
    """Load model and tokenizer with appropriate configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_name}")
    
    # Special handling for gpt-oss models that use pipeline
    if "gpt-oss" in model_name:
        print("Using pipeline for gpt-oss model")
        try:
            pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            print("✅ Pipeline loaded successfully")
            return pipe, None, device
        except Exception as e:
            print(f"Failed to load pipeline: {str(e)}")
            # Continue with standard loading
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with appropriate dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"Failed with auto dtype, trying bfloat16: {e}")
        # Fallback to bfloat16 if auto fails
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        )
    
    return model, tokenizer, device

def generate_batch_responses_pipeline(pipe, messages_list, batch_size=8):
    """Generate responses using pipeline (for gpt-oss models)"""
    all_responses = []
    for i in tqdm(range(0, len(messages_list), batch_size), desc="Processing pipeline batches"):
        batch_messages = messages_list[i:i + batch_size]
        try:
            batch_responses = []
            for messages in batch_messages:
                outputs = pipe(
                    messages,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False,
                    return_full_text=False
                )
                
                if outputs and len(outputs) > 0:
                    response_text = outputs[0]["generated_text"]
                    response_data = {
                        'full_response': response_text,
                        'new_tokens_only': response_text.strip(),
                        'input_length': len(messages[0]["content"]),
                        'output_length': len(response_text),
                        'generated_tokens': len(response_text.split())
                    }
                else:
                    response_data = {
                        'full_response': "",
                        'new_tokens_only': "",
                        'input_length': 0,
                        'output_length': 0,
                        'generated_tokens': 0
                    }
                batch_responses.append(response_data)
            
            all_responses.extend(batch_responses)
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing pipeline batch {i//batch_size + 1}: {e}")
            error_responses = [{
                'full_response': "",
                'new_tokens_only': "",
                'input_length': 0,
                'output_length': 0,
                'generated_tokens': 0,
                'error': str(e)
            } for _ in batch_messages]
            all_responses.extend(error_responses)
    
    return all_responses

def generate_batch_responses(model, tokenizer, prompts, device, batch_size=8, model_name=None):
    """Generate responses for batches of prompts"""
    all_responses = []
    
    # Special handling for Param-1 model
    use_cache = True
    if model_name and "Param-1" in model_name:
        print("Using Param-1 with use_cache=False")
        use_cache = False
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        try:
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                    return_dict_in_generate=True,
                    output_scores=False 
                )
            
            batch_responses = []
            for j in range(len(outputs.sequences)):
                try:
                    output_ids = outputs.sequences[j]
                    input_length = inputs['input_ids'][j].shape[0]
                    
                    full_response = tokenizer.decode(output_ids, skip_special_tokens=True)
                    new_tokens = output_ids[input_length:]
                    new_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    response_data = {
                        'full_response': full_response,
                        'new_tokens_only': new_response.strip(),
                        'input_length': input_length,
                        'output_length': len(output_ids),
                        'generated_tokens': len(new_tokens)
                    }
                    batch_responses.append(response_data)
                except Exception as e:
                    print(f"Error processing sequence {j}: {e}")
                    batch_responses.append({
                        'full_response': "",
                        'new_tokens_only': "",
                        'input_length': 0,
                        'output_length': 0,
                        'generated_tokens': 0
                    })
            
            all_responses.extend(batch_responses)
            
            # Clear cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            error_responses = [{
                'full_response': "",
                'new_tokens_only': "",
                'input_length': 0,
                'output_length': 0,
                'generated_tokens': 0,
                'error': str(e)
            } for _ in batch_prompts]
            all_responses.extend(error_responses)
    
    return all_responses

def extract_answer(response_text):
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

def save_checkpoint(results, model_name, checkpoint_dir):
    """Save checkpoint file for resuming interrupted runs"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{model_name.replace('/', '_')}_checkpoint.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_checkpoint(model_name, checkpoint_dir):
    """Load checkpoint file if it exists"""
    checkpoint_file = os.path.join(checkpoint_dir, f"{model_name.replace('/', '_')}_checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}

def benchmark_model_batch(model_name, df, checkpoint_dir, batch_size=8):
    """Benchmark a model on the dataset with batch processing and checkpointing"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name} (batch_size={batch_size})")
    print(f"{'='*60}")
    
    checkpoint = load_checkpoint(model_name, checkpoint_dir)
    processed_ids = set(checkpoint.get('processed_questions', []))
    
    if len(processed_ids) >= len(df):
        print(f"Model {model_name} already completed! ({len(processed_ids)} questions)")
        return checkpoint
    
    try:
        # Load model once for all questions
        model_or_pipe, tokenizer, device = load_model(model_name)
        
        # Initialize results structure
        results = {
            'model_name': model_name,
            'processed_questions': list(processed_ids),
            'responses': checkpoint.get('responses', {}),
            'timestamps': checkpoint.get('timestamps', {}),
            'generation_stats': checkpoint.get('generation_stats', {}),
            'extraction_debug': checkpoint.get('extraction_debug', {}),
            'batch_processing_info': {
                'batch_size': batch_size,
                'total_questions': len(df),
                'processing_start': datetime.now().isoformat()
            }
        }
        
        unprocessed_df = df[~df['unique_question_id'].isin(processed_ids)]
        if len(unprocessed_df) == 0:
            print("All questions already processed!")
            return results
        
        print(f"Processing {len(unprocessed_df)} remaining questions...")
        
        # Determine if we're using pipeline (gpt-oss) or traditional approach
        is_pipeline = tokenizer is None
        
        if is_pipeline:
            print("Using pipeline approach")
            # Prepare messages for pipeline
            messages_list = []
            question_metadata = []
            for idx, row in unprocessed_df.iterrows():
                messages_list.append(create_harmony_messages(row))
                question_metadata.append({
                    'question_id': row['unique_question_id'],
                    'correct_answer': row['correct_answer'],
                    'subject': row['subject'],
                    'language': row.get('language', row.get('subject', 'Unknown')),
                    'exam_name': row['exam_name'],
                    'paper_number': row['paper_number'],
                    'row_index': idx
                })
            start_time = time.time()
            all_responses = generate_batch_responses_pipeline(model_or_pipe, messages_list, batch_size)
            total_time = time.time() - start_time
        else:
            print("Using traditional model/tokenizer approach")
            # Check if this is Qwen3-30B-A3B model
            is_qwen3_30b = "Qwen3-30B-A3B" or "Qwen3-32B" in model_name
            
            if is_qwen3_30b:
                print("✓ Using Qwen3-30B-A3B with thinking mode disabled (enable_thinking=False)")
            
            # Prepare prompts for traditional approach
            prompts = []
            question_metadata = []
            
            for idx, row in unprocessed_df.iterrows():
                prompt_text = create_prompt(row)
                
                # For Qwen3-30B-A3B, use apply_chat_template with enable_thinking=False
                if is_qwen3_30b:
                    messages = [{"role": "user", "content": prompt_text}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False  # Disable thinking mode for Qwen3-30B-A3B
                    )
                    prompts.append(formatted_prompt)
                else:
                    prompts.append(prompt_text)
                    
                question_metadata.append({
                    'question_id': row['unique_question_id'],
                    'correct_answer': row['correct_answer'],
                    'subject': row['subject'],
                    'language': row.get('language', row.get('subject', 'Unknown')),
                    'exam_name': row['exam_name'],
                    'paper_number': row['paper_number'],
                    'row_index': idx
                })
            
            start_time = time.time()
            all_responses = generate_batch_responses(model_or_pipe, tokenizer, prompts, device, batch_size, model_name)
            total_time = time.time() - start_time
        
        print(f"Batch processing completed in {total_time:.2f} seconds")
        num_questions = len(messages_list) if is_pipeline else len(prompts)
        print(f"Average time per question: {total_time/num_questions:.3f} seconds")
        
        print("Extracting answers and saving results...")
        problematic_responses = 0
        
        for i, (response_data, metadata) in enumerate(zip(all_responses, question_metadata)):
            question_id = metadata['question_id']
            
            # Extract answer with debugging
            predicted_answer, extraction_info = extract_answer(response_data['new_tokens_only'])
            
            # Store comprehensive results
            results['responses'][question_id] = {
                'raw_response': response_data['new_tokens_only'],
                'full_response': response_data['full_response'],
                'predicted_answer': predicted_answer,
                'correct_answer': metadata['correct_answer'],
                'subject': metadata['subject'],
                'language': metadata['language'],
                'exam_name': metadata['exam_name'],
                'paper_number': metadata['paper_number'],
                'is_correct': predicted_answer.upper() == metadata['correct_answer'].upper() if predicted_answer and metadata['correct_answer'] else False
            }
            
            # Store generation statistics
            results['generation_stats'][question_id] = {
                'input_length': response_data['input_length'],
                'output_length': response_data['output_length'],
                'generated_tokens': response_data['generated_tokens'],
                'has_error': 'error' in response_data
            }
            
            # Store extraction debugging info
            results['extraction_debug'][question_id] = {
                'extraction_info': extraction_info,
                'raw_response_length': len(response_data['new_tokens_only']),
                'is_empty_response': len(response_data['new_tokens_only'].strip()) == 0
            }
            
            results['timestamps'][question_id] = total_time / num_questions
            results['processed_questions'].append(question_id)
            
            if not predicted_answer or len(response_data['new_tokens_only'].strip()) == 0:
                problematic_responses += 1
        
        results['batch_processing_info'].update({
            'processing_end': datetime.now().isoformat(),
            'total_processing_time': total_time,
            'avg_time_per_question': total_time / num_questions,
            'problematic_responses': problematic_responses,
            'success_rate': (num_questions - problematic_responses) / num_questions
        })
        
        # Save final checkpoint
        save_checkpoint(results, model_name, checkpoint_dir)
        
        print(f"Processing completed:")
        print(f"  Total questions: {num_questions}")
        print(f"  Problematic responses: {problematic_responses}")
        print(f"  Success rate: {((num_questions - problematic_responses) / num_questions * 100):.1f}%")
        
        # Clean up GPU memory
        if is_pipeline:
            del model_or_pipe
        else:
            del model_or_pipe, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    except Exception as e:
        print(f"Error with model {model_name}: {str(e)}")
        return None

def calculate_metrics(results):
    """Calculate comprehensive metrics from results"""
    if not results or 'responses' not in results:
        return {}
    
    responses = results['responses']
    total_questions = len(responses)
    correct_answers = sum(1 for r in responses.values() if r['is_correct'])
    
    # Calculate additional metrics
    null_predictions = sum(1 for r in responses.values() if r['predicted_answer'] is None)
    empty_responses = sum(1 for r in responses.values() if not r['raw_response'].strip())
    
    metrics = {
        'accuracy': correct_answers / total_questions if total_questions > 0 else 0,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'null_predictions': null_predictions,
        'empty_responses': empty_responses,
        'null_prediction_rate': null_predictions / total_questions if total_questions > 0 else 0,
        'empty_response_rate': empty_responses / total_questions if total_questions > 0 else 0,
        'avg_response_time': np.mean(list(results['timestamps'].values())) if results['timestamps'] else 0
    }
    
    if 'batch_processing_info' in results:
        batch_info = results['batch_processing_info']
        metrics['batch_size'] = batch_info.get('batch_size', 'unknown')
        metrics['total_processing_time'] = batch_info.get('total_processing_time', 0)
        metrics['batch_success_rate'] = batch_info.get('success_rate', 0)
    
    if 'generation_stats' in results:
        gen_stats = list(results['generation_stats'].values())
        metrics['avg_generated_tokens'] = np.mean([s['generated_tokens'] for s in gen_stats])
        metrics['avg_input_length'] = np.mean([s['input_length'] for s in gen_stats])
        metrics['generation_errors'] = sum(1 for s in gen_stats if s.get('has_error', False))
    
    # Calculate subject, language and exam-wise metrics
    subject_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'null': 0})
    language_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'null': 0})
    exam_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'null': 0})
    
    for response in responses.values():
        subject = response['subject']
        language = response.get('language', subject)  # Fallback to subject if language not present
        exam_name = response['exam_name']
        is_correct = response['is_correct']
        is_null = response['predicted_answer'] is None
        
        subject_metrics[subject]['total'] += 1
        language_metrics[language]['total'] += 1
        exam_metrics[exam_name]['total'] += 1
        
        if is_correct:
            subject_metrics[subject]['correct'] += 1
            language_metrics[language]['correct'] += 1
            exam_metrics[exam_name]['correct'] += 1
        
        if is_null:
            subject_metrics[subject]['null'] += 1
            language_metrics[language]['null'] += 1
            exam_metrics[exam_name]['null'] += 1
    
    metrics['subject_accuracy'] = {
        subject: data['correct'] / data['total'] if data['total'] > 0 else 0
        for subject, data in subject_metrics.items()
    }
    
    metrics['subject_null_rate'] = {
        subject: data['null'] / data['total'] if data['total'] > 0 else 0
        for subject, data in subject_metrics.items()
    }
    
    metrics['language_accuracy'] = {
        language: data['correct'] / data['total'] if data['total'] > 0 else 0
        for language, data in language_metrics.items()
    }
    
    metrics['language_null_rate'] = {
        language: data['null'] / data['total'] if data['total'] > 0 else 0
        for language, data in language_metrics.items()
    }
    
    metrics['exam_accuracy'] = {
        exam: data['correct'] / data['total'] if data['total'] > 0 else 0
        for exam, data in exam_metrics.items()
    }
    
    return metrics

def run_benchmark_group_batch(group_name, df, checkpoint_dir, batch_size=8):
    """Run benchmarking for a group of models"""
    print(f"\n{'='*70}")
    print(f"Running {group_name.upper()} model group")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}")
    
    models = MODEL_GROUPS[group_name]
    group_results = {}
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing {model_name}")
        try:
            # Adjust batch size for large models
            if 'large' in group_name or any(size in model_name for size in ['70B', '405B', '30B', '35B', '20b', '120b']):
                model_batch_size = min(batch_size, 16)
            elif 'medium' in group_name:
                model_batch_size = min(batch_size, 16)
            else:
                model_batch_size = batch_size
            
            print(f"Using batch size: {model_batch_size}")
            
            results = benchmark_model_batch(model_name, df, checkpoint_dir, model_batch_size)
            if results:
                group_results[model_name] = calculate_metrics(results)
                # Print summary for this model
                metrics = group_results[model_name]
                print(f"\n✅ {model_name} Results:")
                print(f"   Accuracy: {metrics['accuracy']:.3f}")
                print(f"   Questions processed: {metrics['total_questions']}")
                print(f"   Null predictions: {metrics['null_predictions']} ({metrics['null_prediction_rate']:.3f})")
                print(f"   Batch processing time: {metrics.get('total_processing_time', 0):.1f}s")
                print(f"   Success rate: {metrics.get('batch_success_rate', 0):.3f}")
        except Exception as e:
            print(f"❌ Failed to benchmark {model_name}: {str(e)}")
            continue
    
    return group_results

def generate_batch_report(all_results, output_dir):
    """Generate enhanced report including batch processing metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_data = []
    for model_name, metrics in all_results.items():
        if metrics:
            report_data.append({
                'model': model_name,
                'overall_accuracy': metrics['accuracy'],
                'total_questions': metrics['total_questions'],
                'null_predictions': metrics['null_predictions'],
                'empty_responses': metrics['empty_responses'],
                'null_prediction_rate': metrics['null_prediction_rate'],
                'empty_response_rate': metrics['empty_response_rate'],
                'avg_response_time': metrics['avg_response_time'],
                'batch_size': metrics.get('batch_size', 'unknown'),
                'total_processing_time': metrics.get('total_processing_time', 0),
                'batch_success_rate': metrics.get('batch_success_rate', 0),
                'avg_generated_tokens': metrics.get('avg_generated_tokens', 0),
                'generation_errors': metrics.get('generation_errors', 0),
                **{f'subject_{k}': v for k, v in metrics.get('subject_accuracy', {}).items()},
                **{f'subject_null_{k}': v for k, v in metrics.get('subject_null_rate', {}).items()},
                **{f'language_{k}': v for k, v in metrics.get('language_accuracy', {}).items()},
                **{f'language_null_{k}': v for k, v in metrics.get('language_null_rate', {}).items()},
                **{f'exam_{k}': v for k, v in metrics.get('exam_accuracy', {}).items()}
            })
    
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(os.path.join(output_dir, 'batch_benchmark_results.csv'), index=False)
    
    generate_batch_summary(df_report, output_dir)
    
    print(f"📊 Detailed results saved to: {os.path.join(output_dir, 'batch_benchmark_results.csv')}")

def generate_batch_summary(df_report, output_dir):
    """Generate batch processing summary"""
    # Get language columns for analysis
    language_cols = [col for col in df_report.columns if col.startswith('language_') and not col.startswith('language_null_')]
    
    summary = f"""# Batch Processing LLM Benchmark Results
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Models Evaluated: {len(df_report)}
- Best Accuracy: {df_report['overall_accuracy'].max():.3f}
- Average Accuracy: {df_report['overall_accuracy'].mean():.3f}
- Models with >80% Accuracy: {len(df_report[df_report['overall_accuracy'] > 0.8])}
- Models with High Null Rate (>30%): {len(df_report[df_report['null_prediction_rate'] > 0.3])}
- Languages Tested: {len(language_cols)}

## Top 10 Models by Accuracy
"""
    
    top_10 = df_report.nlargest(10, 'overall_accuracy')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        processing_time = row['total_processing_time']
        summary += f"{i:2d}. {row['model']:<50} | Acc: {row['overall_accuracy']:.3f} | Time: {processing_time:6.1f}s | Batch: {row['batch_size']}\n"
    
    summary += f"""

## Batch Processing Performance
- Average Processing Time per Model: {df_report['total_processing_time'].mean():.1f}s
- Fastest Model Processing: {df_report['total_processing_time'].min():.1f}s
- Slowest Model Processing: {df_report['total_processing_time'].max():.1f}s
- Average Batch Success Rate: {df_report['batch_success_rate'].mean():.3f}

## Language-wise Average Performance (Top Models)
"""
    
    # Calculate average accuracy per language for top models
    top_models = df_report.nlargest(5, 'overall_accuracy')
    if language_cols:
        language_names = [col.replace('language_', '') for col in language_cols]
        avg_accuracies = []
        for lang_col in language_cols:
            if lang_col in top_models.columns:
                avg_acc = top_models[lang_col].mean()
                avg_accuracies.append((lang_col.replace('language_', ''), avg_acc))
        
        # Sort by accuracy
        avg_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        for lang, acc in avg_accuracies[:10]:  # Show top 10 languages
            summary += f"- {lang}: {acc:.3f} average accuracy\n"
    
    summary += f"""

## Models Requiring Investigation (High Null Rate)
"""
    
    problematic = df_report[df_report['null_prediction_rate'] > 0.3].sort_values('null_prediction_rate', ascending=False)
    for _, row in problematic.iterrows():
        summary += f"- {row['model']}: {row['null_prediction_rate']:.3f} null rate\n"
    
    with open(os.path.join(output_dir, 'batch_processing_summary.md'), 'w') as f:
        f.write(summary)

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM models with batch processing")
    parser.add_argument(
        "--group",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "all"],
        help="Model group to run: small, medium, large, or all (default: medium)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing the dataset (default: ./data)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="enhanced_benchmark_results",
        help="Directory for saving results (default: enhanced_benchmark_results)"
    )
    
    args = parser.parse_args()
    
    # Configuration
    group_to_run = args.group
    batch_size = args.batch_size
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    
    print(f"\n🔧 Configuration:")
    print(f"   Group: {group_to_run}")
    print(f"   Batch size: {batch_size}")
    print(f"   Data directory: {data_dir}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    df = load_subject_data(data_dir)
    if df.empty:
        print("No data found! Please check your data directory structure.")
        return
    
    print(f"Loaded {len(df)} questions from {df['subject'].nunique()} subjects")
    
    # Run benchmarking
    all_results = {}
    
    if group_to_run == 'all':
        for group_name in MODEL_GROUPS.keys():
            group_results = run_benchmark_group_batch(group_name, df, checkpoint_dir, batch_size)
            all_results.update(group_results)
    elif group_to_run in MODEL_GROUPS:
        group_results = run_benchmark_group_batch(group_to_run, df, checkpoint_dir, batch_size)
        all_results.update(group_results)
    else:
        print("Invalid group selection!")
        return
    
    generate_batch_report(all_results, output_dir)
    
    print(f"\n🎉 Batch benchmarking complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n📊 Quick Results Summary:")
    print("-" * 80)
    for model_name, metrics in all_results.items():
        if metrics:
            acc = metrics['accuracy']
            null_rate = metrics['null_prediction_rate']
            time_taken = metrics.get('total_processing_time', 0)
            print(f"{model_name:<45} | Acc: {acc:.3f} | Null: {null_rate:.3f} | Time: {time_taken:6.1f}s")

if __name__ == "__main__":
    main()