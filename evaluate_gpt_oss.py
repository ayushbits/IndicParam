"""
GPT-OSS-120B inference script for benchmarking
OpenAI's open-weight model with 117B parameters (5.1B active)
Supports MXFP4 quantization to fit on single 80GB GPU
"""

import torch
import pandas as pd
import json
import time
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import gc

# Set HuggingFace cache directory
os.environ['HF_HOME'] = '/userhome/home/aymaheshwari/hf-cache'
os.environ['TRANSFORMERS_CACHE'] = '/userhome/home/aymaheshwari/hf-cache/hub'

# Cache directory for models
CACHE_DIR = "/userhome/home/aymaheshwari/hf-cache/hub"

# Model configuration
MODEL_NAME = "openai/gpt-oss-120b"

def load_model(reasoning_level="low", use_bf16=True):
    """
    Load GPT-OSS-120B model
    
    Args:
        reasoning_level: "low", "medium", or "high" for reasoning effort
        use_bf16: If True, uses BF16 precision (dequantizes from MXFP4)
                  If False, uses MXFP4 quantization (default for single GPU)
    
    Note: Model weights are stored in MXFP4 by default.
    - torch_dtype="auto" -> Uses MXFP4 (quantized, fits on single 80GB GPU)
    - torch_dtype=torch.bfloat16 -> Dequantizes to BF16 (full precision, needs more VRAM)
    
    With 640GB VRAM, we can afford BF16 for best accuracy.
    """
    print(f"Loading {MODEL_NAME} with reasoning level: {reasoning_level}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    if use_bf16:
        print(f"Loading in BF16 precision (dequantized from MXFP4)")
        dtype = torch.bfloat16
    else:
        print(f"Loading in MXFP4 quantized format (default)")
        dtype = "auto"
    
    try:
        # Load model using Transformers pipeline
        # https://huggingface.co/openai/gpt-oss-120b
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=dtype,   # BF16 for full precision or "auto" for MXFP4
            device_map="auto",   # Auto-distribute across GPUs
        )
        
        print("✅ Model loaded successfully")
        print(f"Model distributed across {torch.cuda.device_count()} GPUs")
        
        # Print memory usage
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return pipe, reasoning_level

def create_messages(question_text, options, language="Unknown", reasoning_level="low"):
    """
    Create messages for GPT-OSS-120B using harmony format
    The Transformers chat template automatically applies the harmony response format
    
    Reasoning levels:
    - low: Fast responses for general dialogue
    - medium: Balanced speed and detail
    - high: Deep and detailed analysis
    """
    
    # Build the prompt with reasoning level in system message as per docs
    question_prompt = f"""Language: {language}

Question: {question_text}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

The above question is written in {language} language. Please analyze the question and options carefully, and select the correct answer. Respond ONLY with one letter (A, B, C, or D) corresponding to the correct option. Do not provide any explanation or additional text."""
    
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that answers multiple choice questions accurately. Reasoning: {reasoning_level}"
        },
        {
            "role": "user",
            "content": question_prompt
        }
    ]
    
    return messages

def generate_response(pipe, messages, max_new_tokens=2048):
    """
    Generate response using the GPT-OSS-120B pipeline
    The pipeline automatically handles the harmony response format
    """
    
    try:
        # Generate using pipeline
        # Note: pipeline doesn't accept 'temperature', use 'do_sample=False' for deterministic output
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for benchmarking
            pad_token_id=pipe.tokenizer.eos_token_id,  # Avoid warnings
        )
        
        # Extract the assistant's response
        # The output is in format: [{"generated_text": [messages + assistant_response]}]
        generated_messages = outputs[0]["generated_text"]
        
        # Get the last message (assistant's response)
        assistant_message = generated_messages[-1]
        
        if assistant_message.get("role") == "assistant":
            response = assistant_message.get("content", "")
        else:
            response = ""
        
        return response.strip()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_batch_responses(pipe, batch_messages, max_new_tokens=2048):
    """
    Generate responses for a batch of messages using the GPT-OSS-120B pipeline
    
    Args:
        pipe: The transformers pipeline
        batch_messages: List of message lists (one per question)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of response strings
    """
    try:
        # Generate using pipeline with batching
        outputs = pipe(
            batch_messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for benchmarking
            pad_token_id=pipe.tokenizer.eos_token_id,
            batch_size=len(batch_messages),  # Process all at once
        )
        
        # Extract responses from batch
        # When batching, outputs is a list where each item is a list containing one dict
        # outputs = [[{"generated_text": [messages]}], [{"generated_text": [messages]}], ...]
        responses = []
        for idx, output_item in enumerate(outputs):
            try:
                # output_item is a list with one dict
                if isinstance(output_item, list) and len(output_item) > 0:
                    output_dict = output_item[0]
                else:
                    output_dict = output_item
                
                # Now extract the assistant response
                # output_dict is like: {"generated_text": [system_msg, user_msg, assistant_msg]}
                assistant_response = output_dict["generated_text"][-1]
                
                # The assistant response should be a dict with 'content' field
                if isinstance(assistant_response, dict):
                    response = assistant_response.get("content", "")
                    if not response:
                        print(f"[DEBUG] Question {idx}: No 'content' field. Full response: {assistant_response}")
                elif isinstance(assistant_response, str):
                    response = assistant_response
                else:
                    response = str(assistant_response)
                
                responses.append(response.strip() if response else "")
                
            except Exception as e:
                print(f"[DEBUG] Error extracting response {idx}: {e}")
                print(f"[DEBUG] Output type: {type(output_item)}")
                print(f"[DEBUG] Output structure: {output_item}")
                responses.append("")
        
        return responses
        
    except Exception as e:
        print(f"Error during batch generation: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to individual generation
        print("Falling back to individual generation...")
        return [generate_response(pipe, msgs, max_new_tokens) for msgs in batch_messages]

def extract_answer(response_text):
    """Extract answer letter from response"""
    import re
    
    if not response_text:
        return None
    
    # Convert to uppercase for matching
    answer_upper = str(response_text).upper().strip()
    
    # GPT-OSS specific pattern: "assistantfinal[A-D]" at the end
    gpt_oss_pattern = re.search(r'ASSISTANTFINAL([ABCD])', answer_upper)
    if gpt_oss_pattern:
        return gpt_oss_pattern.group(1)
    
    # Look for "answer [A-D]" pattern
    answer_pattern = re.search(r'ANSWER\s+([ABCD])', answer_upper)
    if answer_pattern:
        return answer_pattern.group(1)
    
    # Look for isolated letters with word boundaries
    isolated_match = re.search(r'\b([ABCD])\b', answer_upper)
    if isolated_match:
        return isolated_match.group(1)
    
    # Look for letters at start or after patterns
    pattern_match = re.search(r'(?:^|answer[:\s]*|choice[:\s]*|option[:\s]*)([ABCD])(?:\)|\.|\s|$)', answer_upper, re.IGNORECASE)
    if pattern_match:
        return pattern_match.group(1)
    
    # Fallback: last occurrence (most likely to be the final answer)
    for letter in ['D', 'C', 'B', 'A']:  # Reverse order to get last occurrence
        if letter in answer_upper:
            return letter
    
    return None

def benchmark_on_dataset(pipe, reasoning_level, df, batch_size=8, checkpoint_dir="checkpoints"):
    """
    Benchmark GPT-OSS-120B on the dataset with batching support
    
    Args:
        pipe: Transformers pipeline
        reasoning_level: Reasoning level to use
        df: DataFrame with questions
        batch_size: Number of questions to process in parallel (default: 8)
        checkpoint_dir: Directory to save checkpoints
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {MODEL_NAME}")
    print(f"Reasoning Level: {reasoning_level}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}")
    
    results = {
        'model_name': MODEL_NAME,
        'reasoning_level': reasoning_level,
        'responses': {},
        'timestamps': {},
        'processed_questions': [],
        'batch_processing_info': {
            'batch_size': batch_size,
            'total_questions': len(df),
            'processing_start': datetime.now().isoformat()
        }
    }
    
    # Process questions in batches
    correct_count = 0
    total_count = 0
    
    # Prepare batches
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        # Prepare batch data
        batch_messages = []
        batch_metadata = []
        
        for idx, row in batch_df.iterrows():
            question_id = row['unique_question_id']
            
            # Prepare options
            options = {
                'A': row['option_a'],
                'B': row['option_b'],
                'C': row['option_c'],
                'D': row['option_d']
            }
            
            # Get language
            language = row.get('language', row.get('subject', 'Unknown'))
            
            # Create messages
            messages = create_messages(question_text=row['question_text'], 
                                       options=options, 
                                       language=language,
                                       reasoning_level=reasoning_level)
            
            batch_messages.append(messages)
            batch_metadata.append({
                'question_id': question_id,
                'correct_answer': row['correct_answer'],
                'subject': row['subject'],
                'language': language
            })
        
        # Generate responses for batch
        start_time = time.time()
        try:
            batch_responses = generate_batch_responses(pipe, batch_messages)
            batch_elapsed_time = time.time() - start_time
            per_question_time = batch_elapsed_time / len(batch_responses)
            
            # Process batch results
            for response, metadata in zip(batch_responses, batch_metadata):
                question_id = metadata['question_id']
                
                # Extract answer
                predicted_answer = extract_answer(response)
                is_correct = predicted_answer.upper() == metadata['correct_answer'].upper() if predicted_answer and metadata['correct_answer'] else False
                
                # Store results
                results['responses'][question_id] = {
                    'raw_response': response,
                    'predicted_answer': predicted_answer,
                    'correct_answer': metadata['correct_answer'],
                    'subject': metadata['subject'],
                    'language': metadata['language'],
                    'is_correct': is_correct
                }
                
                results['timestamps'][question_id] = per_question_time
                results['processed_questions'].append(question_id)
                
                if is_correct:
                    correct_count += 1
                total_count += 1
            
            # Print progress every 10 questions
            if total_count % 10 == 0:
                current_accuracy = correct_count / total_count
                print(f"Progress: {total_count}/{len(df)} | Accuracy: {current_accuracy:.3f}")
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Store error results for this batch
            for metadata in batch_metadata:
                question_id = metadata['question_id']
                results['responses'][question_id] = {
                    'raw_response': "",
                    'predicted_answer': None,
                    'correct_answer': metadata['correct_answer'],
                    'is_correct': False,
                    'error': str(e)
                }
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Calculate final metrics
    results['accuracy'] = correct_count / total_count if total_count > 0 else 0
    results['correct_count'] = correct_count
    results['total_count'] = total_count
    results['batch_processing_info']['processing_end'] = datetime.now().isoformat()
    
    # Save results
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"gpt_oss_120b_{reasoning_level}_checkpoint.json")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults:")
    print(f"  Total questions: {total_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Results saved to: {checkpoint_file}")
    
    return results

def main():
    """Main function to run GPT-OSS-120B benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark GPT-OSS-120B model')
    parser.add_argument('--reasoning', type=str, choices=['low', 'medium', 'high'], default='low',
                        help='Reasoning level: low (fast), medium (balanced), high (detailed) - default: low')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions for testing')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for parallel processing (default: 8)')
    args = parser.parse_args()
    
    # Load dataset
    data_path = Path("./data/data-3.csv")
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
    
    # Load model with specified reasoning level
    pipe, reasoning_level = load_model(reasoning_level=args.reasoning)
    
    # Run benchmark
    print("\n" + "="*60)
    print(f"Starting GPT-OSS-120B Benchmark (Reasoning: {args.reasoning})")
    print("="*60)
    
    results = benchmark_on_dataset(
        pipe=pipe, 
        reasoning_level=reasoning_level, 
        df=df,
        batch_size=args.batch_size,
        checkpoint_dir="checkpoints-3"
    )
    
    # Print final results
    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Reasoning Level: {args.reasoning.upper()}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total Questions: {results['total_count']}")
    print(f"Correct Answers: {results['correct_count']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Results saved to: checkpoints/gpt_oss_120b_{args.reasoning}_checkpoint.json")
    
    # Clean up
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()

