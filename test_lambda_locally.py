import json
import random
import time
import uuid
from typing import List, Dict, Any

# Attempt to import the lambda_handler from lambda_function.py
# This assumes test_lambda_locally.py is in the same directory as lambda_function.py
try:
    from lambda_function import lambda_handler
except ImportError:
    print("ERROR: Could not import lambda_handler from lambda_function.py.")
    print("Ensure test_lambda_locally.py is in the same directory as lambda_function.py,")
    print("and that lambda_function.py can be imported (e.g., no top-level errors on import).")
    lambda_handler = None # Set to None if import fails

# --- Data Generation ---

GOAL_TYPES = [
    "lose weight", "gain muscle", "improve endurance", "run a marathon", "reduce back pain",
    "eat healthier", "prepare for a 5k run", "increase flexibility", "recover from a knee injury",
    "get stronger for daily tasks", "reduce stress", "improve sleep quality", "train for a triathlon",
    "body recomposition", "learn a new sport (e.g., rock climbing)"
]

SPECIFIC_EVENTS_EXAMPLES = [
    "wedding in 3 months", "beach vacation in 2 months", "company fitness challenge", "New Year's resolution",
    "doctor's recommendation", "upcoming marathon in 6 months", "class reunion", "no specific deadline",
    "personal milestone birthday", "summer hiking trip"
]

PAST_ATTEMPTS_EXAMPLES = [
    "tried keto, too restrictive", "used MyFitnessPal for a bit", "CrossFit for a year, got too intense",
    "followed YouTube workout videos inconsistently", "nothing serious before", "joined a gym but didn't go often",
    "intermittent fasting, saw some results", "worked with a personal trainer for a short period",
    "strictly cardio, got bored", "used to play sports in high school"
]

SCHEDULE_EXAMPLES = [
    "can work out 3-4 times a week, mostly evenings", "weekends are flexible, weekdays very busy",
    "only 30 mins available per day", "prefers morning workouts", "can do 2 long sessions on weekends",
    "travels a lot for work, needs adaptable plan", "full-time student, schedule varies",
    "new parent, very limited and unpredictable time", "can hit the gym during lunch break",
    "looking for home workouts mainly"
]

INJURIES_EXAMPLES = [
    "occasional knee discomfort", "past shin splints from running", "lower back pain when lifting heavy",
    "shoulder impingement", "none", "recovering from ACL surgery 6 months ago", "carpal tunnel syndrome",
    "tight hamstrings and hips", "previous ankle sprain, still a bit weak", "plantar fasciitis flare-ups"
]

BS_ANSWERS = [
    "fly to the moon", "become a wizard", "eat pizza all day and get shredded", "teleport",
    "I dunno, stuff I guess", "N/A", " ", "my dog ate my homework", "to find the meaning of life",
    "reverse time"
]

def generate_test_case_id(goal_type: str, index: int) -> str:
    slug = goal_type.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return f"test_case_{slug}_{index}"

def generate_single_test_case(index: int, num_total_cases: int) -> Dict[str, Any]:
    """Generates a single realistic or edge-case test scenario."""
    
    data: Dict[str, Any] = {}
    
    # Ensure some BS answers are included
    # Make roughly 10% of answers BS answers for various fields
    use_bs_answer_for_goal = random.random() < 0.1
    use_bs_answer_for_field = random.random() < 0.1

    if use_bs_answer_for_goal:
        data["primary_goal"] = random.choice(BS_ANSWERS)
    else:
        data["primary_goal"] = random.choice(GOAL_TYPES)

    data["specific_event"] = random.choice(BS_ANSWERS) if use_bs_answer_for_field and random.random() < 0.3 else random.choice(SPECIFIC_EVENTS_EXAMPLES)
    data["past_attempts"] = random.choice(BS_ANSWERS) if use_bs_answer_for_field and random.random() < 0.3 else random.choice(PAST_ATTEMPTS_EXAMPLES)
    data["schedule"] = random.choice(BS_ANSWERS) if use_bs_answer_for_field and random.random() < 0.3 else random.choice(SCHEDULE_EXAMPLES)
    data["injuries"] = random.choice(BS_ANSWERS) if use_bs_answer_for_field and random.random() < 0.3 else random.choice(INJURIES_EXAMPLES)
    
    # Vary answer length and detail
    if random.random() < 0.2: # 20% chance of very short answer for a field
        field_to_shorten = random.choice(list(data.keys()))
        if field_to_shorten != "primary_goal" or not use_bs_answer_for_goal : # don't shorten BS goals
             words = data[field_to_shorten].split()
             if len(words) > 2:
                data[field_to_shorten] = " ".join(random.sample(words, k=min(len(words), random.randint(1,2))))
             elif len(words) > 0:
                data[field_to_shorten] = words[0]


    if random.random() < 0.1: # 10% chance of empty answer for a field
        field_to_empty = random.choice(["specific_event", "past_attempts", "schedule", "injuries"])
        data[field_to_empty] = ""

    data["user_id"] = generate_test_case_id(data["primary_goal"], index)
    
    return data

def generate_test_data(num_samples: int = 50) -> List[Dict[str, Any]]:
    """Generates a list of test data dictionaries."""
    test_data_list: List[Dict[str, Any]] = []
    for i in range(num_samples):
        test_data_list.append(generate_single_test_case(i, num_samples))
    return test_data_list

# --- Lambda Invocation ---

def call_lambda_locally(test_case_data: Dict[str, Any], case_index: int):
    """
    Calls the imported lambda_handler with the provided test case data.
    Simulates the event structure Lambda expects from an API Gateway or similar trigger.
    """
    if not lambda_handler:
        print(f"Skipping test case {case_index + 1} due to lambda_handler import failure.")
        return

    print(f"--- Running Test Case {case_index + 1}/{total_test_cases} ---")
    print(f"User ID: {test_case_data.get('user_id')}")
    print(f"Input Data: {json.dumps(test_case_data, indent=2)}")

    # Simulate the event payload Lambda receives
    # Typically, the body is a JSON string.
    mock_event = {
        "body": json.dumps(test_case_data)
    }

    try:
        response = lambda_handler(mock_event, None) # Second argument is context, can be None for local tests
        print(f"Lambda Response (status {response.get('statusCode')}):")
        
        response_body = response.get('body')
        if response_body:
            try:
                # Attempt to parse and pretty-print if it's JSON
                parsed_body = json.loads(response_body)
                print(json.dumps(parsed_body, indent=2))
            except json.JSONDecodeError:
                # If not JSON, print as is
                print(response_body)
        else:
            print("Lambda returned no body.")

    except Exception as e:
        print(f"ERROR calling lambda_handler for case {case_index + 1} (User ID: {test_case_data.get('user_id')}): {e}")
        import traceback
        traceback.print_exc()
    print("--------------------------------------\n")


# --- Main Execution ---
total_test_cases = 0 # Will be set after data generation

if __name__ == '__main__':
    num_test_samples = 50 # As per instructions_data.txt (50-100)
    print(f"Generating {num_test_samples} test data samples...")
    
    test_dataset = generate_test_data(num_samples=num_test_samples)
    total_test_cases = len(test_dataset)

    print(f"\n--- Starting Local Lambda Test Suite ---")
    print(f"Found lambda_handler: {'Yes' if lambda_handler else 'No'}")
    
    if lambda_handler:
        for i, test_case in enumerate(test_dataset):
            call_lambda_locally(test_case, i)
            if i < total_test_cases - 1:
                time.sleep(1) # Small delay between calls, can be adjusted or removed
        print("--- Local Lambda Test Suite Finished ---")
    else:
        print("Aborting test suite as lambda_handler could not be imported.") 