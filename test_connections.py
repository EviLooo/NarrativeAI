import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_cloudant():
    print("Checking Cloudant connection...")
    from ibmcloudant.cloudant_v1 import CloudantV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

    apikey = os.getenv("CLOUDANT_API_KEY")
    url = os.getenv("CLOUDANT_URL")
    master_key = os.getenv("IBM_CLOUD_API_KEY")

    print(f"DEBUG: Cloudant URL={url}")
    if not apikey or not url:
        print("[FAIL] Cloudant credentials missing in .env")
        return False

    def try_connect(key, label):
        try:
            print(f"Trying Cloudant connection with {label}...")
            authenticator = IAMAuthenticator(key)
            client = CloudantV1(authenticator=authenticator)
            client.set_service_url(url)
            
            # Try listing databases (requires less privilege than server info sometimes)
            dbs = client.get_all_dbs().get_result()
            print(f"[OK] Cloudant connected successfully using {label}! Available databases: {dbs}")
            return True, key
        except Exception as e:
            print(f"[DEBUG] Cloudant connection with {label} failed: {e}")
            return False, None

    # Try service-specific key first
    success, working_key = try_connect(apikey, "Service API Key")
    if not success and master_key:
        # Fall back to master account key
        success, working_key = try_connect(master_key, "Master IBM Cloud API Key")
        if success:
            print("[NOTE] Cloudant worked with Master API Key. We will use this instead.")
            # We can print a suggestion to update .env
            
    return success

def test_nlu():
    print("Checking Watson NLU connection...")
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions

    apikey = os.getenv("NLU_API_KEY")
    url = os.getenv("NLU_URL")

    if not apikey or not url:
        print("[FAIL] NLU credentials missing in .env")
        return False

    try:
        authenticator = IAMAuthenticator(apikey)
        nlu = NaturalLanguageUnderstandingV1(
            version='2022-04-07',
            authenticator=authenticator
        )
        nlu.set_service_url(url)
        
        # Analyze a simple sentence to test the service
        # (This is lightweight and doesn't consume generation credits)
        response = nlu.analyze(
            text="Jon Snow lives in Winterfell.",
            features=Features(entities=EntitiesOptions(limit=2))
        ).get_result()
        
        entities = [ent['text'] for ent in response.get('entities', [])]
        print(f"[OK] Watson NLU connected successfully! Extracted entities: {entities}")
        return True
    except Exception as e:
        print(f"[FAIL] Watson NLU connection failed: {e}")
        return False

def test_watsonx_ai():
    print("Checking watsonx.ai connection...")
    from ibm_watsonx_ai.foundation_models import ModelInference

    api_key = os.getenv("IBM_CLOUD_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")

    print(f"DEBUG: watsonx URL={url}, Project ID={project_id}")
    if not api_key or not project_id or not url:
        print("[FAIL] watsonx.ai credentials missing in .env")
        return False

    try:
        # Use standard dictionary format for credentials
        credentials = {
            "url": url,
            "apikey": api_key
        }
        
        # Let's try to initialize a supported model
        # Using Llama 3.1 8B, which is supported on this environment
        model_id = "meta-llama/llama-3-1-8b"
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id
        )
        print(f"[OK] Successfully initialized watsonx.ai model: {model_id}")
        return True
    except Exception as e:
        print(f"[FAIL] watsonx.ai connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=========================================")
    print("Starting IBM Cloud Service Connectivity Test")
    print("=========================================")
    
    cloudant_ok = test_cloudant()
    print("-----------------------------------------")
    nlu_ok = test_nlu()
    print("-----------------------------------------")
    watsonx_ok = test_watsonx_ai()
    print("=========================================")
    
    if cloudant_ok and nlu_ok and watsonx_ok:
        print("ALL IBM SERVICES CONNECTED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME SERVICES FAILED TO CONNECT. Please check credentials.")
        sys.exit(1)
