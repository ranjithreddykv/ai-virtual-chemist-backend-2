import traceback
from ..inference import load_model as raw_load_model

model_loaded = False


def load_model():
    """
    Loads your GNN mechanism prediction model only once.
    Called on FastAPI startup.
    """
    global model_loaded

    if model_loaded:
        print("⏭️ Model already loaded, skipping.")
        return

    try:
        raw_load_model()
        model_loaded = True
        print("✅ ML Model Ready")
    except Exception as e:
        print("❌ ERROR: Failed to load ML model!")
        traceback.print_exc()
        raise RuntimeError(f"Model load failed: {e}")
