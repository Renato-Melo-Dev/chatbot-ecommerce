import os

def load_context():
    base_path = os.path.dirname(os.path.dirname(__file__))
    context_path = os.path.join(base_path, "data", "context.txt")

    with open(context_path, "r", encoding="utf-8") as f:
        return f.read()
