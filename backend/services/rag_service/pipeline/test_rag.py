from .generator import generate_answer

while True:
    q = input("\nAsk medical question: ")
    print("\nAnswer:\n", generate_answer(q))