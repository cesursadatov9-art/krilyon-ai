from model import NEXModel


def main():
    ai = NEXModel()
    history = []

    print("\nKrilyonNEX | NEX-1")
    print("Çıxmaq üçün: çıx\n")

    while True:
        try:
            message = input("Sən: ").strip()

            if message.lower() in {"çıx", "exit", "quit"}:
                print("NEX-1: Görüşənədək!")
                break

            if not message:
                continue

            answer = ai.chat(message, history=history)
            print(f"NEX-1: {answer}\n")

            history.append((message, answer))
            history = history[-10:]

        except (KeyboardInterrupt, EOFError):
            print("\nSöhbət bitdi.")
            break


if __name__ == "__main__":
    main()
