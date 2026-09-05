from src.chains.rag_chain import get_rag_chain
from src.utils.logging_setup import logger

def run_cli():
    logger.info("Starting SupportPearlz Interactive CLI...")
    rag_chain = get_rag_chain()
    
    print("\n==========================================")
    print("   SupportPearlz AI Customer Support CLI  ")
    print("==========================================\n")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting SupportPearlz CLI. Goodbye!")
                break

            response = rag_chain.invoke(user_input)
            print(f"\nAssistant: {response.answer}")
            if response.sources:
                print(f"Sources: {', '.join(response.sources)} | Confidence: {response.confidence}")
            print("-" * 50 + "\n")
        except KeyboardInterrupt:
            print("\nExiting CLI session...")
            break
        except Exception as e:
            logger.error(f"Error processing command: {e}")

if __name__ == "__main__":
    run_cli()