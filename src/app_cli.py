from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from src.chains.rag_chain import SupportPearlzRAGChain
from src.chains.memory import ChatSessionMemory
from src.utils.logging_setup import logger

console = Console()

def run_cli():
    console.print(Panel.fit("[bold blue]SupportPearlz Customer Support Agent[/bold blue]\n[dim]Pearlz Home Systems (Pvt.) Ltd.[/dim]"))
    
    chain = SupportPearlzRAGChain()
    memory = ChatSessionMemory(max_turns=5)
    
    console.print("[green]Session started. Type 'exit' to quit or 'reset' to clear chat memory.[/green]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold red]Ending session. Goodbye![/bold red]")
                break
            elif user_input.lower() == "reset":
                memory.clear()
                console.print("[bold cyan]Session history cleared.[/bold cyan]\n")
                continue

            response = chain.run(question=user_input, chat_history=memory.get_messages())

            memory.add_user_message(user_input)
            memory.add_ai_message(response.answer)

            console.print(f"\n[bold green]Bot[/bold green] > {response.answer}")
            
            if response.sources:
                console.print(f"[dim]Sources: {', '.join(response.sources)}[/dim]")
            console.print(f"[dim]Confidence: {response.confidence}[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n[bold red]Session terminated.[/bold red]")
            break
        except Exception as e:
            logger.error(f"Unhandled REPL Error: {str(e)}")
            console.print("[bold red]An unexpected system error occurred. Check log files for detail.[/bold red]\n")

if __name__ == "__main__":
    run_cli()