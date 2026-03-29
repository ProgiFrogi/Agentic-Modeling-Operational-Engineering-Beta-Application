from agents.supervisor import run_supervisor
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    COMPETITION_NAME = "mws-ai-agents-2026"

    print("🚀 Starting Kaggle Competition Agent System")
    print("=" * 60)

    result = run_supervisor(
        competition_name=COMPETITION_NAME,
        max_iterations=3
    )

    print("\n" + "=" * 60)
    print("✅ Process completed!")