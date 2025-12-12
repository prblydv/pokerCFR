# ---------------------------------------------------------------------------
# File overview:
#   main.py exposes a CLI menu for training, evaluation, matches, and playing.
#   Run directly via `python main.py` to explore the bot capabilities.
# ---------------------------------------------------------------------------
"""
Deep CFR Poker Bot - Main Entry Point

Production-grade poker bot using Deep Counterfactual Regret Minimization.

This script provides an interactive menu for:
1. Training a new bot
2. Playing bot vs bot matches
3. Playing interactive bot vs human
4. Analyzing trained models

Usage:
    python main.py
"""

import sys
import os
import logging
from pathlib import Path

# Set up logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("DeepCFR")


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = print_header()  # dtype=Any
def print_header():
    """Print application header."""
    print("\n" + "="*70)
    print(" "*15 + "DEEP CFR POKER BOT - PRODUCTION GRADE")
    print("="*70 + "\n")


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = print_menu()  # dtype=Any
def print_menu():
    """Print main menu."""
    print("MAIN MENU")
    print("-" * 50)
    print("1. Train a new bot")
    print("2. Bot vs Bot match")
    print("3. Play against bot (interactive)")
    print("4. View training curves")
    print("5. List available checkpoints")
    print("6. Exit")
    print("-" * 50)


# Function metadata:
#   Inputs: path  # dtype=varies
#   Sample:
#       sample_output = validate_checkpoint(path='path/to/file.pt')  # dtype=Any
def validate_checkpoint(path: str) -> bool:
    """Validate that a checkpoint directory exists and has required files."""
    if not os.path.isdir(path):
        print(f"ERROR: Directory not found: {path}")
        return False
    
    required_files = ["adv_p0.pt", "adv_p1.pt", "policy.pt"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
    
    if missing:
        print(f"ERROR: Missing files in checkpoint: {', '.join(missing)}")
        return False
    
    return True


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = train_bot()  # dtype=Any
def train_bot():
    """Train a new bot."""
    print("\n" + "="*50)
    print("TRAINING NEW BOT")
    print("="*50 + "\n")
    
    print("Loading config...")
    from config import NUM_ITERATIONS, TRAVERSALS_PER_ITER, STRAT_SAMPLES_PER_ITER
    
    print(f"Training parameters:")
    print(f"  Iterations:           {NUM_ITERATIONS}")
    print(f"  Traversals per iter:  {TRAVERSALS_PER_ITER}")
    print(f"  Strat samples/iter:   {STRAT_SAMPLES_PER_ITER}")
    
    proceed = input("\nProceed with training? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Training cancelled.")
        return
    
    try:
        import run_deep_cfr
        run_deep_cfr.main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = bot_vs_bot()  # dtype=Any
def bot_vs_bot():
    """Run a bot vs bot match."""
    print("\n" + "="*50)
    print("BOT VS BOT MATCH")
    print("="*50 + "\n")
    
    # Get checkpoints
    model1 = input("Model 1 path (default: models): ").strip() or "models"
    if not validate_checkpoint(model1):
        return
    
    model2 = input("Model 2 path (default: models): ").strip() or "models"
    if not validate_checkpoint(model2):
        return
    
    try:
        hands = int(input("Number of hands (default: 100): ").strip() or "100")
    except ValueError:
        print("Invalid number of hands")
        return
    
    verbose = input("Print hand history? (y/n): ").strip().lower() == 'y'
    
    print(f"\nRunning match: {hands} hands")
    print("-" * 50)
    
    try:
        from bot_match_engine import BotMatchEngine
        engine = BotMatchEngine(model1, model2)
        engine.run_match(hands, verbose=verbose)
        engine.print_stats()
    except Exception as e:
        logger.error(f"Match failed: {e}", exc_info=True)


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = play_vs_human()  # dtype=Any
def play_vs_human():
    """Play interactive bot vs human game."""
    print("\n" + "="*50)
    print("BOT VS HUMAN - INTERACTIVE")
    print("="*50 + "\n")
    
    model = input("Bot model path (default: models): ").strip() or "models"
    if not validate_checkpoint(model):
        return
    
    position = input("Your position - (b)utton or (B)ig blind? (default: button): ").strip().lower()
    if position in ['b', 'button']:
        position = "button"
    elif position in ['g', 'big']:
        position = "bb"
    else:
        position = "button"
    
    try:
        hands = int(input("Number of hands (default: 10): ").strip() or "10")
    except ValueError:
        print("Invalid number of hands")
        return
    
    print(f"\nStarting interactive game...")
    print("-" * 50)
    
    try:
        from interactive_play import InteractivePokerGame
        game = InteractivePokerGame(model, human_is_player0=(position == "button"))
        game.play_session(hands)
    except Exception as e:
        logger.error(f"Game failed: {e}", exc_info=True)


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = show_training_curves()  # dtype=Any
def show_training_curves():
    """Display training curves if available."""
    if not os.path.exists("training_curves.png"):
        print("No training curves found. Run training first.")
        return
    
    print("Training curves found at: training_curves.png")
    try:
        import matplotlib.pyplot as plt
        img = plt.imread("training_curves.png")
        print(f"Image shape: {img.shape}")
        print("(Open training_curves.png in your image viewer)")
    except Exception as e:
        print(f"Could not display image: {e}")


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = list_checkpoints()  # dtype=Any
def list_checkpoints():
    """List available checkpoints."""
    print("\nAvailable checkpoints:")
    print("-" * 50)
    
    checkpoints_found = False
    
    # Check models directory
    if os.path.isdir("models"):
        required = ["adv_p0.pt", "adv_p1.pt", "policy.pt"]
        if all(os.path.exists(f"models/{f}") for f in required):
            print("✓ models/ - Main checkpoint")
            checkpoints_found = True
    
    # Check for other directories
    for item in os.listdir("."):
        if os.path.isdir(item) and item != "models" and item != "__pycache__":
            required = ["adv_p0.pt", "adv_p1.pt", "policy.pt"]
            if all(os.path.exists(f"{item}/{f}") for f in required):
                print(f"✓ {item}/")
                checkpoints_found = True
    
    if not checkpoints_found:
        print("No checkpoints found. Train a bot first.")
    print()


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = verify_environment()  # dtype=Any
def verify_environment():
    """Verify required packages are installed."""
    required_packages = {
        'torch': 'torch',
        'treys': 'treys',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("\nERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True


# Function metadata:
#   Inputs: no explicit parameters  # dtype=varies
#   Sample:
#       sample_output = main_menu()  # dtype=Any
def main_menu():
    """Main menu loop."""
    if not verify_environment():
        return 1
    
    print_header()
    
    while True:
        print_menu()
        choice = input("Select option (1-6): ").strip()
        
        if choice == "1":
            train_bot()
        elif choice == "2":
            bot_vs_bot()
        elif choice == "3":
            play_vs_human()
        elif choice == "4":
            show_training_curves()
        elif choice == "5":
            list_checkpoints()
        elif choice == "6":
            print("\nExiting. Goodbye!")
            return 0
        else:
            print("Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        exit_code = main_menu()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
