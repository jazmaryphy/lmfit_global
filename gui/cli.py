# %%
import sys
from pathlib import Path
from streamlit.web import cli as stcli

# %%
def main() -> None:
    """Entry point for the lmfitgedit CLI command."""
    # Dynamically locate app.py relative to this cli.py file
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print(f"Error: Streamlit application not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Pass 'run' and the path to app.py into Streamlit's CLI runner
    sys.argv = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()