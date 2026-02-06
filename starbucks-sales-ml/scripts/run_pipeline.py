from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(["python", "-m", "starbucks_sales_ml.train"], check=True)


if __name__ == "__main__":
    main()
