import argparse

from perspect.utils.io import jsonl_to_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--drop", nargs="*", default=["prompt"], help="Columns to leave out.")
    args = p.parse_args()

    n = jsonl_to_csv(args.input, args.output, drop_columns=tuple(args.drop))
    print(f"Wrote {n:,} rows to {args.output}")


if __name__ == "__main__":
    main()
