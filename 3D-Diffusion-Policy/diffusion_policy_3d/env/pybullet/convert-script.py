#!/usr/bin/env python3

import sys

def main(input_file, output_file=None):
    out = open(output_file, "w") if output_file else sys.stdout

    buffer = []  # holds numbers until we reach 7

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Remove brackets
            line = line.replace("[", "").replace("]", "")

            # Split into numbers
            parts = line.split()

            for val in parts:
                buffer.append(val)

                # Once we have 7 values, write and reset
                if len(buffer) == 7:
                    out.write(" ".join(buffer) + "\n")
                    buffer.clear()

    # If anything left over (should not happen normally)
    if buffer:
        print(
            f"Warning: leftover values ignored ({len(buffer)}): {' '.join(buffer)}",
            file=sys.stderr
        )

    if output_file:
        out.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 format_vectors.py input.txt [output.txt]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_file, output_file)
