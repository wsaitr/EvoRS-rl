#!/usr/bin/env python3
from rs_hmcomm.datasets import stream_vrsbench

def main():
    for sample in stream_vrsbench(limit=3):
        print(sample.sample_id, sample.question[:120], "=>", sample.answer[:80])
        print("metadata:", sample.metadata)

if __name__ == "__main__":
    main()
