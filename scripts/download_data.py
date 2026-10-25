"""Download and verify the real METR-LA dataset."""

from fedroad.data import download_metr_la


if __name__ == "__main__":
    path = download_metr_la()
    print(f"Ready to use: {path}")
