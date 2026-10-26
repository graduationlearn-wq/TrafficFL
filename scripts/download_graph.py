"""Download and verify the official METR-LA sensor adjacency matrix."""

from fedroad.graph_data import DEFAULT_GRAPH_PATH, download_metr_la_graph

if __name__ == "__main__":
    result = download_metr_la_graph()
    print(f"Graph ready: {result}")
