import numpy as np
import igraph as ig
import logging
import argparse
import os
import time
from tqdm.auto import tqdm

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that routes messages through tqdm.write to avoid breaking progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def main(args):
    input_graph = args.input_graph
    type_name = args.membership_attribute
    output_dir = args.output_dir
    delta_output_name = args.delta_output_name
    node_att_output_name = args.node_att_output_name

    # Load the graph
    graph_extension = os.path.splitext(input_graph)[1]
    if graph_extension == ".edgelist":
        g = ig.Graph.Read_Edgelist(input_graph, directed=False)
    elif graph_extension == ".gml":
        g = ig.Graph.Read_GML(input_graph)
    elif graph_extension == ".graphml":
        g = ig.Graph.Read_GraphML(input_graph)
    else:
        logging.error(f"Unsupported graph format: {graph_extension}")
        return 1
    logging.info(f"Graph loaded with {g.vcount()} nodes and {g.ecount()} edges.")
    # Collect node attributes vector
    if type_name not in g.vs.attributes():
        logging.error(f"Attribute '{type_name}' not found in graph vertices.")
        return 1
    node_attributes = g.vs[type_name]
    unique_attributes = list(set(node_attributes))
    attribute_to_index = {attr: idx for idx, attr in enumerate(unique_attributes)}
    node_attribute_vector = np.array([attribute_to_index[attr] for attr in node_attributes])
    num_attributes = len(unique_attributes)
    logging.info(f"Node attribute vector created with {num_attributes} unique attributes.")

    # Collect delta matrix
    delta = np.zeros((num_attributes, num_attributes), dtype=int)
    for edge in g.es:
        source_attr = node_attribute_vector[edge.source]
        target_attr = node_attribute_vector[edge.target]
        delta[source_attr, target_attr] += 1
        delta[target_attr, source_attr] += 1  # Undirected graph

    # Print delta matrix info
    logging.info(f"Delta matrix constructed with shape {delta.shape}.")
    logging.info(f"Delta matrix:\n{delta}")

    # Save outputs
    delta_output_path = os.path.join(output_dir, delta_output_name)
    node_att_output_path = os.path.join(output_dir, node_att_output_name)
    np.savetxt(delta_output_path, delta, delimiter=",", fmt="%d")
    logging.info(f"Delta matrix saved to {delta_output_path}.")
    np.savetxt(node_att_output_path, node_attribute_vector, delimiter=",", fmt="%d")
    logging.info(f"Node attribute vector saved to {node_att_output_path}.")

    return 0
if __name__ == "__main__":
    # Set up logging: should display INFO level messages and time stamps with seconds precision
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(prog='CommunitiesParser', description="Parse graph data from edge list files.")
    parser.add_argument("-i", "--input_graph", required=True, type=str, help="File path to the input graph.")
    parser.add_argument("-m", "--membership_attribute", required=True, type=str, help="Nodes' attribute name, used to infer the communities.")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Directory to save the output files.")
    parser.add_argument("-d", "--delta_output_name", type=str, default="delta.csv", help="File name for the delta matrix output.")
    parser.add_argument("-n", "--node_att_output_name", type=str, default="node_attribute_vector.csv", help="File name for the node attribute vector output.")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Whether to overwrite existing output files.")

    # Print the title of the program
    print("\n=== Geometric Block Model: Graph Data Parser ===\n")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    else:
        # Check for existing files if overwrite is False
        delta_path = os.path.join(args.output_dir, args.delta_output_name)
        node_att_path = os.path.join(args.output_dir, args.node_att_output_name)

        exist_files = []
        if os.path.exists(delta_path):
            exist_files.append(delta_path)
        if os.path.exists(node_att_path):
            exist_files.append(delta_path)

        if exist_files and not args.overwrite:
            logger.error(f"The following output files already exist: {', '.join(exist_files)}. Use --overwrite to replace them.")
            exit(1)
        
    # Print parsed arguments for verification
    logger.info(f"""Parsed arguments:
                \tInput graph: {args.input_graph}
                \tMembership attribute name: {args.membership_attribute}
                \tOutput directory: {args.output_dir}
                \tDelta output name: {args.delta_output_name}
                \tNode attribute output name: {args.node_att_output_name}
                \tOverwrite existing files: {args.overwrite}
                """
                )
    start_time = time.time()
    ret = main(args)
    end_time = time.time()
    message = "Execution succeeded." if ret == 0 else "Execution failed."
    logger.info(f"Execution completed in {end_time - start_time:.2f} seconds. {message}")
