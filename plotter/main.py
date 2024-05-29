import argparse
import yaml

def gnn_plots(config_file):
    # Parse the YAML config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Your main logic here using the config
    print("Config file parsed successfully!")
    print("Config:", config)

def main():
    parser = argparse.ArgumentParser(description="GNN Plots")
    parser.add_argument("--config", help="Path to the YAML config file", required=True)
    args = parser.parse_args()

    gnn_plots(args.config)

if __name__ == "__main__":
    main()