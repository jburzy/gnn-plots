import argparse
import yaml
import sys
import importlib


def gnn_plots(config_file):

    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

    run_config = {}
    with open(config_file) as f:
        try:
            run_config = yaml.safe_load(f.read())
        except Exception as err:
            print("A YAML exception occurred:\n", err)
            sys.exit(1)

    for _, plot_config in run_config['plots'].items():
        if 'class' not in plot_config:
            raise ValueError("YAML configuration must contain a 'class' property")
        class_name = plot_config.pop('class')
        module_name, class_name = class_name.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot find class {class_name} in module {module_name}") from e
        
        plot_obj = cls(**plot_config)
        plot_obj.plot()


def main():
    parser = argparse.ArgumentParser(description="GNN Plots")
    parser.add_argument("--config", help="Path to the YAML config file", required=True)
    args = parser.parse_args()

    gnn_plots(args.config)

if __name__ == "__main__":
    main()