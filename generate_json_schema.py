import json

from training.config import Config


def main():
    with open("./schemas/config.schema.json", "w") as f:
        data = Config.model_json_schema()
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
