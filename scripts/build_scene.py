"""Validate the local Isaac Sim scene contract before launching scene authoring."""

from aerospace_painting.scene import SceneContract


def main() -> None:
    contract = SceneContract.from_environment()
    print(f"Base scene ready: {contract.base_scene}")
    print("Launch this script with Isaac Sim Python after resolving the official asset references.")
    print("See scenes/README.md for the composition contract and non-redistribution boundary.")


if __name__ == "__main__":
    main()
