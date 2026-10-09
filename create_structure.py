import os

# Define the project folder structure
structure = {
    "kinbech": {
        "files": [
            "app.py",
            "config.py",
            "data_preprocessing.py",
            "model_training.py",
        ],
        "folders": {
            "models": [
                "random_forest_model.pkl",
                "scaler.pkl",
            ],
            "templates": [
                "base.html",
                "index.html",
                "predict.html",
                "results.html",
                "dashboard.html",
            ],
            "static": {
                "css": ["style.css"],
                "js": ["script.js"],
            },
        },
    }
}


def create_structure(base_path, structure):
    for name, content in structure.items():
        folder_path = os.path.join(base_path, name)
        os.makedirs(folder_path, exist_ok=True)

        for file_name in content.get("files", []):
            file_path = os.path.join(folder_path, file_name)
            open(file_path, "a").close()

        for item, subcontent in content.get("folders", {}).items():
            if isinstance(subcontent, dict):
                create_structure(
                    folder_path,
                    {item: subcontent}
                )
            else:
                subfolder_path = os.path.join(folder_path, item)
                os.makedirs(subfolder_path, exist_ok=True)

                for file_name in subcontent:
                    file_path = os.path.join(
                        subfolder_path, file_name
                    )
                    open(file_path, "a").close()

    print(f"Project structure created at: {os.path.abspath(base_path)}")


if __name__ == "__main__":
    create_structure(".", structure)