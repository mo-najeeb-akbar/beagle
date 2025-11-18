from __future__ import annotations

from pathlib import Path

from datx import convert_datx_dataset
from filter_utils import filter_dataset_names, process_dataset_names


def main() -> None:
    config = {
        "input_folder": "/data/optical_prof/Defective",
        "output_folder": "/data/defective_pe_processed",
        "datasets": ["intensity", "quality", "surface"],
        "save_dataset_metadata": True,
        "save_instrument_params": True,
    }

    template = "date_polymer_simulant_trial_replicate_magnification_level"
    filters = {
        "date": None,
        "polymer": "PE",
        "simulant": None,
        "trial": None,
        "replicate": None,
        "magnification": None,
        "level": None,
    }

    print("Step 1: Converting DATX files to NPY format")
    print("=" * 60)
    convert_datx_dataset(config, n_workers=None)

    print("\nStep 2: Filtering processed directories")
    print("=" * 60)
    samples_to_process = process_dataset_names(config["output_folder"], template)
    filtered_samples = filter_dataset_names(samples_to_process, filters)

    print(f"\nFound {len(filtered_samples)} directories matching filters:")
    for i, result in enumerate(filtered_samples[:10], 1):
        print(f"\n  {i}. {result['directory_name']}")
        for key, value in result.items():
            if key not in ["directory_path", "directory_name"]:
                print(f"     {key}: {value}")

    if len(filtered_samples) > 10:
        print(f"\n  ... and {len(filtered_samples) - 10} more directories")

    print(f"\nConversion complete!")
    print(f"Output directory: {config['output_folder']}")


if __name__ == "__main__":
    main()

