from __future__ import annotations

from utils.datx import convert_datx_dataset
from utils.filter_utils import filter_dataset_names, process_dataset_names


def main() -> None:
    config = {
        "input_folder": "/data/optical_prof/Pristine",
        "output_folder": "/data/pristine_pe_processed",
        "datasets": ["intensity", "quality", "surface"],
        "save_dataset_metadata": True,
        "save_instrument_params": True,
    }

    template = "date_polymer_simulant_trial_replicate_magnification_level"
    filters = {
        "date": None,
        "polymer": None,
        "simulant": None,
        "trial": None,
        "replicate": None,
        "magnification": None,
        "level": None,
    }

    print("\nStep 1: Finding and filtering datx files")
    print("=" * 60)
    samples_to_process = process_dataset_names(config["input_folder"], template)
    filtered_samples = filter_dataset_names(samples_to_process, filters)

    print(f"\nFound {len(filtered_samples)} datx files matching filters:")
    for i, result in enumerate(filtered_samples[:10], 1):
        print(f"\n  {i}. {result['directory_name']}")
        print(f"     path: {result['directory_path']}")
        for key, value in result.items():
            if key not in ["directory_path", "directory_name"]:
                print(f"     {key}: {value}")

    if len(filtered_samples) > 10:
        print(f"\n  ... and {len(filtered_samples) - 10} more files")
    
    print("=" * 60)
    
    # Extract datx file paths from filtered samples
    datx_files = [sample['directory_path'] for sample in filtered_samples]
    
    print(f"\nStep 2: Converting {len(datx_files)} datx files")
    print("=" * 60)
    convert_datx_dataset(config, datx_files, n_workers=None)

    print(f"\nConversion complete!")
    print(f"Output directory: {config['output_folder']}")


if __name__ == "__main__":
    main()

